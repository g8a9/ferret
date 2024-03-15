from typing import List
from pydub import AudioSegment
from captum.attr import Saliency, InputXGradient
import numpy as np
import torch
from .explanation_speech import ExplanationSpeech
from ...speechxai_utils import pydub_to_np, FerretAudio
# TODO - include in utils

class GradientSpeechExplainer:
    NAME = "Gradient"

    def __init__(self, model_helper, multiply_by_inputs: bool = False):
        self.model_helper = model_helper
        self.multiply_by_inputs = multiply_by_inputs

        if self.multiply_by_inputs:
            self.NAME += " (x Input)"

    def _get_gradient_importance_frame_level(
        self, audio, target_class, target_label=None
    ):
        """
        Compute the gradient importance for each frame of the audio w.r.t. the target class.
        Args:
            audio: audio - np.array
            target_class: target class - int
            target_label: target label - int - Used only in a multilabel scenario as for FSC
        """
        torch.set_grad_enabled(True)  # Context-manager

        # Function which returns the logits
        if self.model_helper.n_labels > 1:
            # We get the logits for the given label
            func = self.model_helper.get_logits_function_from_input_embeds_by_label(
                target_label
            )
        else:
            func = self.model_helper.get_logits_from_input_embeds

        dl = InputXGradient(func) if self.multiply_by_inputs else Saliency(func)

        inputs = self.model_helper.feature_extractor(
            [audio_i.squeeze() for audio_i in [audio]],
            sampling_rate=self.model_helper.feature_extractor.sampling_rate,
            padding=True,
            return_tensors="pt",
        )
        input_len = inputs["attention_mask"].sum().item()
        attr = dl.attribute(inputs.input_values, target=target_class)
        attr = attr[0, :input_len].detach().cpu()

        # pool over hidden size
        attr = attr.numpy()
        return attr

    def compute_explanation(
        self,
        audio: FerretAudio,
        target_class=None,
        words_trascript: List = None,
        no_before_span: bool = True,
        aggregation: str = "mean",
    ) -> ExplanationSpeech:
        """
        Compute the word-level explanation for the given audio.
        Args:
        audio: An instance of the FerretAudio class containing the input audio data.
        target_class: target class - int - If None, use the predicted class
        no_before_span: if True, it also consider the span before the word. This is because we observe gradient give importance also for the frame just before the word
        aggregation: aggregation method for the frames of the word. Can be "mean" or "max"
        """

        if aggregation not in ["mean", "max"]:
            raise ValueError(
                "Aggregation method not supported, choose between 'mean' and 'max'"
            )

        # Load audio and convert to np.array
        audio_array = audio.array

        # Predict logits/probabilities
        logits_original = self.model_helper.predict([audio_array])

        # Check if single label or multilabel scenario as for FSC
        n_labels = self.model_helper.n_labels

        # TODO
        if target_class is not None:
            targets = target_class

        else:
            if n_labels > 1:
                # Multilabel scenario as for FSC
                targets = [
                    int(np.argmax(logits_original[i], axis=1)[0])
                    for i in range(n_labels)
                ]
            else:
                targets = [int(np.argmax(logits_original, axis=1)[0])]

        if words_trascript is None:
            # Transcribe audio
            words_trascript = audio.transcription

        # Compute gradient importance for each target label
        # This also handles the multilabel scenario as for FSC
        scores = []
        for target_label, target_class in enumerate(targets):
            # Get gradient importance for each frame
            attr = self._get_gradient_importance_frame_level(
                audio_array, target_class, target_label
            )

            old_start = 0
            old_start_ms = 0
            features = []
            importances = []
            a, b = 0, 0  # 50, 20

            for word in words_trascript:
                if no_before_span:
                    # We directly consider the transcribed word
                    start_ms = (word["start"] * 1000 - a) / 1000
                    end_ms = (word["end"] * 1000 + b) / 1000

                else:
                    # We also include the frames before the word
                    start_ms = old_start_ms
                    end_ms = (word["end"] * 1000) / 1000

                start, end = int(
                    start_ms * self.model_helper.feature_extractor.sampling_rate
                ), int(end_ms * self.model_helper.feature_extractor.sampling_rate)

                # Slice of the importance for the given word
                word_importance = attr[start:end]

                # Consider also the spans between words
                # #span_before = attr[old_start:start]

                if aggregation == "max":
                    word_importance = np.max(word_importance)
                else:
                    word_importance = np.mean(word_importance)

                old_start = end
                old_start_ms = end_ms
                importances.append(word_importance)
                features.append(word["word"])

            # Consider also the spans between words
            # importances.append(np.mean(span_before))
            # features.append('-')

            # Consider also the spans between words
            # Final span
            # final_span = attr[old_start:len(audio_np)]
            # features.append('-')

            # if aggregation == "max":
            #    importances.append(np.max(final_span))
            # else:
            #    importances.append(np.mean(final_span))
            scores.append(np.array(importances))

        if n_labels > 1:
            # Multilabel scenario as for FSC
            scores = np.array(scores)
        else:
            scores = np.array([importances])

        features = [word["word"] for word in words_trascript]

        explanation = ExplanationSpeech(
            features=features,
            scores=scores,
            explainer=self.NAME + "-" + aggregation,
            target=targets if n_labels > 1 else targets,
            audio=audio,
        )

        return explanation