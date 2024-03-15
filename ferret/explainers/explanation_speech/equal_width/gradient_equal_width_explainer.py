from typing import List
from pydub import AudioSegment
from captum.attr import Saliency, InputXGradient
import numpy as np
import torch
from ..explanation_speech import ExplanationSpeech
from ....speechxai_utils import pydub_to_np


class GradientEqualWidthSpeechExplainer:
    NAME = "Gradient_equal_width"

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
        audio_path: str,
        target_class=None,
        aggregation: str = "mean",
        num_s_split: float = 0.25,
    ) -> ExplanationSpeech:
        """
        Compute the word-level explanation for the given audio.
        Args:
        audio_path: path to the audio file
        target_class: target class - int - If None, use the predicted class
        no_before_span: if True, it also consider the span before the word. This is because we observe gradient give importance also for the frame just before the word
        aggregation: aggregation method for the frames of the word. Can be "mean" or "max"
        num_s_split: float = number of seconds of each audio segment in which to split the audio,
        """

        if aggregation not in ["mean", "max"]:
            raise ValueError(
                "Aggregation method not supported, choose between 'mean' and 'max'"
            )

        # Load audio and convert to np.array
        audio_as = AudioSegment.from_wav(audio_path)
        audio = pydub_to_np(audio_as)[0]

        # Predict logits/probabilities
        logits_original = self.model_helper.predict([audio])

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

        # Compute gradient importance for each target label
        # This also handles the multilabel scenario as for FSC
        scores = []
        for target_label, target_class in enumerate(targets):
            # Get gradient importance for each frame
            attr = self._get_gradient_importance_frame_level(
                audio, target_class, target_label
            )

            old_start = 0
            old_start_ms = 0
            features = []
            importances = []
            a, b = 0, 0  # 50, 20

            duration_s = len(audio_as) / 1000

            a, b = 0, 0
            for e, i in enumerate(np.arange(0, duration_s, num_s_split)):
                start = i
                end = min(i + num_s_split, duration_s)

                start_ms = (start * 1000 - a) / 1000
                end_ms = (end * 1000 + b) / 1000

                start, end = int(
                    start_ms * self.model_helper.feature_extractor.sampling_rate
                ), int(end_ms * self.model_helper.feature_extractor.sampling_rate)

                # Slice of the importance for the given word
                segment_importance = attr[start:end]

                # Consider also the spans between words
                # #span_before = attr[old_start:start]

                if aggregation == "max":
                    segment_importance = np.max(segment_importance)
                else:
                    segment_importance = np.mean(segment_importance)

                old_start = end
                old_start_ms = end_ms
                importances.append(segment_importance)
                features.append(e)

            scores.append(np.array(importances))

        if n_labels > 1:
            # Multilabel scenario as for FSC
            scores = np.array(scores)
        else:
            scores = np.array([importances])

        explanation = ExplanationSpeech(
            features=features,
            scores=scores,
            explainer=self.NAME + "-" + aggregation,
            target=targets if n_labels > 1 else targets,
            audio_path=audio_path,
        )

        return explanation