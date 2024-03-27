from typing import List
from pydub import AudioSegment
import numpy as np
from .lime_timeseries import LimeTimeSeriesExplainer
from .explanation_speech import ExplanationSpeech
from ...speechxai_utils import FerretAudio

EMPTY_SPAN = "---"


class LIMESpeechExplainer:
    NAME = "LIME"

    def __init__(self, model_helper):
        self.model_helper = model_helper

    def compute_explanation(
        self,
        audio: FerretAudio,
        word_timestamps: List,
        target_class=None,
        removal_type: str = "silence",
        num_samples: int = 1000,
    ) -> ExplanationSpeech:
        """
        Compute the word-level explanation for the given audio.
        Args:
        audio: An instance of the FerretAudio class containing the input audio data.
        target_class: target class - int - If None, use the predicted class
        removal_type:
        """

        if removal_type not in ["silence", "noise"]:
            raise ValueError(
                "Removal method not supported, choose between 'silence' and 'noise'"
            )

        # Note: we use the normalized array for consistency with the original
        #       SpeechXAI code (it used to come from the `pydub_to_np`
        #       function).
        audio_array = audio.normalized_array

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

        # Get the start and end indexes of the words. These will be used to split the audio and derive LIME interpretable features
        tot_len = audio_array.shape[0]
        sampling_rate = self.model_helper.feature_extractor.sampling_rate
        splits = []
        old_start = 0
        a, b = 0, 0
        for word in word_timestamps:
            start, end = int((word["start"] + a) * sampling_rate), int(
                (word["end"] + b) * sampling_rate
            )
            splits.append({"start": old_start, "end": start, "word": EMPTY_SPAN})
            splits.append({"start": start, "end": end, "word": word["word"]})
            old_start = end
        splits.append({"start": old_start, "end": tot_len, "word": EMPTY_SPAN})

        lime_explainer = LimeTimeSeriesExplainer()

        # Compute gradient importance for each target label
        # This also handles the multilabel scenario as for FSC
        scores = []
        for target_label, target_class in enumerate(targets):
            if self.model_helper.n_labels > 1:
                # We get the prediction probability for the given label
                predict_proba_function = (
                    self.model_helper.get_prediction_function_by_label(target_label)
                )
            else:
                predict_proba_function = self.model_helper.predict
            from copy import deepcopy

            input_audio = deepcopy(audio_array.reshape(1, -1))

            # Explain the instance using the splits as interpretable features
            exp = lime_explainer.explain_instance(
                input_audio,
                predict_proba_function,
                num_features=len(splits),
                num_samples=num_samples,
                num_slices=len(splits),
                replacement_method=removal_type,
                splits=splits,
                labels=(target_class,),
            )

            map_scores = {k: v for k, v in exp.as_map()[target_class]}
            map_scores = {
                k: v
                for k, v in sorted(map_scores.items(), key=lambda x: x[0], reverse=False)
            }

            # Remove the 'empty' spans, the spans between words
            map_scores = [
                (splits[k]["word"], v)
                for k, v in map_scores.items()
                if splits[k]["word"] != EMPTY_SPAN
            ]
            if map_scores == []:
                features = []
                importances = []
            else:
                features = list(list(zip(*map_scores))[0])
                importances = list(list(zip(*map_scores))[1])
            scores.append(np.array(importances))

        if n_labels > 1:
            # Multilabel scenario as for FSC
            scores = np.array(scores)
        else:
            scores = np.array([importances])

        explanation = ExplanationSpeech(
            features=features,
            scores=scores,
            explainer=self.NAME + "+" + removal_type,
            target=targets if n_labels > 1 else targets,
            audio=audio,
            word_timestamps=word_timestamps,
        )

        return explanation
