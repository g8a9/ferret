from typing import List
from pydub import AudioSegment
import numpy as np
from ..lime_timeseries import LimeTimeSeriesExplainer
from ..explanation_speech import ExplanationSpeech
from ....speechxai_utils import FerretAudio

EMPTY_SPAN = "---"


class LIMEEqualWidthSpeechExplainer:
    NAME = "LIME_equal_width"

    def __init__(self, model_helper):
        self.model_helper = model_helper

    def compute_explanation(
        self,
        audio: FerretAudio,
        target_class=None,
        removal_type: str = "silence",
        num_samples: int = 1000,
        num_s_split: float = 0.25,
    ) -> ExplanationSpeech:
        """
        Compute the word-level explanation for the given audio.
        audio: An instance of the FerretAudio class containing the input audio data.
        target_class: target class - int - If None, use the predicted class
        removal_type:
        """

        if removal_type not in ["silence", "noise"]:
            raise ValueError(
                "Removal method not supported, choose between 'silence' and 'noise'"
            )

        audio_np = audio.normalized_array

        # Predict logits/probabilities
        logits_original = self.model_helper.predict([audio_np])

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

        # Get the start and end indexes of the segments. These will be used to split the audio and derive LIME interpretable features
        sampling_rate = self.model_helper.feature_extractor.sampling_rate
        splits = []

        duration_s = len(audio_np) / audio.sample_rate 

        a, b = 0, 0
        for e, i in enumerate(np.arange(0, duration_s, num_s_split)):
            start_s = i
            end_s = min(i + num_s_split, duration_s)

            start, end = int((start_s + a) * sampling_rate), int(
                (end_s + b) * sampling_rate
            )
            splits.append({"start": start, "end": end, "word": e})

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

            # WARNING: this is the original reshaping, which assumes that
            #          `LimeTimeSeriesExplainer` accepts an array with shape
            #           (1, n_samples).
            input_audio = deepcopy(audio_np.reshape(1, -1))

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
                for k, v in sorted(
                    map_scores.items(), key=lambda x: x[0], reverse=False
                )
            }

            # Remove the 'empty' spans, the spans between words
            map_scores = [
                (splits[k]["word"], v)
                for k, v in map_scores.items()
                if splits[k]["word"] != EMPTY_SPAN
            ]

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
        )

        return explanation