"""LOO Speech Explainer module"""

import numpy as np
from typing import Dict, List, Union, Tuple
from pydub import AudioSegment
from IPython.display import display
from .explanation_speech import ExplanationSpeech
from .utils_removal import remove_word
from ...speechxai_utils import pydub_to_np, FerretAudio
from logging import getLogger

logger = getLogger(__name__)


class LOOSpeechExplainer:
    NAME = "loo_speech"

    def __init__(self, model_helper):
        self.model_helper = model_helper

    def remove_words(
        self,
        audio: FerretAudio,
        word_timestamps: List,
        removal_type: str = "nothing",
        display_audio: bool = False,
    ) -> Tuple[List[AudioSegment], List[Dict[str, Union[str, float]]]]:
        """
        Remove words from audio using pydub, by replacing them with:
        - nothing
        - silence
        - white noise
        - pink noise
        """

        ## Load audio as pydub.AudioSegment
        pydub_segment = audio.to_pydub()

        ## Remove word
        audio_no_words = list()

        for word in word_timestamps:
            audio_removed = remove_word(pydub_segment, word, removal_type)

            # Note: we might potentially put `audio_removed` into a
            #       `FerretAudio` object, but it'd be an additional step.
            audio_no_words.append(pydub_to_np(audio_removed)[0])

            if display_audio:
                print(word["word"])
                display(audio_removed)

        return audio_no_words, word_timestamps

    def compute_explanation(
        self,
        audio: FerretAudio,
        target_class=None,
        removal_type: str = None,
        word_timestamps: List = None,
    ) -> ExplanationSpeech:
        """
        Computes the importance of each word in the audio.
        """

        ## Get modified audio by leaving a single word out and the words
        modified_audios, words = self.remove_words(
            audio=audio, word_timestamps=word_timestamps, removal_type=removal_type
        )

        logits_modified = self.model_helper.predict(modified_audios)

        # Note: we use the normalized array for consistency with the original
        #       SpeechXAI code (it used to come from the `pydub_to_np`
        #       function).
        audio_array = audio.normalized_array

        logits_original = self.model_helper.predict([audio_array])

        # Check if single label or multilabel scenario as for FSC
        n_labels = self.model_helper.n_labels

        # TODO
        # TODO GA: what?
        if target_class is not None:
            targets = target_class

        else:
            if n_labels > 1:
                # Multilabel scenario as for FSC
                targets = [
                    np.argmax(logits_original[i], axis=1)[0] for i in range(n_labels)
                ]
            else:
                targets = np.argmax(logits_original, axis=1)[0]

        ## Get the most important word for each class (action, object, location)

        if n_labels > 1:
            # Multilabel scenario as for FSC
            modified_trg = [logits_modified[i][:, targets[i]] for i in range(n_labels)]
            original_gt = [logits_original[i][:, targets[i]][0] for i in range(n_labels)]

        else:
            modified_trg = logits_modified[:, targets]
            original_gt = logits_original[:, targets][0]

        features = [word["word"] for word in words]

        if n_labels > 1:
            # Multilabel scenario as for FSC
            prediction_diff = [original_gt[i] - modified_trg[i] for i in range(n_labels)]
        else:
            prediction_diff = [original_gt - modified_trg]

        scores = np.array(prediction_diff)

        explanation = ExplanationSpeech(
            features=features,
            scores=scores,
            explainer=self.NAME + "+" + removal_type,
            target=targets if n_labels > 1 else [targets],
            audio=audio,  # TODO GA: I don't know if this is something we want to keep
            word_timestamps=word_timestamps,
        )

        return explanation
