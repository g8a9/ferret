"""LOO Speech Explainer module"""
import numpy as np
from typing import Dict, List, Union, Tuple
from pydub import AudioSegment
from IPython.display import display
from .explanation_speech import ExplanationSpeech
from .utils_removal import transcribe_audio, remove_word
from ...speechxai_utils import pydub_to_np, print_log


class LOOSpeechExplainer:
    NAME = "loo_speech"

    def __init__(self, model_helper):
        self.model_helper = model_helper

    def remove_words(
        self,
        audio_path: str,
        removal_type: str = "nothing",
        words_trascript: List = None,
        display_audio: bool = False,
    ) -> Tuple[List[AudioSegment], List[Dict[str, Union[str, float]]]]:
        """
        Remove words from audio using pydub, by replacing them with:
        - nothing
        - silence
        - white noise
        - pink noise
        """

        ## Transcribe audio

        if words_trascript is None:
            text, words_trascript = transcribe_audio(
                audio_path=audio_path,
                device=self.model_helper.device.type,
                batch_size=2,
                compute_type="float32",
                language=self.model_helper.language,
            )

        ## Load audio as pydub.AudioSegment
        audio = AudioSegment.from_wav(audio_path)

        ## Remove word
        audio_no_words = []

        for word in words_trascript:
            audio_removed = remove_word(audio, word, removal_type)

            audio_no_words.append(pydub_to_np(audio_removed)[0])

            if display_audio:
                print_log(word["word"])
                display(audio_removed)

        return audio_no_words, words_trascript

    def compute_explanation(
        self,
        audio_path: str,
        target_class=None,
        removal_type: str = None,
        words_trascript: List = None,
    ) -> ExplanationSpeech:
        """
        Computes the importance of each word in the audio.
        """

        ## Get modified audio by leaving a single word out and the words
        modified_audios, words = self.remove_words(
            audio_path, removal_type, words_trascript=words_trascript
        )

        logits_modified = self.model_helper.predict(modified_audios)

        audio = pydub_to_np(AudioSegment.from_wav(audio_path))[0]

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
                    np.argmax(logits_original[i], axis=1)[0] for i in range(n_labels)
                ]
            else:
                targets = np.argmax(logits_original, axis=1)[0]

        ## Get the most important word for each class (action, object, location)

        if n_labels > 1:
            # Multilabel scenario as for FSC
            modified_trg = [logits_modified[i][:, targets[i]] for i in range(n_labels)]
            original_gt = [
                logits_original[i][:, targets[i]][0] for i in range(n_labels)
            ]

        else:
            modified_trg = logits_modified[:, targets]
            original_gt = logits_original[:, targets][0]

        features = [word["word"] for word in words]

        if n_labels > 1:
            # Multilabel scenario as for FSC
            prediction_diff = [
                original_gt[i] - modified_trg[i] for i in range(n_labels)
            ]
        else:
            prediction_diff = [original_gt - modified_trg]

        scores = np.array(prediction_diff)

        explanation = ExplanationSpeech(
            features=features,
            scores=scores,
            explainer=self.NAME + "+" + removal_type,
            target=targets if n_labels > 1 else [targets],
            audio_path=audio_path,
        )

        return explanation
