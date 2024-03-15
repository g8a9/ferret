"""LOO Speech Explainer module"""
import os
import numpy as np
from typing import Dict, List, Union, Tuple
import whisperx
from pydub import AudioSegment
from IPython.display import display
from ..explanation_speech import ExplanationSpeech
from ....speechxai_utils import pydub_to_np, print_log, FerretAudio


def remove_audio_segment(audio, start_s, end_s, removal_type: str = "silence"):
    """
    Remove an audio segment from audio using pydub, by replacing it with:
    - nothing
    - silence
    - white noise
    - pink noise

    Args:
        audio (pydub.AudioSegment): audio
        word: word to remove with its start and end times
        removal_type (str, optional): type of removal. Defaults to "nothing".
    """

    start_idx = int(start_s * 1000)
    end_idx = int(end_s * 1000)
    before_word_audio = audio[:start_idx]
    after_word_audio = audio[end_idx:]
    word_duration = end_idx - start_idx

    if removal_type == "nothing":
        replace_word_audio = AudioSegment.empty()
    elif removal_type == "silence":
        replace_word_audio = AudioSegment.silent(duration=word_duration)

    elif removal_type == "white noise":
        sound_path = (os.path.join(os.path.dirname(__file__), "white_noise.mp3"),)
        replace_word_audio = AudioSegment.from_mp3(sound_path)[:word_duration]

        # display(audio_removed)
    elif removal_type == "pink noise":
        sounds_path = (os.path.join(os.path.dirname(__file__), "pink_noise.mp3"),)
        replace_word_audio = AudioSegment.from_mp3(sound_path)[:word_duration]

    audio_removed = before_word_audio + replace_word_audio + after_word_audio
    return audio_removed
class LOOSpeechEqualWidthExplainer:
    NAME = "loo_speech_equal_width"

    def __init__(self, model_helper):
        self.model_helper = model_helper

    def compute_explanation(
        self,
        audio: FerretAudio,
        target_class=None,
        removal_type: str = "silence",
        num_s_split: float = 0.25,
        display_audio: bool = False,
    ) -> ExplanationSpeech:
        """
        Computes the importance of each equal width audio segment in the audio.
        """

        audio_array = audio.array

        ## Remove word
        audio_remove_segments = []

        duration_s = len(audio_array) / audio.sample_rate # finds the duration from the array 

        for i in np.arange(0, duration_s, num_s_split):
            start_s = i
            end_s = min(i + num_s_split, duration_s)
            audio_removed = remove_audio_segment(audio.to_pydub(), start_s, end_s, removal_type)

            audio_remove_segments.append(pydub_to_np(audio_removed)[0])

            if display_audio:
                print_log(int(start_s / num_s_split), start_s, end_s)
                display(audio_removed)

        # Get original logits
        logits_original = self.model_helper.predict([audio_array])

        # Get logits for the modified audio by leaving out the equal width segments
        logits_modified = self.model_helper.predict(audio_remove_segments)

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

        features = [idx for idx in range(len(audio_remove_segments))]

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
            audio=audio,
        )

        return explanation