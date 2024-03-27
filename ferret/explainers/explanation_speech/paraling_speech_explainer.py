"""Paralinguistic Speech Explainer module"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Tuple
from pydub import AudioSegment
from IPython.display import display
from audiomentations import (
    Compose,
    TimeStretch,
    PitchShift,
    RoomSimulator,
    AddBackgroundNoise,
    PolarityInversion,
)
from .explanation_speech import ExplanationSpeech
from ...speechxai_utils import pydub_to_np, FerretAudio
import torchaudio.functional as F
import torch
from audiostretchy.stretch import AudioStretch
import audio_effects
import tempfile
from io import BytesIO
import requests

from copy import deepcopy


# If True, We use the audiostretchy library to perform time stretching
USE_AUDIOSTRETCH = True

# If True, We use the add_noise of torch audio
USE_ADD_NOISE_TORCHAUDIO = True
REFERENCE_STR = "-"


ENDPOINTS = {
    "WHITE_NOISE": "https://github.com/g8a9/ferret/raw/feat/support-speech-from-array/ferret/explainers/explanation_speech/white_noise.mp3",
    "PINK_NOISE": "https://github.com/g8a9/ferret/raw/feat/support-speech-from-array/ferret/explainers/explanation_speech/pink_noise.mp3",
}


def _tmp_log1(
    verbose_target,
    original_gt,
    modified_trg,
    n_labels,
):
    if n_labels > 1:
        print("Target label: ", verbose_target)
        print("gt", original_gt[verbose_target])
        print("m", modified_trg[verbose_target])

    else:
        print("gt", original_gt)
        print("m", modified_trg)


# def _tmp_log2(
#     verbose_target,
#     original_gt,
#     modified_trg,
#     n_labels,
# ):
#     if n_labels > 1:
#         print_log(
#             [
#                 original_gt[verbose_target] - modified_trg[verbose_target][i]
#                 for i in range(modified_trg[verbose_target].shape[0])
#             ]
#         )

#     else:
#         print_log([original_gt - modified_trg[i] for i in range(modified_trg.shape[0])])


class ParalinguisticSpeechExplainer:
    NAME = "paralinguistic_explainer_speech"

    def __init__(self, model_helper):
        self.model_helper = model_helper

    def augmentation(
        self,
        perturbation_type: str,
        perturbation_value: float,
    ) -> List[np.ndarray]:
        """
        Creating a list of augmentations for each perturb_paraling type
        """
        if "pitch shifting" in perturbation_type:
            augment = Compose(
                [
                    PitchShift(
                        min_semitones=perturbation_value,
                        max_semitones=perturbation_value,
                        p=1.0,
                    ),
                ]
            )

        elif "time stretching" in perturbation_type:
            augment = Compose(
                [
                    TimeStretch(
                        min_rate=perturbation_value,
                        max_rate=perturbation_value,
                        p=1.0,
                        leave_length_unchanged=False,
                    ),
                ]
            )
        elif perturbation_type == "reverberation":
            augment = Compose(
                [
                    RoomSimulator(
                        min_size_x=perturbation_value,
                        max_size_x=perturbation_value,
                        min_size_y=perturbation_value,
                        max_size_y=perturbation_value,
                        min_size_z=perturbation_value,
                        max_size_z=perturbation_value,
                        padding=0.5,
                        min_absorption_value=0.1,
                        max_absorption_value=0.1,
                        p=1.0,
                    ),
                ]
            )
        elif perturbation_type == "noise":
            augment = Compose(
                [
                    AddBackgroundNoise(
                        sounds_path=os.path.join(
                            os.path.dirname(__file__), "white_noise.mp3"
                        ),
                        min_snr_in_db=perturbation_value,
                        max_snr_in_db=perturbation_value,
                        noise_transform=PolarityInversion(),
                        p=1.0,
                    )
                ]
            )
        return augment

    def time_stretching_augmentation(
        self, audio_as: AudioSegment, perturbation_value: float
    ):

        if perturbation_value < 1:
            perturbed_audio_as = audio_effects.speed_down(audio_as, perturbation_value)
        else:
            perturbed_audio_as = audio_as.speedup(perturbation_value)
        perturbed_audio, _ = pydub_to_np(perturbed_audio_as)
        return perturbed_audio.squeeze()

    def time_stretching_augmentation_AudioStretch(
        self, audio: FerretAudio, perturbation_value: float
    ):

        pydub_segment = audio.to_pydub()
        audio_stretch = AudioStretch()
        with tempfile.NamedTemporaryFile(suffix=".wav") as temp_audio:
            pydub_segment.export(temp_audio.name, format="wav")
            temp_audio.seek(0)

            audio_stretch.open(temp_audio.name)
            audio_stretch.stretch(ratio=perturbation_value)
            perturbated_audio_samples = np.array(audio_stretch.samples, dtype=np.float32)

        return perturbated_audio_samples

    def pitch_shifting_augmentation(
        self, audio_as: AudioSegment, perturbation_value: float
    ):
        # perturbation_value = octaves

        new_sample_rate = int(audio_as.frame_rate * (2.0**perturbation_value))

        perturbed_audio_as = audio_as._spawn(
            audio_as.raw_data, overrides={"frame_rate": new_sample_rate}
        )
        perturbed_audio_as = perturbed_audio_as.set_frame_rate(audio_as.frame_rate)
        perturbed_audio, _ = pydub_to_np(perturbed_audio_as)

        return perturbed_audio.squeeze()

    def add_white_noise_torchaudio(self, original_speech, noise_rate):
        """Args:
        original_speech: np.array of shape (1, n_samples)
        noise_rate: signal-to-noise ratios in dB
        """

        # WHITE_NOISE = os.path.join(os.path.dirname(__file__), "white_noise.mp3")
        # noise_as = AudioSegment.from_mp3(WHITE_NOISE)

        res = requests.get(ENDPOINTS["WHITE_NOISE"])
        noise_as = AudioSegment.from_file(BytesIO(res.content), "mp3")
        noise, frame_rate = pydub_to_np(noise_as)

        # Reshape and convert to torch tensor
        original_speech = torch.tensor(original_speech.reshape(1, -1))
        # Reshape and convert to torch tensor
        noise = torch.tensor(noise.reshape(1, -1))

        def extend_noise(noise, desired_length):
            noise_new = deepcopy(noise)
            while noise_new.shape[1] < desired_length:
                noise_new = torch.concat([noise_new[0], noise[0]]).reshape(1, -1)
            noise_new = noise_new[:, :desired_length]
            return noise_new

        noise_eq_length = extend_noise(noise, original_speech.shape[1])
        snr_dbs = torch.tensor([noise_rate, 10, 3])
        noisy_speeches = F.add_noise(original_speech, noise_eq_length, snr_dbs)
        noisy_speech = noisy_speeches[0:1].numpy()
        return noisy_speech

    def change_pitch_torchaudio(self, original_speech, frame_rate, perturbation_value):
        """Args:
        original_speech: np.array of shape (1, n_samples)
        perturbation_value:
        """

        # Reshape and convert to torch tensor
        audio_t = torch.tensor(original_speech.reshape(1, -1))
        perturbated_audio = F.pitch_shift(
            audio_t, frame_rate, n_steps=perturbation_value
        )
        perturbated_audio = perturbated_audio.numpy()
        return perturbated_audio

    def perturbe_waveform(
        self,
        audio: FerretAudio,
        perturbation_type: str,
        return_perturbations=False,
        verbose: bool = False,
        verbose_target: int = 0,
    ):
        """
        Perturbate audio using pydub, by adding:
        - pitch shifting
        - time stretching
        - reverberation
        - noise
        """

        ## Load audio as pydub.AudioSegment
        # audio_as = AudioSegment.from_wav(audio_path)
        # audio, frame_rate = pydub_to_np(audio_as)

        ## Perturbate audio
        perturbated_audios = []
        if perturbation_type == "pitch shifting":
            # perturbations = [-3.5, -2.5, -2, 2, 2.5, 3.5]
            # OK v2
            # perturbations = [-0.3, -0.2, -0.1, 0.1, 0.2, 0.3]
            perturbations = np.arange(-5, 5.5, 0.5)

        elif perturbation_type == "pitch shifting down":
            # perturbations = [-3.5, -2.5, -2]
            # OK v2
            # perturbations = [-0.3, -0.2, -0.1]
            perturbations = np.arange(-5, 0, 0.5)

        elif perturbation_type == "pitch shifting up":
            # perturbations = [2, 2.5, 3.5]
            # ok v2
            # perturbations = [0.1, 0.2, 0.3]
            perturbations = np.arange(0.5, 5.5, 0.5)

        elif perturbation_type == "time stretching":
            # perturbations = [0.75, 0.80, 0.85, 1.15, 1.20, 1.25]
            # perturbations = [0.75, 0.85, 0.9, 1.15, 1.25, 1.35, 1.5]
            perturbations = [
                #
                0.65,
                0.7,
                0.75,
                0.8,
                0.85,
                0.9,
                0.95,
                1.05,
                1.10,
                1.15,
                1.2,
                1.25,
                1.3,
                1.35,
            ]
            if USE_AUDIOSTRETCH:
                perturbations = [0.55, 0.6] + perturbations + [1.4, 1.45]

        elif perturbation_type == "time stretching down":
            # perturbations = [1.15, 1.20, 1.25]

            if USE_AUDIOSTRETCH:
                # For audio stretch it is the contrary..
                perturbations = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            else:
                perturbations = [1.05, 1.10, 1.15, 1.2, 1.25, 1.3, 1.35]

        elif perturbation_type == "time stretching up":
            if USE_AUDIOSTRETCH:
                # For audio stretch it is the contrary..
                perturbations = [1.05, 1.10, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45]
            else:
                perturbations = [0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

        elif perturbation_type == "reverberation":
            perturbations = [3, 4, 5, 6, 7]

        elif perturbation_type == "noise":
            perturbations = [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]
            if USE_ADD_NOISE_TORCHAUDIO:
                perturbations = [40, 20, 10, 7.5, 5, 4, 3, 2, 1, 0.5, 0.1]
        else:
            raise ValueError(f"Perturbation '{perturbation_type}' is not available")

        if verbose:

            # Display the original audio and show its info for a single class
            self._tmp_log_show_info(
                "Original audio",
                "",
                audio.squeeze(),
                verbose_target,
            )

        pydub_segment = audio.to_pydub()

        for perturbation_value in perturbations:
            if "time stretching" in perturbation_type:
                if USE_AUDIOSTRETCH:
                    perturbated_audio = self.time_stretching_augmentation_AudioStretch(
                        audio=audio, perturbation_value=perturbation_value
                    )
                else:
                    perturbated_audio = self.time_stretching_augmentation(
                        pydub_segment, perturbation_value
                    )
            elif "pitch shifting" in perturbation_type:
                # perturbated_audio = self.pitch_shifting_augmentation(
                #    audio_as, perturbation_value
                # )


                # Note: here we assume frame rate and sample rate are the
                #       same, which is always true for single-channel (mono)
                #       audio.
                perturbated_audio = self.change_pitch_torchaudio(
                    audio.normalized_array,
                    audio.current_sr,
                    perturbation_value,
                )

            elif perturbation_type == "noise" and USE_ADD_NOISE_TORCHAUDIO:
                perturbated_audio = self.add_white_noise_torchaudio(
                    audio.normalized_array, perturbation_value
                )
            else:
                augment = self.augmentation(
                    perturbation_value=perturbation_value,
                    perturbation_type=perturbation_type,
                )
                perturbated_audio = augment(
                    samples=audio.normalized_array.squeeze(),
                    sample_rate=audio.current_sr
                )

            if verbose:
                # Display the perturbated audio an show its info for a single class
                self._tmp_log_show_info(
                    perturbation_type,
                    perturbation_value,
                    perturbated_audio,
                    verbose_target,
                )
            perturbated_audios.append(perturbated_audio)

        if return_perturbations:
            return perturbated_audios, perturbations
        else:
            return perturbated_audios

    def compute_explanation(
        self,
        audio: FerretAudio,
        target_class=None,
        perturbation_type: str = None,
        verbose: bool = False,
        verbose_target: int = 0,
    ) -> ExplanationSpeech:
        """
        Computes the importance of each paralinguistic feature in the audio.
        """

        modified_audios = self.perturbe_waveform(
            audio,
            perturbation_type,
            verbose=verbose,
            verbose_target=verbose_target,
        )

        ## Get logits for each class

        logits_modified = self.model_helper.predict(modified_audios)

        # Note: we use the normalized array for consistency with the original
        #       SpeechXAI code (it used to come from the `pydub_to_np`
        #       function).
        logits_original = self.model_helper.predict([audio.normalized_array])

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

        ## Get the most important word for each label

        if n_labels > 1:
            # Multilabel scenario as for FSC
            modified_trg = [logits_modified[i][:, targets[i]] for i in range(n_labels)]
            original_gt = [logits_original[i][:, targets[i]][0] for i in range(n_labels)]

        else:
            modified_trg = logits_modified[:, targets]
            original_gt = logits_original[:, targets][0]

        if verbose:
            _tmp_log1(verbose_target, original_gt, modified_trg, n_labels)
            # _tmp_log2(verbose_target, original_gt, modified_trg, n_labels)

        ## Compute the difference between the ground truth and the modified audio
        # prediction_diff = original_gt - np.mean(modified_trg)

        if n_labels > 1:
            # Multilabel scenario as for FSC
            prediction_diff = [
                original_gt[i] - np.mean(modified_trg[i]) for i in range(n_labels)
            ]
        else:
            prediction_diff = [original_gt - np.mean(modified_trg)]

        features = [perturbation_type]

        scores = np.array(prediction_diff)

        explanation = ExplanationSpeech(
            features=features,
            scores=scores,
            explainer=self.NAME,
            target=targets if n_labels > 1 else [targets],
            audio=audio,
        )

        return explanation

    def explain_variations(
        self, audio: FerretAudio, perturbation_types: List[int], target_class=None
    ):
        n_labels = self.model_helper.n_labels

        audio_array = audio.normalized_array

        original_gt = self.model_helper.get_predicted_probs(audio=audio_array)

        if target_class is None:
            targets = self.model_helper.get_predicted_classes(audio=audio_array)
        else:
            targets = target_class

        target_classes_show = self.model_helper.get_text_labels(targets)

        perturbation_df_by_type = {}
        for perturbation_type in perturbation_types:
            perturbated_audios, perturbations = self.perturbe_waveform(
                audio, perturbation_type, return_perturbations=True
            )

            if "time stretching" in perturbation_type:
                reference_value = 1
            else:
                reference_value = 0
                if "noise" == perturbation_type and USE_ADD_NOISE_TORCHAUDIO:
                    reference_value = 100

            prob_variations = []
            for perturbated_audio in perturbated_audios:
                probs_modified = self.model_helper.predict([perturbated_audio])

                if n_labels > 1:
                    # Multilabel scenario as for FSC
                    prob_variations.append(
                        [probs_modified[i][:, targets[i]][0] for i in range(n_labels)]
                    )
                else:
                    prob_variations.append([probs_modified[:, targets][0]])

            if n_labels > 1:
                prob_variations.append(original_gt)
            else:
                prob_variations.append([original_gt])

            x_labels = perturbations + [reference_value]
            prob_variations = np.array(prob_variations)[np.argsort(x_labels)]
            x_labels = np.array(x_labels)[np.argsort(x_labels)]

            if perturbation_type == "noise" and USE_ADD_NOISE_TORCHAUDIO:
                x_labels = x_labels[::-1]
                prob_variations = prob_variations[::-1]

            x_labels = [
                x_label if x_label != reference_value else REFERENCE_STR
                for x_label in x_labels
            ]

            perturbation_df = pd.DataFrame(prob_variations.T, columns=x_labels)
            perturbation_df.index = (
                target_classes_show if n_labels > 1 else [target_classes_show]
            )
            perturbation_df_by_type[perturbation_type] = perturbation_df
        return perturbation_df_by_type

    def _tmp_log_show_info(
        self,
        perturbation_type: str,
        perturbation_value: float,
        perturbated_audio: np.ndarray,
        verbose_target: int,
    ):
        from IPython.display import Audio

        # Display the perturbated audio an show its info for a single class
        # For multi label scenario, we show it for a single class: verbose_target

        # Note that in a single label scenario, verbose_target is ignored (always 0)

        print(perturbation_type, perturbation_value)
        # Prediction probability
        predictions = self.model_helper.predict([perturbated_audio])

        # Predicted label (idx)
        predicted_labels = [v.argmax() for v in predictions]

        # Predicted label (text)
        preds = self.model_helper.get_text_labels(predicted_labels)

        if self.model_helper.n_labels > 1:
            print(f"Target label: {verbose_target}")
            print(
                f"Predicted probs:",
                np.round(predictions[verbose_target], 3),
            )
            print(
                "Predicted class: ",
                preds[verbose_target],
                f"id: {predicted_labels[verbose_target]}",
            )
        else:
            print(
                f"Predicted probs: ",
                np.round(predictions[0], 3),
            )
            print(
                "Predicted class: ",
                preds,
                f"id: {predicted_labels[0]}",
            )
        display(Audio(perturbated_audio, rate=16000))
