import numpy as np
from typing import Dict, List, Union, Tuple
import torch
from pydub import AudioSegment
from ...speechxai_utils import pydub_to_np


class ModelHelperFSC:
    """
    Wrapper class to interface with HuggingFace models
    """

    def __init__(self, model, feature_extractor, device, language="en"):
        self.model = model
        self.feature_extractor = feature_extractor
        self.device = device
        self.n_labels = 3  # Multi label problem
        self.language = language
        self.label_name = ["action", "object", "location"]

    # PREDICT SINGLE
    def predict(
        self,
        audios: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predicts action, object and location from audio one sample at a time.
        Returns probs for each class.
        # This is to fix the bug of padding
        """

        action_probs = np.empty((len(audios), 6))
        object_probs = np.empty((len(audios), 14))
        location_probs = np.empty((len(audios), 4))
        for e, audio in enumerate(audios):
            action_probs[e], object_probs[e], location_probs[e] = self._predict([audio])
        return action_probs, object_probs, location_probs

    def predict_action(
        self,
        audios: List[np.ndarray],
    ):
        action_probs = np.empty((len(audios), 6))
        for e, audio in enumerate(audios):
            action_probs[e], _, _ = self._predict([audio])
        return action_probs

    def predict_object(
        self,
        audios: List[np.ndarray],
    ):
        object_probs = np.empty((len(audios), 14))
        for e, audio in enumerate(audios):
            _, object_probs[e], _ = self._predict([audio])
        return object_probs

    def predict_location(
        self,
        audios: List[np.ndarray],
    ):
        location_probs = np.empty((len(audios), 4))
        for e, audio in enumerate(audios):
            _, _, location_probs[e] = self._predict([audio])
        return location_probs

    def get_prediction_function_by_label(self, label):
        if label == 0:
            return self.predict_action
        elif label == 1:
            return self.predict_object
        elif label == 2:
            return self.predict_location
        else:
            raise ValueError("label should be 0, 1 or 2")

    def _predict(
        self,
        audios: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predicts action, object and location from audio.
        Returns probs for each class.
        """

        ## Extract features
        inputs = self.feature_extractor(
            [audio.squeeze() for audio in audios],
            sampling_rate=self.feature_extractor.sampling_rate,
            padding=True,
            return_tensors="pt",
        )

        ## Predict logits
        with torch.no_grad():
            logits = (
                self.model(inputs.input_values.to(self.device))
                .logits.detach()
                .cpu()
                # .numpy()
            )
            action_logits = logits[:, :6]
            object_logits = logits[:, 6:20]
            location_logits = logits[:, 20:24]

        return (
            action_logits.softmax(-1).numpy(),
            object_logits.softmax(-1).numpy(),
            location_logits.softmax(-1).numpy(),
        )

    def get_logits_action(self, input_embeds):
        logits = self.model(input_embeds.to(self.device)).logits
        logits = logits[:, :6]
        return logits

    def get_logits_object(self, input_embeds):
        logits = self.model(input_embeds.to(self.device)).logits
        logits = logits[:, 6:20]
        return logits

    def get_logits_location(self, input_embeds):
        logits = self.model(input_embeds.to(self.device)).logits
        logits = logits[:, 20:24]
        return logits

    def get_logits_function_from_input_embeds_by_label(self, label):
        if label == 0:
            return self.get_logits_action
        elif label == 1:
            return self.get_logits_object
        elif label == 2:
            return self.get_logits_location
        else:
            raise ValueError("label should be 0, 1 or 2")

    def get_text_labels(self, targets) -> Tuple[str, str, str]:
        action_ind, object_ind, location_ind = targets
        return (
            self.model.config.id2label[action_ind],
            self.model.config.id2label[object_ind + 6],
            self.model.config.id2label[location_ind + 20],
        )

    def get_text_labels_with_class(self, targets) -> Tuple[str, str, str]:
        """
        Return the text labels with the class name as strings (e.g., ['action = increase', 'object = lights', 'location = kitchen']])
        """
        text_targets = self.get_text_labels(targets)
        label_and_target_class_names = [
            f"{label}={target_class_name}"
            for label, target_class_name in zip(self.label_name, text_targets)
        ]
        return label_and_target_class_names

    def get_predicted_classes(self, audio_path=None, audio=None):
        if audio is None and audio_path is None:
            raise ValueError("Specify the audio path or the audio as a numpy array")

        if audio is None:
            audio = pydub_to_np(AudioSegment.from_wav(audio_path))[0]

        logits_action, logits_object, logits_location = self.predict([audio])
        action_ind = np.argmax(logits_action, axis=1)[0]
        object_ind = np.argmax(logits_object, axis=1)[0]
        location_ind = np.argmax(logits_location, axis=1)[0]
        return action_ind, object_ind, location_ind

    def get_predicted_probs(self, audio_path=None, audio=None):
        if audio is None and audio_path is None:
            raise ValueError("Specify the audio path or the audio as a numpy array")

        if audio is None:
            audio = pydub_to_np(AudioSegment.from_wav(audio_path))[0]

        logits_action, logits_object, logits_location = self.predict([audio])
        action_ind = np.argmax(logits_action, axis=1)[0]
        object_ind = np.argmax(logits_object, axis=1)[0]
        location_ind = np.argmax(logits_location, axis=1)[0]

        action_gt = logits_action[:, action_ind][0]
        object_gt = logits_object[:, object_ind][0]
        location_gt = logits_location[:, location_ind][0]
        return action_gt, object_gt, location_gt