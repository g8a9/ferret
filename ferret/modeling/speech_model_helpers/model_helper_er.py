import numpy as np
from typing import Dict, List, Union, Tuple
from pydub import AudioSegment
import torch
from ...speechxai_utils import pydub_to_np



class ModelHelperER:
    """
    Wrapper class to interface with HuggingFace models
    """

    def __init__(self, model, feature_extractor, device, language="en"):
        self.model = model
        self.feature_extractor = feature_extractor
        self.device = device
        self.n_labels = 1  # Single label problem
        self.language = language
        self.label_name = "class"

    # PREDICT SINGLE
    def predict(
        self,
        audios: List[np.ndarray],
    ) -> np.ndarray:
        """
        Predicts action, object and location from audio one sample at a time.
        Returns probs for each class.
        # We do separately for consistency with FSC/IC model and the bug of padding
        """

        probs = np.empty((len(audios), self.model.config.num_labels))
        for e, audio in enumerate(audios):
            probs[e] = self._predict([audio])
        return probs

    def _predict(
        self,
        audios: List[np.ndarray],
    ) -> np.ndarray:
        """
        Predicts emotion from audio.
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
            logits = logits

        return logits.softmax(-1).numpy()

    def get_text_labels(self, targets) -> str:
        if type(targets) is list:
            class_index = targets[0]
        else:
            class_index = targets
        return self.model.config.id2label[class_index]

    def get_text_labels_with_class(self, targets) -> str:
        """
        Return the text labels with the class name as strings (e.g., ['action = increase', 'object = lights', 'location = kitchen']])
        """
        text_target = self.get_text_labels(targets)
        return f"{self.label_name}={text_target}"

    def get_predicted_classes(self, audio_path=None, audio=None):
        if audio is None and audio_path is None:
            raise ValueError("Specify the audio path or the audio as a numpy array")

        if audio is None:
            audio = pydub_to_np(AudioSegment.from_wav(audio_path))[0]

        logits = self.predict([audio])
        predicted_ids = np.argmax(logits, axis=1)[0]
        return predicted_ids

    def get_predicted_probs(self, audio_path=None, audio=None):
        if audio is None and audio_path is None:
            raise ValueError("Specify the audio path or the audio as a numpy array")

        if audio is None:
            audio = pydub_to_np(AudioSegment.from_wav(audio_path))[0]

        logits = self.predict([audio])
        predicted_id = np.argmax(logits, axis=1)[0]

        # TODO - these are not the logits, but the probs.. rename!

        predicted_probs = logits[:, predicted_id][0]
        return predicted_probs