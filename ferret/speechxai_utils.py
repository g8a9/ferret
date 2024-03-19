import os
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import pydub
import torch
from datasets import Dataset
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import librosa
import whisperx
from typing import Dict, List, Union, Tuple


def transcribe_audio(
    audio: np.ndarray,
    device: str = "cuda",
    batch_size: int = 2,
    compute_type: str = "float32",
    language: str = "en",
    model_name_whisper: str = "large-v2",
) -> Tuple[str, List[Dict[str, Union[str, float]]]]:
    """
    Transcribe audio using whisperx,
    and return the text (transcription) and the words with their start and end times.
    """

    ## Load whisperx model
    model_whisperx = whisperx.load_model(
        model_name_whisper,
        device,
        compute_type=compute_type,
        language=language,
    )

    ## Transcribe audio
    # TODO: we are assuming that the array does not come already normalized
    # audio_array = audio.normalized_array
    # The normalization occurs in the FerretAudio Class

    result = model_whisperx.transcribe(
        audio,
        batch_size=batch_size
    )
    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device
    )

    ## Align timestamps
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    if result is None or "segments" not in result or len(result["segments"]) == 0:
        return "", []

    if len(result["segments"]) == 1:
        text = result["segments"][0]["text"]
        words = result["segments"][0]["words"]
    else:
        text = " ".join(
            result["segments"][i]["text"] for i in range(len(result["segments"]))
        )
        words = [word for segment in result["segments"] for word in segment["words"]]

    # Remove words that are not properly transcribed
    words = [word for word in words if "start" in word]
    return text, words


def transcribe_audio_given_model(
    model_whisperx,
    audio_path: str,
    batch_size: int = 2,
    device: str = "cuda",
) -> Tuple[str, List[Dict[str, Union[str, float]]]]:
    """
    Transcribe audio using whisperx,
    and return the text (transcription) and the words with their start and end times.
    """

    ## Transcribe audio
    audio = whisperx.load_audio(audio_path)
    result = model_whisperx.transcribe(audio, batch_size=batch_size)
    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device
    )

    ## Align timestamps
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    if result is None or "segments" not in result or len(result["segments"]) == 0:
        return "", []

    if len(result["segments"]) == 1:
        text = result["segments"][0]["text"]
        words = result["segments"][0]["words"]
    else:
        text = " ".join(
            result["segments"][i]["text"] for i in range(len(result["segments"]))
        )
        words = [word for segment in result["segments"] for word in segment["words"]]

    # Remove words that are not properly transcribed
    words = [word for word in words if "start" in word]
    return text, words


class FerretAudio:
    """
    Internal class to handle audio data. We force signal to 1) mono, 2) a sampling rate of 16000, 3) np.float32 (i.e., 4 bytes to represent each sample).
    We infer the native sampling rate using librosa.
    """

    def __init__(
        self,
        audio_path_or_array: Union[str, np.ndarray],
        native_sr: int = None,
        model_helper=None,
    ):
        self.target_sr = 16000
        self.native_sr = native_sr
        self.audio_path_or_array = audio_path_or_array
        self.model_helper = model_helper
        self._transcription = None

        if isinstance(audio_path_or_array, str):
            self.native_sr = librosa.get_samplerate(audio_path_or_array)

            # Note: by default, librosa returns an array normalized in [-1,1].
            self.array, self.sample_rate = librosa.load(
                audio_path_or_array, sr=self.target_sr, dtype=np.float32
            )

        elif isinstance(audio_path_or_array, np.ndarray):
            if native_sr is None:
                raise ValueError(
                    "If audio is provided as a numpy array, native_sr must be provided"
                )
            self.array, self.sample_rate = librosa.resample(
                audio_path_or_array, self.native_sr, self.target_sr
            )

    @property
    def is_normalized(self) -> bool:
        """Check if the array is already normalized."""
        return np.max(np.abs(self.array)) <= 1.0
    
    @property
    def normalized_array(self) -> np.ndarray:
        if not self.is_normalized:
            return self.array / 32768.0
        else:
            return self.array

    @property
    def transcription(self):
        if self._transcription is None:
            if self.model_helper and hasattr(self.model_helper, 'device') and hasattr(self.model_helper, 'language'):
                _ , self._transcription = transcribe_audio(
                    audio=self.normalized_array,        # is normalization needed when transcribing? i am assumimg so
                    device=self.model_helper.device.type,
                    batch_size=2,
                    compute_type="float32",
                    language=self.model_helper.language,
                )
            else:
                raise AttributeError("model_helper is not correctly configured")
        return self._transcription
    
    @staticmethod
    def unnormalize_array(arr, dtype=np.int16):
        """
        Given a NumPy array normalized in `[-1, 1]`, returns an array rescaled
        in `[-max, max]`, where `max` is the maximum (in absolute value)
        (integer) number representable by the selected `dtype`. In practice,
        we convert a normalized array of dtype `float32` into a normalized
        one of dtype `int16`, as needed to create a PyDub `AudioSegment`
        object.
        """
        max_val = np.maximum(
            np.iinfo(dtype).max,
            np.abs(np.iinfo(dtype).min)
        )

        return (arr * max_val).astype(dtype)

    def to_pydub(self) -> pydub.AudioSegment:
        """
        Converts audio to `pydub.AudioSegment`.

        Notes:
            * In order to convert to PyDub `AudioSegment` type we need the
              array to be
                * of dtype int16,
                * NOT normalized.
              Therefore, if the array is normalized, we unnormalize it.
            * In any case, PyDub only works with unnormalized arrays of dtype
              int16, so that's what we need to pass as the input to
              `AudioSegment`.
            * Because we only manipulate mono audio, the array can either have
              shape `(n_samples, 1)` or `(n_samples,)` (flat array). Either is
              fine for PyDub (the extra dimension is taken care of
              automatically for mono audio).
        """
        if self.is_normalized:
            unnormalized_array = self.unnormalize_array(self.array)
        else:
            unnormalized_array = self.array

        return pydub.AudioSegment(
            unnormalized_array.tobytes(),
            frame_rate=self.target_sr,
            sample_width=unnormalized_array.dtype.itemsize,
            channels=1,
        )


def pydub_to_np(audio: pydub.AudioSegment) -> Tuple[np.ndarray, int]:
    """
    Converts pydub audio segment into np.float32 of shape [duration_in_seconds*sample_rate, channels],
    where each value is in range [-1.0, 1.0].
    Returns tuple (audio_np_array, sample_rate).
    """

    return (
        np.array(audio.get_array_of_samples(), dtype=np.float32).reshape(
            (-1, audio.channels)
        )
        / (1 << (8 * audio.sample_width - 1)),
        audio.frame_rate,
    )


def plot_word_importance_summary(
    df_labels,
    top_k=15,
    at_least=5,
    figsize=(7, 5),
    height=0.5,
    title="",
    ALL="-",
    label_id_to_text=None,
    legend_outside=True,
):
    """
    Plot the importance score for each word, separately for each class value

    Args:
        df_labels: dataframe with the word importance scores
        top_k: number of words to plot
        at_least: minimum number of times a word must appear to be considered
        figsize: size of the figure
        height: height of the bars
        title: title
        ALL: name of the column with the average importance score
        label_id_to_text: id to text label mapping (list)
        legend_outside: if True, the legend is outside the plot
    """

    import matplotlib.pyplot as plt

    import matplotlib.colors as cm

    sel = (
        df_labels.loc[df_labels["count"] > at_least]
        .sort_values(by=[ALL, "count"], ascending=False)
        .head(top_k)
    )

    n_colors = len([i for i in list(sel.T.index)])

    if n_colors < 20:
        cmap = plt.get_cmap("tab20")
        colors = {
            color_label: cm.to_hex(cmap(e))
            for e, color_label in enumerate(list(sel.T.index))
        }
    else:
        cmap = plt.get_cmap("gist_rainbow")

        colors = {
            color_label: cm.to_hex(cmap(1.0 * e / n_colors))
            for e, color_label in enumerate(list(sel.T.index))
        }

    fig, ax = plt.subplots(figsize=figsize)
    stacked = np.zeros(len(sel.T.columns))
    stacked_pos = np.zeros(len(sel.T.columns))
    stacked_neg = np.zeros(len(sel.T.columns))

    # This is just to have the words in the defined order: by highest average importance and count
    ax.barh(
        list(sel.T.columns)[::-1],
        np.zeros(len(sel.T.columns)),
        height=height,
        label=None,
    )

    for class_value_id in sel.T.index:
        if class_value_id not in ["count", ALL]:
            vals = sel.T.loc[class_value_id].values
            labels = sel.T.columns

            if n_colors > 20:
                label_name = (
                    f"{class_value_id}: {label_id_to_text[class_value_id]}"
                    if label_id_to_text is not None
                    else class_value_id
                )

            else:
                label_name = (
                    label_id_to_text[class_value_id]
                    if label_id_to_text is not None
                    else class_value_id
                )

            for i, v in enumerate(vals):
                if v > 0:
                    p = ax.barh(
                        labels[i],
                        v,
                        height=height,
                        left=stacked_pos[i],
                        label=label_name,
                        color=colors[class_value_id],
                    )
                    if n_colors > 20:
                        ax.bar_label(
                            p, labels=[class_value_id], label_type="center", fontsize=6
                        )
                elif v < 0:
                    p = ax.barh(
                        labels[i],
                        v,
                        height=height,
                        left=stacked_neg[i],
                        label=label_name,
                        color=colors[class_value_id],
                    )
                    if n_colors > 20:
                        ax.bar_label(
                            p, labels=[class_value_id], label_type="center", fontsize=6
                        )

            vals = np.nan_to_num(vals)
            stacked += vals
            stacked_neg += np.where(vals > 0, 0, vals)
            stacked_pos += np.where(vals < 0, 0, vals)

    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

    # To remove duplicates
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if legend_outside:
        ax.legend(
            by_label.values(),
            by_label.keys(),
            loc="upper left",
            bbox_to_anchor=(1, 1),
            fontsize=8,
        )
    else:
        ax.legend(by_label.values(), by_label.keys(), loc="upper left")
    if title != "":
        ax.set_title(title, fontsize=9)
    plt.show()
    return fig


def load_dataset_and_model(dataset_name, data_dir, model_dir=None, model_name=None):
    if dataset_name == "FSC":
        ## Load audio

        df = pd.read_csv(f"{data_dir}/data/test_data.csv")
        df["path"] = df["path"].apply(lambda x: os.path.join(data_dir, x))

        dataset = Dataset.from_pandas(df)

        if model_name is None:
            ## Load model
            model = Wav2Vec2ForSequenceClassification.from_pretrained(
                "superb/wav2vec2-base-superb-ic"
            )
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                "superb/wav2vec2-base-superb-ic"
            )
        else:
            ## Load model
            model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

    elif dataset_name == "IEMOCAP":
        # Note that the directory should contain the IEMOCAP folder
        # If the dir is already in the format /IEMOCAP, remove the last part (as for torchaudio loading)
        if os.path.basename(data_dir) == "IEMOCAP":
            data_dir = data_dir.replace(f"{os.path.sep}IEMOCAP", "")

        import torchaudio

        dataset_ta = torchaudio.datasets.IEMOCAP(data_dir, sessions=[1])
        data_dir = data_dir + f"{os.path.sep}IEMOCAP"

        df = pd.DataFrame(
            [dataset_ta.get_metadata(i) for i in range(len(dataset_ta))],
            columns=["path", "sample_rate", "filename", "label", "speaker"],
        )
        df = df[
            [
                "path",
                "label",
            ]
        ]
        df["path"] = df["path"].apply(lambda x: os.path.join(data_dir, x))
        dataset = Dataset.from_pandas(df)

        ## Load model
        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            "superb/wav2vec2-base-superb-er"
        )
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "superb/wav2vec2-base-superb-er"
        )

    elif dataset_name == "ITALIC":
        from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

        if model_dir is None:
            model_dir = f"{str(Path.home())}/ITALIC"

        ## Load model and feature extractor
        model_checkpoint = (
            model_dir
            + "/"
            + "wav2vec2-xls-r-300m-ic-finetuning-hard-speaker/checkpoint-3605"
        )

        feature_extractor = AutoFeatureExtractor.from_pretrained(
            "facebook/wav2vec2-xls-r-300m"
        )
        model = AutoModelForAudioClassification.from_pretrained(
            model_checkpoint, local_files_only=True
        )
        from datasets import load_dataset

        dataset_da = load_dataset("RiTA-nlp/ITALIC", "hard_speaker", use_auth_token=True)

        dataset = pd.DataFrame(
            {
                "path": [audio_i["path"] for audio_i in dataset_da["test"]["audio"]],
                "label": dataset_da["test"]["intent"],
            }
        )
        dataset = Dataset.from_pandas(dataset)
    else:
        raise ValueError("Dataset not supported")
    if torch.cuda.is_available():
        model = model.cuda()
    return dataset, model, feature_extractor
