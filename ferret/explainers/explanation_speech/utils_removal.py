from pydub import AudioSegment
import whisperx
import os
import numpy as np
from typing import Dict, List, Union, Tuple
from ...speechxai_utils import FerretAudio


def remove_specified_words(audio, words, removal_type: str = "nothing"):
    """
    Remove a word from audio using pydub, by replacing it with:
    - nothing
    - silence
    - white noise
    - pink noise

    Args:
        audio (pydub.AudioSegment): audio
        word: word to remove with its start and end times
        removal_type (str, optional): type of removal. Defaults to "nothing".
    """

    from copy import deepcopy

    audio_removed = deepcopy(audio)

    a, b = 100, 40

    from IPython.display import display

    for word in words:
        start = int(word["start"] * 1000)
        end = int(word["end"] * 1000)

        before_word_audio = audio_removed[: start - a]
        after_word_audio = audio_removed[end + b :]

        word_duration = (end - start) + a + b

        if removal_type == "nothing":
            replace_word_audio = AudioSegment.empty()
        elif removal_type == "silence":
            replace_word_audio = AudioSegment.silent(duration=word_duration)
        elif removal_type == "white noise":
            sound_path = (os.path.join(os.path.dirname(__file__), "white_noise.mp3"),)
            replace_word_audio = AudioSegment.from_mp3(sound_path)[:word_duration]
        elif removal_type == "pink noise":
            sound_path = (os.path.join(os.path.dirname(__file__), "pink_noise.mp3"),)
            replace_word_audio = AudioSegment.from_mp3(sound_path)[:word_duration]

        audio_removed = before_word_audio + replace_word_audio + after_word_audio
    return audio_removed


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


def remove_word(audio, word, removal_type: str = "nothing"):
    """
    Remove a word from audio using pydub, by replacing it with:
    - nothing
    - silence
    - white noise
    - pink noise

    Args:
        audio (pydub.AudioSegment): audio
        word: word to remove with its start and end times
        removal_type (str, optional): type of removal. Defaults to "nothing".
    """

    a, b = 100, 40

    before_word_audio = audio[: word["start"] * 1000 - a]
    after_word_audio = audio[word["end"] * 1000 + b :]
    word_duration = (word["end"] * 1000 - word["start"] * 1000) + a + b

    # TODO GA: we don't really to use pydub here, we can use numpy directly

    if removal_type == "nothing":
        replace_word_audio = AudioSegment.empty()
    elif removal_type == "silence":
        replace_word_audio = AudioSegment.silent(duration=word_duration)

    elif removal_type == "white noise":
        sound_path = (os.path.join(os.path.dirname(__file__), "white_noise.mp3"),)
        replace_word_audio = AudioSegment.from_mp3(sound_path)[:word_duration]

        # display(audio_removed)
    elif removal_type == "pink noise":
        sound_path = (os.path.join(os.path.dirname(__file__), "pink_noise.mp3"),)
        replace_word_audio = AudioSegment.from_mp3(sound_path)[:word_duration]

    audio_removed = before_word_audio + replace_word_audio + after_word_audio
    return audio_removed


def remove_word_np(audio_array, sr, word, removal_type: str = "nothing"):
    """
    Remove a word from audio as an array, by replacing it with:
    - nothing
    - silence
    - white noise
    - pink noise

    Args:
        audio_array (np.ndarray): audio_array
        sr : sample rate of audio
        word: word to remove with its start and end times
        removal_type (str, optional): type of removal. Defaults to "nothing".
    """

    a, b = 100, 40

    start = int((word["start"] * 1000 - a) * sr / 1000)
    end = int((word["end"] * 1000 + b) * sr / 1000)
    before_word_audio = audio_array[:start]
    after_word_audio = audio_array[end:]
    word_duration = (end - start) + a + b

    if removal_type == "nothing":
        replace_word_audio = np.array([], dtype=audio_array.dtype)

    elif removal_type == "silence":
        replace_word_audio = np.zeros(word_duration, dtype=audio_array.dtype)

    elif removal_type == "pink noise":
        pass # to change the pink_noise.mp3 to a numpy array 

    elif removal_type == "white noise":
        pass # to change the white_noise.mp3 tp a numpy array

    audio_removed = np.concatenate(
        [before_word_audio, replace_word_audio, after_word_audio]
    )
    return audio_removed
