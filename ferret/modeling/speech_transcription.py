import whisperx
from pydub import AudioSegment
from ..speechxai_utils import pydub_to_np


# Default values (see default values for the CLI arguments in: 
# https://github.com/m-bain/whisperX/blob/main/whisperx/transcribe.py).
# Note: could be moved to a default config file.
BATCH_SIZE = 8
CHUNK_SIZE = 30
PRINT_PROGRESS = False


class SpeechTrancriber:
    def __init__(
            self,
            asr_model_name,
            device,
            language_code,
            compute_type='float16',
        ):
        """
        Class initializer. Loads the specified model for speech transcription
        and a model for text alignment.

        Args:
            asr_model_name (str): name of the ASR model to load.
            device (str): name of the device on which to run the computations.
            language_code (str): code for the selected language (e.g. 'en').
            compute_type (str, default: 'float16'): string indicating the
                datatype with which to work with the ASR model.
        """
        self.asr_model_name = asr_model_name
        self.device = device
        self.compute_type = compute_type
        self.language_code = language_code

        # Load ASR model.
        self.asr_model = whisperx.load_model(
            asr_model_name, device=device, compute_type=compute_type
        )

        # Load ASR alignment model.
        (
            self.asr_alignment_model,
            self.asr_alignment_model_metadata
        ) = whisperx.load_align_model(
            language_code=language_code, device=device
        )

    def transcribe_audio(self, audio_file_path, return_audio_obj=False):
        """
        Transcribes the audio file (containing speech) at the specified path.

        Args:
            audio_file_path (str): path (relative or absolute) to the audio
                file.
            return_audio_obj (bool, default: False): flag indicating whether
                to return the Python object (of type
                `pydub.audio_segment.AudioSegment`) corresponding to the
                audio.

        Returns:
            transcription (dict): output dictionary for the transcription
                operation.
            audio (optional, pydub.audio_segment.AudioSegment): Python object
                correspnding to the audio. Only returned if
                `return_audio_obj=True`.
        """
        audio_extension = audio_file_path.split('.')[-1].lower()

        if audio_extension == 'wav':
            audio = AudioSegment.from_wav(audio_file_path)
        else:
            raise NotImplementedError(
                f'Audio extension {audio_extension} not supported'
            )
        
        transcription = self.asr_model.transcribe(
            pydub_to_np(audio)[0].flatten(),
            batch_size=BATCH_SIZE,
            chunk_size=CHUNK_SIZE,
            print_progress=PRINT_PROGRESS
        )

        if not return_audio_obj:
            return transcription
        else:
            return transcription, audio
    
    def align_transcription(
            self,
            audio,
            segments,
            return_char_alignments=False
        ):
        """
        Aligns the transcription, i.e. returns a dictionary with start and
        end time for every word.

        Args:
            audio (pydub.audio_segment.AudioSegment): Python object
                correspnding to the audio.
            segments (list): list of dicts corresponding to the segments found
                in the audio transcription.
            return_char_alignments (bool, default=False): flag to indicate
                whether to return the per-character alignments.

        Returns
            aligned_transcription (dict): dictionary containing the results
                of the alignment process (segments and word segments).
        """
        aligned_transcription = whisperx.align(
            segments,
            self.asr_alignment_model,
            self.asr_alignment_model_metadata,
            pydub_to_np(audio)[0].flatten(),
            self.device,
            return_char_alignments=return_char_alignments
        )

        return aligned_transcription
    
    def align_transcription_from_file(
            self,
            audio_file_path,
            return_char_alignments=False
        ):
        """
        Same as `align_transcription`, but starting from the path of the audio
        file (includes the transcription phase).

        Args:
            audio_file_path (str): path (relative or absolute) to the audio
                file.
            return_char_alignments (bool, default=False): flag to indicate
                whether to return the per-character alignments.
        
        Returns:
            aligned_transcription (dict): dictionary containing the results
                    of the alignment process (segments and word segments).
        """
        transcription, audio = self.transcribe_audio(
            audio_file_path,
            return_audio_obj=True
        )

        aligned_transcriptions = self.align_transcription(
            audio,
            transcription['segments'],
            return_char_alignments=return_char_alignments
        )

        return aligned_transcriptions