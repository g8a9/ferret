import pytest
import torch
import os
import numpy as np
import pandas as pd
from pydub import AudioSegment
from ferret import SpeechBenchmark
from ferret.explainers.explanation_speech.loo_speech_explainer import LOOSpeechExplainer
from ferret.explainers.explanation_speech.gradient_speech_explainer import (
    GradientSpeechExplainer,
)
from ferret.explainers.explanation_speech.lime_speech_explainer import (
    LIMESpeechExplainer,
)
from ferret.explainers.explanation_speech.paraling_speech_explainer import (
    ParalinguisticSpeechExplainer,
)
from scipy.io.wavfile import write
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor



# ================================================================
# = Fixtures creation audio sample to use throughout the testing =
# ================================================================
@pytest.fixture(scope="module")
def sample_audio_file():
    return os.path.join(os.path.dirname(__file__), 'data', 'sample_audio.wav')


@pytest.fixture(scope="module")
def benchmark():
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "superb/wav2vec2-base-superb-ic"
    )
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    "superb/wav2vec2-base-superb-ic"
    )
    return SpeechBenchmark(model, feature_extractor)

# ==========
# = Tests  =
# ==========

def test_initialization_benchmark(benchmark):
    assert benchmark.model is not None
    assert benchmark.feature_extractor is not None
    assert isinstance(benchmark, SpeechBenchmark)

def test_explainer_types(benchmark):
    for explainer_name, explainer in benchmark.explainers.items():
        assert explainer is not None
        assert explainer_name in ['LOO', 'Gradient', 'GradientXInput', 'LIME', 'perturb_paraling']
        assert isinstance(explainer, (LOOSpeechExplainer, GradientSpeechExplainer, LIMESpeechExplainer, ParalinguisticSpeechExplainer))


def test_audio_transcription(benchmark, sample_audio_file):     
    audio = AudioSegment.from_wav(sample_audio_file)
    sr = audio.frame_rate
    transcription = benchmark.transcribe(sample_audio_file, current_sr=sr)
    
    assert transcription[0] is not None
    assert transcription[0] == ' Turn up the bedroom heat.'

def test_prediction(benchmark, sample_audio_file):
    audio = AudioSegment.from_wav(sample_audio_file)
    audio_array = np.array(audio.get_array_of_samples()).astype(np.float32)
    audio_array /= np.max(np.abs(audio_array))
    predictions = benchmark.predict([audio_array])
    
    assert predictions is not None
    assert len(predictions) == 3  
    action_probs, object_probs, location_probs = benchmark.predict([audio_array])

    assert len(action_probs) == 1
    assert len(object_probs) == 1
    assert len(location_probs) == 1
    assert action_probs[0].shape == (6,)
    assert object_probs[0].shape == (14,)
    assert location_probs[0].shape == (4,)

@pytest.mark.parametrize("methodology", ["LOO", "Gradient", "LIME", "perturb_paraling"])
def test_explain_method(benchmark, sample_audio_file, methodology):
    explanations = benchmark.explain(
        audio_path_or_array=sample_audio_file,
        current_sr=16000,
        methodology=methodology,
    )
    
    assert explanations is not None
    
    if methodology != "perturb_paraling":
        assert hasattr(explanations, 'scores')
        assert hasattr(explanations, 'features')
        assert len(explanations.scores) > 0
        assert len(explanations.features) > 0
    else:
        assert isinstance(explanations, list)
        assert len(explanations) > 0
        for explanation in explanations:
            assert hasattr(explanation, 'scores')
            assert hasattr(explanation, 'features')


def test_explain_features(benchmark, sample_audio_file):
    explanations = benchmark.explain(
        audio_path_or_array=sample_audio_file,
        current_sr=16000,
        methodology='LOO',
    )
    
    expected_features = ['Turn', 'up', 'the', 'bedroom', 'heat.']
    assert explanations.features == expected_features

def test_invalid_audio_file(benchmark):
    with pytest.raises(Exception):
        benchmark.explain(
            audio_path_or_array='non_existent_file.wav',
            current_sr=16000,
            methodology='LOO',
        )

def test_silence_audio(benchmark):
    silent_audio = np.zeros(int(16000 * 1))  # 1 second of silent audio at 16kHz
    explanations = benchmark.explain(
        audio_path_or_array=silent_audio,
        current_sr=16000,
        methodology='LOO',
    )
    assert explanations is not None
    assert explanations.scores.shape == (3,0)
    assert len(explanations.features) == 0 

def test_explain_variations(benchmark, sample_audio_file):
    perturbation_types = ['time stretching', 'pitch shifting', 'noise']
    variations_table = benchmark.explain_variations(
        audio_path_or_array=sample_audio_file,
        current_sr=16000,
        perturbation_types=perturbation_types
    )
    assert isinstance(variations_table, dict)
    assert all(pt in variations_table for pt in perturbation_types)
    for pt, df in variations_table.items():
        assert isinstance(df, pd.DataFrame)
        assert not df.empty