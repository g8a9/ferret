"""Top-level package for ferret."""

__author__ = """Giuseppe Attanasio"""
__email__ = "giuseppeattanasio6@gmail.com"
__version__ = "0.5.0"

import logging

# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

from .benchmark import Benchmark

# Dataset Interface
from .datasets import BaseDataset
from .datasets.datamanagers import HateXplainDataset, MovieReviews, SSTDataset
from .datasets.datamanagers_thermostat import ThermostatDataset

# Benchmarking methods
from .evaluators import BaseEvaluator
from .evaluators.faithfulness_measures import (
    AOPC_Comprehensiveness_Evaluation,
    AOPC_Sufficiency_Evaluation,
    TauLOO_Evaluation,
)
from .evaluators.plausibility_measures import (
    AUPRC_PlausibilityEvaluation,
    Tokenf1_PlausibilityEvaluation,
    TokenIOU_PlausibilityEvaluation,
)

# Explainers
from .explainers import BaseExplainer
from .explainers.dummy import DummyExplainer
from .explainers.gradient import GradientExplainer, IntegratedGradientExplainer
from .explainers.lime import LIMEExplainer
from .explainers.shap import SHAPExplainer
from .modeling.text_helpers import TokenClassificationHelper


# Check for manual installation of `WhisperX`.
try:
    import whisperx
except ImportError as e:
    logging.error(
        'Library whisperx not found. Please install it manually from GitHub: '
        '`pip install git+https://github.com/m-bain/whisperx.git`'
    )

    raise e


# Conditional imports for speech-related tasks
try:
    # Explainers
    from .explainers.explanation_speech.paraling_speech_explainer import (
        ParalinguisticSpeechExplainer,
    )
    from .explainers.explanation_speech.loo_speech_explainer import LOOSpeechExplainer
    from .explainers.explanation_speech.explanation_speech import ExplanationSpeech

    # Model Helpers
    from .modeling.speech_model_helpers.model_helper_er import ModelHelperER
    from .modeling.speech_model_helpers.model_helper_fsc import ModelHelperFSC
    from .modeling.speech_model_helpers.model_helper_italic import ModelHelperITALIC
    from .benchmark_speech import SpeechBenchmark

    # Benchmarking methods
    from .evaluators.faithfulness_measures_speech import (
        AOPC_Comprehensiveness_Evaluation_Speech,
        AOPC_Sufficiency_Evaluation_Speech,
    )
except ImportError as e:
    logger.error(
        'Speech-related modules could not be imported. It is very likely that'
        ' ferret was installed in the standard, text-only mode. Run '
        '`pip install ferret-xai[speech]` or `pip install ferret-xai[all]` to'
        ' include them'
    )

    raise e
