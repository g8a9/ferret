"""Top-level package for ferret."""

__author__ = """Giuseppe Attanasio"""
__email__ = "giuseppeattanasio6@gmail.com"
__version__ = "0.4.1"


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

# Speech-XAI section.
from .benchmark_speech import Benchmark

# Benchmarking methods
from .evaluators.faithfulness_measures_speech import (
    AOPC_Comprehensiveness_Evaluation_Speech,
    AOPC_Sufficiency_Evaluation_Speech,
)

# Explainers
from .explainers.explanation_speech.paraling_speech_explainer import ParalinguisticSpeechExplainer
from .explainers.explanation_speech.loo_speech_explainer import LOOSpeechExplainer
from .explainers.explanation_speech.explanation_speech import ExplanationSpeech

# Model Helpers
from .modeling.speech_model_helpers.model_helper_er import ModelHelperER
from .modeling.speech_model_helpers.model_helper_fsc import ModelHelperFSC
from .modeling.speech_model_helpers.model_helper_italic import ModelHelperITALIC