"""Top-level package for ferret."""

__author__ = """Giuseppe Attanasio"""
__email__ = "giuseppeattanasio6@gmail.com"
__version__ = "0.4.1"


from .benchmark import Benchmark
from .datasets.datamanagers import HateXplainDataset
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
from .explainers.dummy import DummyExplainer
from .explainers.gradient import GradientExplainer, IntegratedGradientExplainer
from .explainers.lime import LIMEExplainer
from .explainers.shap import SHAPExplainer

# from .evaluators.classes_evaluation_measures import (
#    AOPC_Comprehensiveness_Evaluation_by_class,
# )
