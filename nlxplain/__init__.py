"""Top-level package for nlxplain."""

__author__ = """Giuseppe Attanasio"""
__email__ = "giuseppeattanasio6@gmail.com"
__version__ = "0.1.0"

from .explainer import Explainer
from .evaluation.explanation_evaluation import ExplanationEvalutator
from .evaluation.faithfulness_measures import (
    AOPC_Comprehensiveness_Evaluation,
    AOPC_Sufficiency_Evaluation,
    TauLOO_Evaluation,
)
from .evaluation.plausibility_measures import (
    AUPRC_PlausibilityEvaluation,
    Tokenf1_PlausibilityEvaluation,
    TokenIOU_PlausibilityEvaluation,
)
from .evaluation.classes_evaluation_measures import (
    AOPC_Comprehensiveness_Evaluation_by_class,
)
