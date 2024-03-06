from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

from ferret.explainers.explanation import Explanation

from . import BaseEvaluator


@dataclass
class EvaluationMetricOutput:
    """Output to store any metric result."""

    metric: BaseEvaluator
    value: float


@dataclass
class ExplanationEvaluation:
    """Generic class to represent an Evaluation"""

    explanation: Explanation
    evaluation_outputs: List[EvaluationMetricOutput]
