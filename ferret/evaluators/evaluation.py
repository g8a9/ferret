import numpy as np
from dataclasses import dataclass
from ferret.explainers.explanation import Explanation
from typing import List


@dataclass
class Evaluation:
    """Generic class to represent an Evaluation"""

    name: str
    score: float


@dataclass
class ExplanationEvaluation:
    """Generic class to represent an Evaluation"""

    explanation: Explanation
    evaluation_scores: List[Evaluation]
