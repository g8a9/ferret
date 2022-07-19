import numpy as np
from dataclasses import dataclass


@dataclass
class Explanation:
    """Generic class to represent an Explanation"""

    text: str
    tokens: str
    scores: np.array
    explainer: str


@dataclass
class ExplanationWithRationale(Explanation):
    """Specific explanation to contain the golden rationale"""

    rationale: np.array