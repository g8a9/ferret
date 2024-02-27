from dataclasses import dataclass
import numpy as np
from typing import Optional


@dataclass
class ExplanationSpeech:
    features: list
    scores: np.array
    explainer: str
    target: list
    audio_path: Optional[str] = None


@dataclass
class EvaluationSpeech:
    """
    Generic class to represent a speech Evaluation.

    Note: this has a subset of the `Explanation` dataclass' attributes, so it
          should be possible to write smaller common parent class for both
          very similar to this (the `Explanation` class - for text - is more
          specific).
    """

    name: str
    score: list
    target: list