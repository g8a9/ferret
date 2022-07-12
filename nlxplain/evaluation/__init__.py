"""Evaluators API"""

from abc import ABC, abstractmethod
from typing import List, Any


class BaseEvaluator(ABC):
    @property
    @abstractmethod
    def NAME(self):
        pass

    @property
    @abstractmethod
    def SHORT_NAME(self):
        pass

    @property
    @abstractmethod
    def BEST_SORTING_ASCENDING(self):
        # True: the lower the better
        # False: the higher the better
        pass

    @property
    @abstractmethod
    def TYPE_METRIC(self):
        # plausibility
        # faithfulness
        pass

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, text: str = None, score_explanation: List[float] = []):
        return self.evaluate_explanation(text, score_explanation)

    @abstractmethod
    def evaluate_explanation(
        self, text: str = None, score_explanation: List[float] = []
    ):
        pass

    @abstractmethod
    def evaluate_explanations(
        self,
        texts: List[str],
        score_explanations: List[List[float]],
        targets: List[Any],
    ):

        """
        Compute the aggregate evaluation score for a list of score explanations.
        texts: list of texts
        score_explanations: list of (corresponding) explanations
        targets: list of targets

        Return the aggregate evaluation score.
        """
        pass
