"""Evaluators API"""

from abc import ABC, abstractmethod
from typing import Any, List, Union

from ferret.explainers.explanation import Explanation, ExplanationWithRationale

from ..model_utils import create_helper


class BaseEvaluator(ABC):

    INIT_VALUE = 0

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

    @property
    def tokenizer(self):
        return self.helper.tokenizer

    def __init__(self, model, tokenizer, task_name):
        if model is None or tokenizer is None:
            raise ValueError("Please specify a model and a tokenizer.")

        self.helper = create_helper(model, tokenizer, task_name)

    def __call__(self, explanation: Explanation):
        return self.compute_evaluation(explanation)

    @abstractmethod
    def compute_evaluation(
        self, explanation: Union[Explanation, ExplanationWithRationale]
    ):
        pass

    def aggregate_score(self, score, total, **aggregation_args):
        return score / total
