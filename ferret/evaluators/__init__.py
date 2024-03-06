"""Evaluators API"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, List, Union

from ..explainers.explanation import Explanation, ExplanationWithRationale
from ..modeling import create_helper
from ..explainers.explanation_speech.explanation_speech import (
    ExplanationSpeech, EvaluationSpeech)


class EvaluationMetricFamily(Enum):
    """Enum to represent the family of an EvaluationMetric"""

    FAITHFULNESS = "faithfulness"
    PLAUSIBILITY = "plausibility"


class BaseEvaluator(ABC):
    @property
    @abstractmethod
    def NAME(self):
        pass

    @property
    @abstractmethod
    def MIN_VALUE(self):
        pass

    @property
    @abstractmethod
    def MAX_VALUE(self):
        pass

    @property
    @abstractmethod
    def SHORT_NAME(self):
        pass

    @property
    @abstractmethod
    def LOWER_IS_BETTER(self):
        pass

    @property
    @abstractmethod
    def METRIC_FAMILY(self) -> EvaluationMetricFamily:
        pass

    def __repr__(self) -> str:
        return str(
            dict(
                NAME=self.NAME,
                SHORT_NAME=self.SHORT_NAME,
                MIN_VALUE=self.MIN_VALUE,
                MAX_VALUE=self.MAX_VALUE,
                LOWER_IS_BETTER=self.LOWER_IS_BETTER,
                METRIC_FAMILY=self.METRIC_FAMILY,
            )
        )

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

    # def aggregate_score(self, score, total, **aggregation_args):
    #     return score / total


class SpeechBaseEvaluator(ABC):
    """
    Abstract base class for evaluator objects (metrics) for speech
    explainability.

    Notes:
        * Should we include `MIN_VALUE` and `MAX_VALUE` properties, as for
        the text-based evaluators?
    """
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
    def LOWER_IS_BETTER(self):
        pass

    @property
    @abstractmethod
    def METRIC_FAMILY(self) -> EvaluationMetricFamily:
        pass

    def __repr__(self) -> str:
        return str(
            dict(
                NAME=self.NAME,
                SHORT_NAME=self.SHORT_NAME,
                LOWER_IS_BETTER=self.LOWER_IS_BETTER,
                METRIC_FAMILY=self.METRIC_FAMILY,
            )
        )

    def __init__(self, model_helper, **kwargs):
        self.model_helper = model_helper

    @abstractmethod
    def compute_evaluation(
        self,
        explanation: ExplanationSpeech,
        target: List = None,
        words_trascript: List = None,
        **evaluation_args,
    ) -> EvaluationSpeech:
        pass