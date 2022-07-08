"""Explainers API"""

from abc import ABC, abstractmethod, abstractproperty


class BaseExplainer(ABC):
    @property
    @abstractmethod
    def NAME(self):
        pass

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, text: str = None):
        return self.compute_feature_importance(text)

    @abstractmethod
    def compute_feature_importance(self, text: str = None):
        pass