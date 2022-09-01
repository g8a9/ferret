"""Explainers API"""

from abc import ABC, abstractmethod
from ..model_utils import ModelHelper


class BaseExplainer(ABC):
    @property
    @abstractmethod
    def NAME(self):
        pass

    def __init__(self, model, tokenizer):
        self.helper = ModelHelper(model, tokenizer)

    @property
    def device(self):
        return self.helper.model.device

    @property
    def tokenizer(self):
        return self.helper.tokenizer

    def _tokenize(self, text, **tok_kwargs):
        return self.helper._tokenize(text, **tok_kwargs)

    def get_tokens(self, text):
        return self.helper.get_tokens(text)

    def get_input_embeds(self, text):
        return self.helper.get_input_embeds(text)

    @abstractmethod
    def compute_feature_importance(self, text: str, target: int, **explainer_args):
        pass

    def __call__(self, text: str, target: int = 1, **explainer_args):
        return self.compute_feature_importance(text, target, **explainer_args)
