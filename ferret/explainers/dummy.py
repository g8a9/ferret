"""Dummy Explainer module"""
import numpy as np

from . import BaseExplainer
from .explanation import Explanation
from .utils import parse_explainer_args


class DummyExplainer(BaseExplainer):
    """Dummy Explainer that assigns random scores to tokens."""

    NAME = "dummy"

    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer)

    def compute_feature_importance(self, text, target=1, **explainer_args):
        tokens = self._tokenize(text)
        output = Explanation(
            text, self.get_tokens(text), np.random.randn(len(tokens)), self.NAME, target
        )
        return output
