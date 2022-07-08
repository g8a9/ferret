"""Dummy Explainer module"""
from . import BaseExplainer
from .utils import parse_explainer_args
import numpy as np


class DummyExplainer(BaseExplainer):
    NAME = "dummy"

    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer)

    def compute_feature_importance(self, text, target=1, **explainer_args):
        tokens = self.tokenizer(text)
        return np.random.randn(len(tokens))
