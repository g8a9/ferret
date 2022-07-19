"""Client Interface Module"""

from typing import List
from . import SHAPExplainer, GradientExplainer, IntegratedGradientExplainer
from .explainers.explanation import Explanation
from copy import copy
import numpy as np


def normalize(explanations, ord=1):
    """Run Lp noramlization of explanation attribution scores"""

    # TODO can vectorize
    new_exps = list()
    for exp in explanations:
        new_exp = copy(exp)
        new_exp.scores /= np.linalg.norm(exp.scores, axis=-1, ord=1)  # L1 normalization
        new_exps.append(new_exp)

    return new_exps


class Explainer:
    def __init__(self, model, tokenizer, explainers: List = None):
        self.model = model
        self.tokenizer = tokenizer

        if not explainers:
            self._used_explainers = [
                SHAPExplainer,
                GradientExplainer,
                IntegratedGradientExplainer,
            ]
            self.explainers = [
                e(self.model, self.tokenizer) for e in self._used_explainers
            ]

    def compute_table(self, text):
        explanations = [exp(text) for exp in self.explainers]
        explanations = normalize(explanations)
        return explanations
