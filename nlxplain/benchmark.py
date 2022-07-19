"""Client Interface Module"""

from typing import List
from . import SHAPExplainer, GradientExplainer, IntegratedGradientExplainer
from .explainers.explanation import Explanation
from copy import copy
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


SCORES_PALETTE = sns.diverging_palette(240, 10, as_cmap=True)


def normalize(explanations, ord=1):
    """Run Lp noramlization of explanation attribution scores"""

    # TODO can vectorize
    new_exps = list()
    for exp in explanations:
        new_exp = copy(exp)
        new_exp.scores /= np.linalg.norm(exp.scores, axis=-1, ord=1)  # L1 normalization
        new_exps.append(new_exp)

    return new_exps


class Benchmark:
    """Generic interface to compute multiple explanations."""

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

    def explain(self, text, progress_bar: bool = True):
        """Compute explanations."""

        if progress_bar:
            pbar = tqdm(total=len(self.explainers), desc="Explainer")

        explanations = list()
        for exp in self.explainers:
            explanations.append(exp(text))
            if progress_bar:
                pbar.update(1)

        if progress_bar:
            pbar.close()

        explanations = normalize(explanations)
        return explanations

    def get_dataframe(self, explanations):
        scores = {e.explainer: e.scores for e in explanations}
        scores["Token"] = explanations[0].tokens
        table = pd.DataFrame(scores).set_index("Token").T
        return table

    def show_table(self, explanations, apply_style: bool = True):
        """Format explanations scores into a colored table."""
        table = self.get_dataframe(explanations)

        return (
            table.style.background_gradient(
                axis=1, cmap=SCORES_PALETTE, vmin=-1, vmax=1
            )
            if apply_style
            else table
        )
