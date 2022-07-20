"""Client Interface Module"""

from typing import List, Union

from ferret.datasets import BaseDataset

from . import (
    SHAPExplainer,
    GradientExplainer,
    IntegratedGradientExplainer,
    LIMEExplainer,
)

from .evaluators.faithfulness_measures import (
    AOPC_Comprehensiveness_Evaluation,
    AOPC_Sufficiency_Evaluation,
    TauLOO_Evaluation,
)

from .evaluators.evaluation import Evaluation

from .evaluators.plausibility_measures import (
    AUPRC_PlausibilityEvaluation,
    Tokenf1_PlausibilityEvaluation,
    TokenIOU_PlausibilityEvaluation,
)

from .evaluators.evaluation import ExplanationEvaluation
from .explainers.explanation import Explanation, ExplanationWithRationale

from .modelw import Model
import copy

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import seaborn as sns


SCORES_PALETTE = sns.diverging_palette(240, 10, as_cmap=True)
EVALUATION_PALETTE_HIGHER_BETTER = sns.light_palette("purple", as_cmap=True)
EVALUATION_PALETTE_LOWER_BETTER = sns.light_palette("blue", as_cmap=True, reverse=True)


def normalize(explanations, ord=1):
    """Run Lp noramlization of explanation attribution scores"""

    # TODO vectorize to improve perf
    new_exps = list()
    for exp in explanations:
        new_exp = copy.copy(exp)
        new_exp.scores /= np.linalg.norm(exp.scores, axis=-1, ord=1)  # L1 normalization
        new_exps.append(new_exp)

    return new_exps


class Benchmark:
    """Generic interface to compute multiple explanations."""

    def __init__(
        self, model, tokenizer, explainers: List = None, evaluators: List = None
    ):
        self.model = model
        self.tokenizer = tokenizer

        if not explainers:
            self._used_explainers = [
                SHAPExplainer,
                GradientExplainer,
                IntegratedGradientExplainer,
                LIMEExplainer,
            ]
            self.explainers = [
                e(self.model, self.tokenizer) for e in self._used_explainers
            ]
        if not evaluators:
            self._used_evaluators = [
                AOPC_Comprehensiveness_Evaluation,
                AOPC_Sufficiency_Evaluation,
                TauLOO_Evaluation,
                AUPRC_PlausibilityEvaluation,
                Tokenf1_PlausibilityEvaluation,
                TokenIOU_PlausibilityEvaluation,
            ]
            model_wrapper = Model(self.model)
            self.evaluators = [
                ev(model_wrapper, self.tokenizer) for ev in self._used_evaluators
            ]

    def explain(self, text, progress_bar: bool = True) -> List[Explanation]:
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

    def evaluate_explanation(
        self,
        explanation: Union[Explanation, ExplanationWithRationale],
        target,
        human_rationale=None,
        **evaluation_args
    ) -> ExplanationEvaluation:

        evaluations = list()

        add_first_last = evaluation_args.get("add_first_last", True)
        explanation = (
            self._add_rationale(explanation, human_rationale, add_first_last)
            if human_rationale is not None
            else explanation
        )

        for evaluator in self.evaluators:
            evaluation = evaluator.compute_evaluation(
                explanation, target, **evaluation_args
            )
            if (
                evaluation is not None
            ):  # return None for plausibility measure if rationale is not available
                evaluations.append(evaluation)
        explanation_eval = ExplanationEvaluation(explanation, evaluations)
        return explanation_eval

    def _add_rationale(
        self,
        explanation: Explanation,
        rationale: List,
        add_first_last=True,
        assign_equal_importance=True,
    ) -> ExplanationWithRationale:
        if rationale == []:
            if assign_equal_importance:
                rationale = [1] * len(explanation.tokens)
        else:
            if add_first_last:
                # We add the importance of the first and last token (0 as default)
                rationale = [0] + rationale + [0]
        if len(explanation.tokens) != len(rationale):
            raise ValueError()
        return ExplanationWithRationale(
            explanation.text,
            explanation.tokens,
            explanation.scores,
            explanation.explainer,
            rationale,
        )

    def evaluate_explanations(
        self,
        explanations: List[Explanation],
        target,
        human_rationale=None,
        **evaluation_args
    ) -> List[ExplanationEvaluation]:
        explanation_evaluations = list()
        for explanation in explanations:
            explanation_evaluations.append(
                self.evaluate_explanation(
                    explanation, target, human_rationale, **evaluation_args
                )
            )
        return explanation_evaluations

    def generate_dataset_explanations(self, data: BaseDataset, target=None):
        """
        Data
        Target: if target is none, the explanation is with respect the predicted class
        """
        if isinstance(data, BaseDataset) == False:
            raise ValueError("Type of data should be BaseDataset")
        dataset_explanations = list()
        for i in range(0, 2):  # len(data)):
            instance = data[i]
            text = instance["text"]
            explanations = self.explain(text)
            if "rationale" in instance:
                explanations = [
                    self._add_rationale(explanation, instance["rationale"])
                    for explanation in explanations
                ]
            dataset_explanations.append(explanations)
        return dataset_explanations

    def evaluate_dataset_explanations(
        self,
        dataset_explanations: List[List[Union[Explanation, ExplanationWithRationale]]],
        target=1,
        **evaluation_args
    ):
        # We want to accumulate the results.
        evaluation_args["accumulate_result"] = True

        # Init dictionary with accumation scores
        dataset_evaluation_scores = {}

        for explainer in self.explainers:
            dataset_evaluation_scores[explainer.NAME] = {}
            for evaluator in self.evaluators:
                dataset_evaluation_scores[explainer.NAME][
                    evaluator.SHORT_NAME
                ] = copy.deepcopy(evaluator.INIT_VALUE)

        n_explanations = len(dataset_explanations)
        for explanations in dataset_explanations:
            for explanation in explanations:
                evaluation = self.evaluate_explanation(
                    explanation, target, **evaluation_args
                )
                # Accumulate scores
                for evaluation_score in evaluation.evaluation_scores:
                    dataset_evaluation_scores[explanation.explainer][
                        evaluation_score.name
                    ] += evaluation_score.score

        # From accumulated results to average vaalue
        for evaluator in self.evaluators:
            for explainer_name, evaluations in dataset_evaluation_scores.items():
                dataset_evaluation_scores[explainer_name][
                    evaluator.SHORT_NAME
                ] = evaluator.aggregate_score(
                    evaluations[evaluator.SHORT_NAME], n_explanations
                )
        return dataset_evaluation_scores

    def get_dataframe(self, explanations) -> pd.DataFrame:
        scores = {e.explainer: e.scores for e in explanations}
        scores["Token"] = explanations[0].tokens
        table = pd.DataFrame(scores).set_index("Token").T
        return table

    def show_table(self, explanations, apply_style: bool = True) -> pd.DataFrame:
        """Format explanations scores into a colored table."""
        table = self.get_dataframe(explanations)

        # Rename duplicate columns (tokens) by adding a suffix
        if sum(table.columns.duplicated().astype(int)) > 0:
            table.columns = pd.io.parsers.base_parser.ParserBase(
                {"names": table.columns, "usecols": None}
            )._maybe_dedup_names(table.columns)

        return (
            table.style.background_gradient(
                axis=1, cmap=SCORES_PALETTE, vmin=-1, vmax=1
            )
            if apply_style
            else table
        )

    def show_evaluation_table(
        self,
        explanation_evaluations: List[ExplanationEvaluation],
        apply_style: bool = True,
    ) -> pd.DataFrame:
        """Format explanations and evaluations scores into a colored table."""

        explanations = [
            explanation_evaluation.explanation
            for explanation_evaluation in explanation_evaluations
        ]
        table = self.get_dataframe(explanations)

        # Rename duplicate columns (tokens) by adding a suffix
        if sum(table.columns.duplicated().astype(int)) > 0:
            table.columns = pd.io.parsers.base_parser.ParserBase(
                {"names": table.columns, "usecols": None}
            )._maybe_dedup_names(table.columns)

        explainer_scores = {}
        for explanation_evaluation in explanation_evaluations:
            explainer_scores[explanation_evaluation.explanation.explainer] = {
                evaluation.name: evaluation.score
                for evaluation in explanation_evaluation.evaluation_scores
            }

        table = pd.concat([table, pd.DataFrame(explainer_scores).T], axis=1)

        if apply_style:

            table_style = self.style_evaluation(table)

            return table_style

        else:
            table

    def show_dataset_evaluation_table(
        self,
        dataset_evaluation_average_scores,
        apply_style: bool = True,
    ) -> pd.DataFrame:
        """Format explanations and evaluations scores into a colored table."""

        table = pd.DataFrame(dataset_evaluation_average_scores).T

        if apply_style:

            table_style = self.style_evaluation(table)

            return table_style

        else:
            return table

    def style_evaluation(self, table):
        table_style = table.style.background_gradient(
            axis=1, cmap=SCORES_PALETTE, vmin=-1, vmax=1
        )
        show_higher_cols, show_lower_cols = list(), list()
        # Highlight with two different palettes
        for evaluation_measure in self.evaluators:
            if evaluation_measure.SHORT_NAME in table.columns:
                if evaluation_measure.BEST_SORTING_ASCENDING == False:
                    # Higher is better
                    show_higher_cols.append(evaluation_measure.SHORT_NAME)
                else:
                    # Lower is better
                    show_lower_cols.append(evaluation_measure.SHORT_NAME)

        if show_higher_cols:
            table_style.background_gradient(
                axis=1,
                cmap=EVALUATION_PALETTE_HIGHER_BETTER,
                vmin=-1,
                vmax=1,
                subset=show_higher_cols,
            )

        if show_lower_cols:
            table_style.background_gradient(
                axis=1,
                cmap=EVALUATION_PALETTE_LOWER_BETTER,
                vmin=-1,
                vmax=1,
                subset=show_lower_cols,
            )
        return table_style
