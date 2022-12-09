"""Client Interface Module"""

import copy
import dataclasses
import json
from typing import Dict, List, Union

import datasets
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from joblib import Parallel, delayed
from torch.nn.functional import softmax
from tqdm.auto import tqdm

from ferret.datasets import BaseDataset

from . import (
    GradientExplainer,
    IntegratedGradientExplainer,
    LIMEExplainer,
    SHAPExplainer,
)
from .datasets.datamanagers import HateXplainDataset, MovieReviews, SSTDataset
from .evaluators.class_measures import AOPC_Comprehensiveness_Evaluation_by_class
from .evaluators.evaluation import Evaluation, ExplanationEvaluation
from .evaluators.faithfulness_measures import (
    AOPC_Comprehensiveness_Evaluation,
    AOPC_Sufficiency_Evaluation,
    TauLOO_Evaluation,
)
from .evaluators.plausibility_measures import (
    AUPRC_PlausibilityEvaluation,
    Tokenf1_PlausibilityEvaluation,
    TokenIOU_PlausibilityEvaluation,
)
from .explainers.explanation import Explanation, ExplanationWithRationale
from .model_utils import create_helper

SCORES_PALETTE = sns.diverging_palette(240, 10, as_cmap=True)
EVALUATION_PALETTE = sns.light_palette("purple", as_cmap=True)
EVALUATION_PALETTE_REVERSED = sns.light_palette("purple", as_cmap=True, reverse=True)

NONE_RATIONALE = []


def lp_normalize(explanations, ord=1):
    """Run Lp-noramlization of explanation attribution scores"""

    new_exps = list()
    for exp in explanations:
        new_exp = copy.copy(exp)
        new_exp.scores /= np.linalg.norm(
            exp.scores, axis=-1, ord=ord
        )  # L1 normalization
        new_exps.append(new_exp)

    return new_exps


class Benchmark:
    """Generic interface to compute multiple explanations."""

    def __init__(
        self,
        model,
        tokenizer,
        task_name: str = "text-classification",
        explainers: List = None,
        evaluators: List = None,
        class_based_evaluators: List = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.task_name = task_name
        self.helper = create_helper(self.model, self.tokenizer, self.task_name)

        self.explainers = explainers
        self.evaluators = evaluators
        self.class_based_evaluators = class_based_evaluators

        if not explainers:
            self.explainers = [
                SHAPExplainer(self.model, self.tokenizer),
                LIMEExplainer(self.model, self.tokenizer),
                GradientExplainer(self.model, self.tokenizer, multiply_by_inputs=False),
                GradientExplainer(self.model, self.tokenizer, multiply_by_inputs=True),
                IntegratedGradientExplainer(
                    self.model, self.tokenizer, multiply_by_inputs=False
                ),
                IntegratedGradientExplainer(
                    self.model, self.tokenizer, multiply_by_inputs=True
                ),
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
            self.evaluators = [
                ev(self.model, self.tokenizer, self.task_name)
                for ev in self._used_evaluators
            ]
        if not class_based_evaluators:
            self._used_class_evaluators = [AOPC_Comprehensiveness_Evaluation_by_class]
            self.class_based_evaluators = [
                class_ev(self.model, self.tokenizer, self.task_name)
                for class_ev in self._used_class_evaluators
            ]

    def explain(
        self,
        text,
        target=1,
        show_progress: bool = True,
        normalize_scores: bool = True,
        order: int = 1,
    ) -> List[Explanation]:
        """Compute explanations using all the explainers stored in the class.

        Args:
            text (str): text string to explain.
            target (int): class label to produce the explanations for
            show_progress (bool): enable progress bar
            normalize_scores (bool): do lp-normalization to make scores comparable

        Returns:
            List[Explanation]: list of all explanations produced
        """

        # sanity check and transformation to integer targets (if required)
        # here we are assuming the same target format (e.g., positional integer will work
        # for every explanation method. We might need to chage this in the future, when
        # we will add new explanation methods.
        requested_target = target
        target = self.helper.check_format_target(target)

        # we might optimize running the loop in parallel
        explanations = list()
        for explainer in tqdm(
            self.explainers,
            total=len(self.explainers),
            desc="Explainer",
            leave=False,
            disable=not show_progress,
        ):
            exp = explainer(text, target)
            exp.requested_target = requested_target
            explanations.append(exp)

        if normalize_scores:
            explanations = lp_normalize(explanations, order)

        return explanations

    def evaluate_explanation(
        self,
        explanation: Union[Explanation, ExplanationWithRationale],
        target,
        human_rationale=None,
        class_explanation: List[Union[Explanation, ExplanationWithRationale]] = None,
        progress_bar=True,
        **evaluation_args,
    ) -> ExplanationEvaluation:

        """
        explanation: Explanation to evaluate.
        target: target class for which we evaluate the explanation
        human rationale: List in one-hot-encoding indicating if the token is in the rationale (1) or not (i)
        class_explanation: list of explanations. The explanation in position 'i' is computed using as target class the class 'i'.

        len = #target classes. If available, class-based scores are computed
        """
        evaluations = list()

        if progress_bar:
            total_evaluators = (
                len(self.evaluators) + len(self.class_based_evaluators)
                if class_explanation is not None
                else len(self.evaluators)
            )
            pbar = tqdm(total=total_evaluators, desc="Evaluator", leave=False)

        add_first_last = evaluation_args.get("add_first_last", True)
        explanation = (
            self._add_rationale(explanation, human_rationale, add_first_last)
            if human_rationale is not None
            else explanation
        )

        target = self.helper.check_format_target(target)

        for evaluator in self.evaluators:
            evaluation = evaluator.compute_evaluation(
                explanation, target, **evaluation_args
            )
            if (
                evaluation is not None
            ):  # return None for plausibility measure if rationale is not available
                evaluations.append(evaluation)
            if progress_bar:
                pbar.update(1)

        if class_explanation is not None:
            for class_based_evaluator in self.class_based_evaluators:
                class_based_evaluation = class_based_evaluator.compute_evaluation(
                    class_explanation, **evaluation_args
                )
                evaluations.append(class_based_evaluation)
                if progress_bar:
                    pbar.update(1)

        if progress_bar:
            pbar.close()
        explanation_eval = ExplanationEvaluation(explanation, evaluations)

        return explanation_eval

    def evaluate_explanations(
        self,
        explanations: List[Union[Explanation, ExplanationWithRationale]],
        target,
        human_rationale=None,
        class_explanations=None,
        progress_bar=True,
        **evaluation_args,
    ) -> List[ExplanationEvaluation]:
        explanation_evaluations = list()

        class_explanations_by_explainer = self._get_class_explanations_by_explainer(
            class_explanations
        )
        if progress_bar:
            pbar = tqdm(total=len(explanations), desc="Explanation eval", leave=False)

        for i, explanation in enumerate(explanations):
            class_explanation = None
            if class_explanations_by_explainer is not None:
                class_explanation = class_explanations_by_explainer[i]

            explanation_evaluations.append(
                self.evaluate_explanation(
                    explanation,
                    target,
                    human_rationale,
                    class_explanation,
                    progress_bar=False,
                    **evaluation_args,
                )
            )
            if progress_bar:
                pbar.update(1)
        if progress_bar:
            pbar.close()
        return explanation_evaluations

    def _add_rationale(
        self,
        explanation: Explanation,
        rationale: List,
        add_first_last=True,
    ) -> ExplanationWithRationale:
        if rationale == NONE_RATIONALE:
            return explanation
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
            explanation.target,
            rationale,
        )

    def _get_class_explanations_by_explainer(self, class_explanations):
        """
        We convert from #target, #explainer to #explainer, #target
        """
        class_explanations_by_explainer = None
        if class_explanations is not None:
            n_explainers = len(class_explanations[0])
            n_targets = len(class_explanations)
            class_explanations_by_explainer = [
                [class_explanations[i][explainer_type] for i in range(n_targets)]
                for explainer_type in range(n_explainers)
            ]
        return class_explanations_by_explainer

    def _forward(self, text):
        item = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**item)
        return outputs

    def score(self, text: str, return_dict: bool = True):
        """Compute prediction scores for a single query

        :param text str: query to compute the logits from
        :param return_dict bool: return a dict in the format Class Label -> score. Otherwise, return softmaxed logits as torch.Tensor. Default True
        """
        return self.helper._score(text, return_dict)

    def get_dataframe(self, explanations) -> pd.DataFrame:
        scores = {e.explainer: e.scores for e in explanations}
        scores["Token"] = explanations[0].tokens
        table = pd.DataFrame(scores).set_index("Token").T
        return table

    def show_table(
        self, explanations, apply_style: bool = True, remove_first_last: bool = True
    ) -> pd.DataFrame:
        """Format explanations scores into a colored table."""
        table = self.get_dataframe(explanations)
        if remove_first_last:
            table = table.iloc[:, 1:-1]

        # Rename duplicate columns (tokens) by adding a suffix
        if sum(table.columns.duplicated().astype(int)) > 0:
            table.columns = pd.io.parsers.base_parser.ParserBase(
                {"names": table.columns, "usecols": None}
            )._maybe_dedup_names(table.columns)

        return (
            table.style.background_gradient(
                axis=1, cmap=SCORES_PALETTE, vmin=-1, vmax=1
            ).format("{:.2f}")
            if apply_style
            else table.style.format("{:.2f}")
        )

    def show_evaluation_table(
        self,
        explanation_evaluations: List[ExplanationEvaluation],
        apply_style: bool = True,
    ) -> pd.DataFrame:
        """Format explanations and evaluations scores into a colored table."""

        explainer_scores = {}
        for explanation_evaluation in explanation_evaluations:
            explainer_scores[explanation_evaluation.explanation.explainer] = {
                evaluation.name: evaluation.score
                for evaluation in explanation_evaluation.evaluation_scores
            }

        table = pd.DataFrame(explainer_scores).T

        if apply_style:
            table_style = self.style_evaluation(table)
            return table_style.format("{:.2f}")
        else:
            return table.format("{:.2f}")

    def style_evaluation(self, table):
        table_style = table.style.background_gradient(
            axis=1, cmap=SCORES_PALETTE, vmin=-1, vmax=1
        )

        show_higher_cols, show_lower_cols = list(), list()
        # Highlight with two different palettes
        for evaluation_measure in self.evaluators + self.class_based_evaluators:
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
                cmap=EVALUATION_PALETTE,
                vmin=-1,
                vmax=1,
                subset=show_higher_cols,
            )

        if show_lower_cols:
            table_style.background_gradient(
                axis=1,
                cmap=EVALUATION_PALETTE_REVERSED,
                vmin=-1,
                vmax=1,
                subset=show_lower_cols,
            )
        return table_style

    def load_dataset(self, dataset_name: str, **kwargs):
        if dataset_name == "hatexplain":
            data = HateXplainDataset(self.tokenizer)
        elif dataset_name == "movie_rationales":
            data = MovieReviews(self.tokenizer)
        elif dataset_name == "sst":
            data = SSTDataset(self.tokenizer)
        else:
            try:
                data = datasets.load_dataset(dataset_name)
            except:
                raise ValueError(f"Dataset {dataset_name} is not supported")
        return data

    def evaluate_samples(
        self,
        dataset: BaseDataset,
        sample: Union[int, List[int]],
        target=None,
        show_progress_bar: bool = True,
        n_workers: int = 1,
        **evaluation_args,
    ) -> Dict:
        """Explain a dataset sample, evaluate explanations, and compute average scores."""

        #  Use list to index datasets
        if isinstance(sample, int):
            sample = [sample]

        sample = list(map(int, sample))
        instances = [dataset[s] for s in sample]

        # For the IOU and Token F1 plausibility scores we specify the K for deriving the top-k rationale
        # As in DeYoung et al. 2020, we set it as the average size of the human rationales of the dataset
        evaluation_args["top_k_rationale"] = dataset.avg_rationale_size

        # Default, w.r.t. predicted class
        if target is None:
            #  Compute explanations for the predicted class
            predicted_classes = [
                self.score(i["text"], return_dict=False).argmax(-1).tolist()
                for i in instances
            ]

            targets = predicted_classes
        else:
            targets = [target] * len(sample)

        if show_progress_bar:
            pbar = tqdm(total=len(targets), desc="explain", leave=False)

        # Create an empty dict of dict to collect the results
        evaluation_scores_by_explainer = {}
        for explainer in self.explainers:
            evaluation_scores_by_explainer[explainer.NAME] = {}
            for evaluator in self.evaluators:
                evaluation_scores_by_explainer[explainer.NAME][evaluator.SHORT_NAME] = []

        if n_workers > 1:
            raise NotImplementedError()
            #  Parallel(n_jobs=2)(delayed(sqrt)(i ** 2) for i in range(10))
        else:

            for instance, target in zip(instances, targets):
                # Generate explanations - list of explanations (one for each explainers)
                explanations = self.explain(instance["text"], target, progress_bar=False)
                # If available, we add the human rationale
                # It will be used in the evaluation of plausibility
                if "rationale" in instance and len(instance["rationale"]) > target:
                    # Add the human rationale for the corresponding class
                    explanations = [
                        self._add_rationale(explanation, instance["rationale"][target])
                        for explanation in explanations
                    ]

                for explanation in explanations:
                    # We evaluate the explanation and we obtain an ExplanationEvaluation
                    evaluation = self.evaluate_explanation(
                        explanation, target, progress_bar=False, **evaluation_args
                    )
                    # We accumulate the results for each explainer
                    for evaluation_score in evaluation.evaluation_scores:
                        evaluation_scores_by_explainer[explanation.explainer][
                            evaluation_score.name
                        ].append(evaluation_score.score)
                if show_progress_bar:
                    pbar.update(1)

        # We compute mean and std, separately for each explainer and evaluator
        for explainer in evaluation_scores_by_explainer:
            for score_name, list_scores in evaluation_scores_by_explainer[
                explainer
            ].items():
                evaluation_scores_by_explainer[explainer][score_name] = (
                    np.mean(list_scores),
                    np.std(list_scores),
                )

        if show_progress_bar:
            pbar.close()

        return evaluation_scores_by_explainer

    def show_samples_evaluation_table(
        self,
        evaluation_scores_by_explainer,
        apply_style: bool = True,
    ) -> pd.DataFrame:
        """Format dataset average evaluations scores into a colored table."""

        # We only vizualize the average
        table = pd.DataFrame(
            {
                explainer: {
                    evaluator: mean_std[0] for evaluator, mean_std in inner.items()
                }
                for explainer, inner in evaluation_scores_by_explainer.items()
            }
        ).T

        # Avoid visualizing a columns with all nan (default value if plausibility could not computed)
        table = table.dropna(axis=1, how="all")

        if apply_style:
            table_style = self.style_evaluation(table)
            return table_style
        else:
            return table
