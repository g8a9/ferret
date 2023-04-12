"""Client Interface Module"""

import copy
from typing import Dict, List, Union

import datasets
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm.auto import tqdm

from .datasets import BaseDataset
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
from .explainers.gradient import GradientExplainer, IntegratedGradientExplainer
from .explainers.lime import LIMEExplainer
from .explainers.shap import SHAPExplainer
from .model_utils import ModelHelper

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
        explainers: List = None,
        evaluators: List = None,
        class_based_evaluators: List = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.helper = ModelHelper(self.model, self.tokenizer)

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
                ev(self.model, self.tokenizer) for ev in self._used_evaluators
            ]
        if not class_based_evaluators:
            self._used_class_evaluators = [AOPC_Comprehensiveness_Evaluation_by_class]
            self.class_based_evaluators = [
                class_ev(self.model, self.tokenizer)
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
        """
        Compute explanations using all the explainers stored in the class.

        Parameters
        ----------
        text : str
            Text string to explain.
        target : int
            Class label to produce the explanations for.
        show_progress : bool, default False
            Enable progress bar.
        normalize_scores : bool, default True
            Apply lp-normalization across tokens to make attribution weights comparable across different explainers.
        order : int, default 1
            If *normalize_scores=True*, this is the normalization order, as passed to *numpy.linalg.norm*.

        Returns
        -------

        List[Explanation]
            List of all explanations produced.

        Notes
        -----

        Please reference to :ref:`User Guide <explaining>` for more information.

        Examples
        --------
        >>> bench = Benchmark(model, tokenizer)
        >>> explanations = bench.explain("I love your style!", target=2)

        Please note that by default we apply L1 normalization across tokens, to make feature attribution weights comparable among explainers. To turn it off, you should use:

        >>> bench = Benchmark(model, tokenizer)
        >>> explanations = bench.explain("I love your style!", target=2, normalize_scores=False)
        """

        if show_progress:
            pbar = tqdm(total=len(self.explainers), desc="Explainer", leave=False)

        explanations = list()
        for exp in self.explainers:
            explanations.append(exp(text, target))
            if show_progress:
                pbar.update(1)

        if show_progress:
            pbar.close()

        if normalize_scores:
            explanations = lp_normalize(explanations, order)

        return explanations

    def evaluate_explanation(
        self,
        explanation: Union[Explanation, ExplanationWithRationale],
        target,
        human_rationale=None,
        class_explanation: List[Union[Explanation, ExplanationWithRationale]] = None,
        show_progress: bool = True,
        **evaluation_args,
    ) -> ExplanationEvaluation:

        """Evaluate an explanation using all the evaluators stored in the class.

        Args:
            explanation (Union[Explanation, ExplanationWithRationale]): explanation to evaluate.
            target (int): class label for which the explanation is evaluated
            human_rationale (list): list with values 0 or 1. A value of 1 means that the corresponding token is part of the human (or ground truth) rationale, 0 otherwise.  Tokens are indexed by position. The size of the list is the number of tokens.
            class_explanation (list): list of explanations. The explanation in position i is computed using as target class the class label i. The size is #target classes. If available, class-based scores are computed.
            show_progress (bool): enable progress bar

        Returns:
            ExplanationEvaluation: the evaluation of the explanation
        """

        evaluations = list()

        if show_progress:
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

        for evaluator in self.evaluators:
            evaluation = evaluator.compute_evaluation(
                explanation, target, **evaluation_args
            )
            if (
                evaluation is not None
            ):  # return None for plausibility measure if rationale is not available
                evaluations.append(evaluation)
            if show_progress:
                pbar.update(1)

        if class_explanation is not None:
            for class_based_evaluator in self.class_based_evaluators:
                class_based_evaluation = class_based_evaluator.compute_evaluation(
                    class_explanation, **evaluation_args
                )
                evaluations.append(class_based_evaluation)
                if show_progress:
                    pbar.update(1)

        if show_progress:
            pbar.close()
        explanation_eval = ExplanationEvaluation(explanation, evaluations)

        return explanation_eval

    def evaluate_explanations(
        self,
        explanations: List[Union[Explanation, ExplanationWithRationale]],
        target,
        human_rationale=None,
        class_explanations=None,
        show_progress=True,
        **evaluation_args,
    ) -> List[ExplanationEvaluation]:

        """Evaluate explanations using all the evaluators stored in the class.

        Args:
            explanation ( List[Union[Explanation, ExplanationWithRationale]]): list of explanations to evaluate.
            target (int): class label for which the explanations are evaluated
            human rationale (list): one-hot-encoding indicating if the token is in the human rationale (1) or not (0). If available, all explanations are evaluated for the human rationale (if provided)
            class_explanation (list): list of list of explanations. The k-th element represents the list of explanations computed varying the target class: the explanation in position k, i is computed using as target class the class label i. The size is # explanation, #target classes. If available, class-based scores are computed.
            show_progress (bool): enable progress bar

        Returns:
            List[ExplanationEvaluation]: the evaluation for each explanation
        """

        explanation_evaluations = list()

        class_explanations_by_explainer = self._get_class_explanations_by_explainer(
            class_explanations
        )
        if show_progress:
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
                    show_progress=False,
                    **evaluation_args,
                )
            )
            if show_progress:
                pbar.update(1)
        if show_progress:
            pbar.close()
        return explanation_evaluations

    def _add_rationale(
        self,
        explanation: Explanation,
        rationale: List,
        add_first_last=True,
    ) -> ExplanationWithRationale:

        """Add the ground truth rationale to the explanation.

        Args:
            explanation (Explanation): explanation
            rationale (list): one-hot-encoding indicating if the token is in the human rationale (1) or not (0)
            add_first_last (bool): consider the first and last tokens. Set it to True if the scores of the explanation also include the importance of the first and last tokens (typically cls and eos tokens)

        Returns:
            ExplanationWithRationale: explanation with the ground truth rationale
        """

        if rationale == NONE_RATIONALE:
            return explanation
        else:
            if add_first_last:
                # Include the first and last token (0 as default)
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

        _, logits = self.helper._forward(text, output_hidden_states=False)
        scores = logits[0].softmax(-1)

        if return_dict:
            scores = {
                self.model.config.id2label[idx]: value.item()
                for idx, value in enumerate(scores)
            }
        return scores

    def get_dataframe(self, explanations: List[Explanation]) -> pd.DataFrame:
        """Convert explanations into a pandas DataFrame.

        Args:
            explanations (List[Explanation]): list of explanations

        Returns:
            pd.DataFrame: explanations in table format. The columns are the tokens and the rows are the explanation scores, one for each explainer.
        """
        scores = {e.explainer: e.scores for e in explanations}
        scores["Token"] = explanations[0].tokens
        table = pd.DataFrame(scores).set_index("Token").T
        return table

    def show_table(
        self,
        explanations: List[Explanation],
        apply_style: bool = True,
        remove_first_last: bool = True,
    ) -> pd.DataFrame:
        """Format explanation scores into a colored table.

        Args:
            explanations (List[Explanation]): list of explanations
            apply_style (bool): apply color to the table of explanation scores
            remove_first_last (bool): do not visualize the first and last tokens, typically cls and eos tokens

        Returns:
            pd.DataFrame: a colored (styled) pandas dataframed
        """

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
        """Format evaluation scores into a colored table.

        Args:
            explanation_evaluations (List[ExplanationEvaluation]): a list of evaluations of explanations
            apply_style (bool): color the table of evaluation scores

        Returns:
            pd.DataFrame: a colored (styled) pandas dataframe of evaluation scores
        """

        # Get the evaluation scores from the explanation evaluations
        explainer_scores = {}
        for explanation_evaluation in explanation_evaluations:
            explainer_scores[explanation_evaluation.explanation.explainer] = {
                evaluation.name: evaluation.score
                for evaluation in explanation_evaluation.evaluation_scores
            }

        table = pd.DataFrame(explainer_scores).T

        if apply_style:
            table_style = self._style_evaluation(table)
            return table_style.format("{:.2f}")
        else:
            return table.format("{:.2f}")

    def _style_evaluation(self, table: pd.DataFrame) -> pd.DataFrame:

        """Apply style to evaluation scores.

        Args:
            table (pd.DataFrame): the evaluation scores as pandas DataFrame

        Returns:
            pd.io.formats.style.Styler: a colored and styled pandas dataframe of evaluation scores
        """

        table_style = table.style.background_gradient(
            axis=1, cmap=SCORES_PALETTE, vmin=-1, vmax=1
        )

        show_higher_cols, show_lower_cols = list(), list()

        # Color differently the evaluation measures for which "high score is better" or "low score is better"
        # Darker colors mean better performance
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
        target: int = None,
        show_progress_bar: bool = True,
        n_workers: int = 1,
        **evaluation_args,
    ) -> Dict:
        """Explain a dataset sample, evaluate explanations, and compute average scores.

        Args:
            dataset (BaseDataset): XAI dataset to explain and evaluate
            sample (Union[int, List[int]]): index or list of indexes
            target (int): class label for which the explanations are computed and evaluated. If None, explanations are computed and evaluated for the predicted class
            show_progress (bool): enable progress bar
            n_workers (int) : number of workers

        Returns:
            Dict : the average evaluation scores and their standard deviation for each explainer. The form is the following: {explainer: {"evaluation_measure": (avg_score, std)}
        """

        #  Use list to index datasets
        if isinstance(sample, int):
            sample = [sample]

        sample = list(map(int, sample))
        instances = [dataset[s] for s in sample]

        # For the IOU and Token F1 plausibility scores we specify the K for deriving the top-k rationale
        # As in DeYoung et al. 2020, we set it as the average size of the human rationales of the dataset
        evaluation_args["top_k_rationale"] = dataset.avg_rationale_size

        # is_thermostatdata = isinstance(dataset, ThermostatDataset) --> problem with reload
        is_thermostatdata = dataset.NAME == "Thermostat"

        # Set the explanation target class
        if is_thermostatdata:
            # The explanations in thermostat are pre-computed for the predicted class
            targets = [i["predicted_label"] for i in instances]
        else:
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

        if is_thermostatdata:
            name_explainers = dataset.explainers
        else:
            name_explainers = [e.NAME for e in self.explainers]

        if show_progress_bar:
            pbar = tqdm(total=len(targets), desc="explain", leave=False)

        # Create an empty dict of dict to collect the results
        evaluation_scores_by_explainer = {}
        for explainer in name_explainers:
            evaluation_scores_by_explainer[explainer] = {}
            for evaluator in self.evaluators:
                evaluation_scores_by_explainer[explainer][evaluator.SHORT_NAME] = []

        if n_workers > 1:
            raise NotImplementedError()
            #  Parallel(n_jobs=2)(delayed(sqrt)(i ** 2) for i in range(10))
        else:

            for instance, target in zip(instances, targets):

                if is_thermostatdata:
                    # If it is Thermostat instance, explanation are already pre-computed
                    explanations = instance["explanations"]
                else:
                    # We generate explanations - list of explanations (one for each explainers)
                    explanations = self.explain(
                        instance["text"], target, show_progress=False
                    )
                    # If available, we add the human rationale
                    # It will be used in the evaluation of plausibility
                    if "rationale" in instance and len(instance["rationale"]) > target:
                        # Add the human rationale for the corresponding class
                        explanations = [
                            self._add_rationale(
                                explanation, instance["rationale"][target]
                            )
                            for explanation in explanations
                        ]

                for explanation in explanations:
                    # We evaluate the explanation and we obtain an ExplanationEvaluation
                    evaluation = self.evaluate_explanation(
                        explanation, target, show_progress=False, **evaluation_args
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
            for score_name in list(evaluation_scores_by_explainer[explainer]):
                list_scores = evaluation_scores_by_explainer[explainer][score_name]
                if list_scores:
                    # Compute mean and standard deviation
                    evaluation_scores_by_explainer[explainer][score_name] = (
                        np.mean(list_scores),
                        np.std(list_scores),
                    )
                else:
                    evaluation_scores_by_explainer[explainer].pop(score_name, None)

        if show_progress_bar:
            pbar.close()

        return evaluation_scores_by_explainer

    def show_samples_evaluation_table(
        self,
        evaluation_scores_by_explainer,
        apply_style: bool = True,
    ) -> pd.DataFrame:
        """Format average evaluation scores into a colored table.

        Args:
            evaluation_scores_by_explainer (Dict): the average evaluation scores and their standard deviation for each explainer (output of the evaluate_samples function)
             apply_style (bool): color the table of average evaluation scores

        Returns:
            pd.DataFrame: a colored (styled) pandas dataframe of average evaluation scores of explanations of a sample
        """

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
            table_style = self._style_evaluation(table)
            return table_style
        else:
            return table
