import copy
import numpy as np
import pandas as pd
from typing import List

from ..modelw import Model

from .faithfulness_measures import (
    AOPC_Comprehensiveness_Evaluation,
    AOPC_Sufficiency_Evaluation,
    TauLOO_Evaluation,
)
from .plausibility_measures import (
    AUPRC_PlausibilityEvaluation,
    Tokenf1_PlausibilityEvaluation,
    TokenIOU_PlausibilityEvaluation,
)

from .classes_evaluation_measures import (
    AOPC_Comprehensiveness_Evaluation_by_class,
    AOPC_Sufficiency_Evaluation_by_class,
)
from ..evaluation import BaseEvaluator


def color_nan_black(val):
    """Color the nan text black"""
    if np.isnan(val):
        return "color: black"


def color_nan_white_background(val):
    """Color the nan cell background white"""
    if np.isnan(val):
        return "background-color: white"


class ExplanationEvalutator:
    def __init__(
        self, model, tokenizer, evaluation_metrics: List[BaseEvaluator] = None, **kwargs
    ):
        self.modelw = Model(model)
        self.tokenizer = tokenizer
        self.faithfulness_metrics = []
        self.plausibility_metrics = []
        self.class_metrics = []
        self.other_metrics = []

        if evaluation_metrics is None:
            # We use all the default evaluation metrics.

            compr_eval = AOPC_Comprehensiveness_Evaluation(
                self.modelw,
                self.tokenizer,
            )
            self.faithfulness_metrics.append(compr_eval)

            suff_eval = AOPC_Sufficiency_Evaluation(self.modelw, self.tokenizer)
            self.faithfulness_metrics.append(suff_eval)

            use_correlation = kwargs.get("use_correlation", True)

            tau_eval = TauLOO_Evaluation(
                self.modelw, self.tokenizer, use_correlation=use_correlation
            )
            self.faithfulness_metrics.append(tau_eval)

            class_compr_eval = AOPC_Comprehensiveness_Evaluation_by_class(
                aopc_compr_eval=compr_eval
            )
            self.class_metrics.append(class_compr_eval)

            class_suff_eval = AOPC_Sufficiency_Evaluation_by_class(
                aopc_suff_eval=suff_eval
            )
            self.class_metrics.append(class_suff_eval)

            use_plausibility_metrics = kwargs.get("use_plausibility_metrics", True)

            if use_plausibility_metrics:
                auprc_plausibility_eval = AUPRC_PlausibilityEvaluation(
                    self.modelw, self.tokenizer
                )
                self.plausibility_metrics.append(auprc_plausibility_eval)

                tokenf1_eval = Tokenf1_PlausibilityEvaluation(
                    self.modelw, self.tokenizer
                )
                self.plausibility_metrics.append(tokenf1_eval)

                tokeniou_eval = TokenIOU_PlausibilityEvaluation(
                    self.modelw, self.tokenizer
                )
                self.plausibility_metrics.append(tokeniou_eval)

        else:
            # TODO class measures
            for evaluation_metric in evaluation_metrics:
                if (
                    isinstance(evaluation_metric, BaseEvaluator) == False
                    and evaluation_metric.TYPE_METRIC != "class_faithfulness"
                ):
                    raise ValueError(f"{evaluation_metric} not supported")
                if evaluation_metric.TYPE_METRIC == "faithfulness":
                    self.faithfulness_metrics.append(evaluation_metric)
                elif evaluation_metric.TYPE_METRIC == "plausibility":
                    self.plausibility_metrics.append(evaluation_metric)
                elif evaluation_metric.TYPE_METRIC == "class_faithfulness":
                    self.class_metrics.append(evaluation_metric)
                else:
                    self.other_metrics.append(evaluation_metric)

    def get_true_rational_tokens(
        self, original_tokens: List[str], rationale_original_tokens: List[int]
    ) -> List[int]:
        # original_tokens --> list of words.
        # rationale_original_tokens --> 0 or 1, if the token belongs to the rationale or not
        # Typically, the importance is associated with each word rather than each token.
        # We convert each word in token using the tokenizer. If a word is in the rationale,
        # we consider as important all the tokens of the word.
        token_rationale = []
        for t, rationale_t in zip(original_tokens, rationale_original_tokens):
            converted_token = self.tokenizer.encode(t)[1:-1]

            for token_i in converted_token:
                token_rationale.append(rationale_t)
        return token_rationale

    def evaluate_explainers(
        self,
        text,
        explanations,
        true_rationale=None,
        rank_explainer=True,
        style_df=True,
        target=1,
        explanations_by_target=None,
        **evaluation_args,
    ):
        evaluation_scores = {}

        df_eval = copy.deepcopy(explanations)

        # Rename duplicate columns (tokens) by adding a suffix
        if sum(explanations.columns.duplicated().astype(int)) > 0:
            df_eval.columns = pd.io.parsers.base_parser.ParserBase(
                {"names": explanations.columns, "usecols": None}
            )._maybe_dedup_names(explanations.columns)

        text_tokens = list(df_eval.columns)

        for explainer_type in explanations.index:
            soft_score_explanation = explanations.loc[explainer_type].values

            # Faithfulness

            for faithfulness_measure in self.faithfulness_metrics:
                if faithfulness_measure.SHORT_NAME not in evaluation_scores:
                    evaluation_scores[faithfulness_measure.SHORT_NAME] = {}

                evaluation_scores[faithfulness_measure.SHORT_NAME][
                    explainer_type
                ] = faithfulness_measure.evaluate_explanation(
                    text,
                    soft_score_explanation,
                    target=target,
                    **evaluation_args,
                )

            if true_rationale is not None and len(self.plausibility_metrics) > 0:
                # We can compute the plausibility metrics.

                for plausibility_measure in self.plausibility_metrics:

                    if plausibility_measure.SHORT_NAME not in evaluation_scores:
                        evaluation_scores[plausibility_measure.SHORT_NAME] = {}

                    evaluation_scores[plausibility_measure.SHORT_NAME][
                        explainer_type
                    ] = plausibility_measure.evaluate_explanation(
                        text,
                        soft_score_explanation,
                        true_rationale,
                        **evaluation_args,
                    )

            for other_measure in self.other_metrics:
                if other_measure.SHORT_NAME not in evaluation_scores:
                    evaluation_scores[other_measure.SHORT_NAME] = {}

                evaluation_scores[other_measure.SHORT_NAME][
                    explainer_type
                ] = other_measure.evaluate_explanation(
                    text,
                    soft_score_explanation,
                    target=target,
                    **evaluation_args,
                )

        if explanations_by_target is not None:
            for class_measure in self.class_metrics:

                if class_measure.SHORT_NAME not in evaluation_scores:
                    evaluation_scores[class_measure.SHORT_NAME] = {}
                evaluation_scores[
                    class_measure.SHORT_NAME
                ] = class_measure.evaluate_class_explanation(
                    text,
                    explanations_by_target,
                    **evaluation_args,
                )

        df_eval = pd.concat([df_eval, pd.DataFrame(evaluation_scores)], axis=1)

        df_style = None

        if rank_explainer:
            df_eval = self._rank_explainers(df_eval, text_tokens)

        if style_df:
            df_style = self._style_result(df_eval)

        return df_eval, df_style

    # Temporary
    def _style_result(self, df_eval, th=None):
        import seaborn as sns

        palette = sns.diverging_palette(240, 10, as_cmap=True)
        df_st = df_eval.style.background_gradient(axis=1, cmap=palette, vmin=-1, vmax=1)

        evaluation_measures = (
            self.faithfulness_metrics
            + self.plausibility_metrics
            + self.class_metrics
            + self.other_metrics
        )

        # Higher is better
        show_higher_cols = [
            evaluation_measure.SHORT_NAME
            for evaluation_measure in evaluation_measures
            if evaluation_measure.BEST_SORTING_ASCENDING == False
            and evaluation_measure.SHORT_NAME in df_eval.columns
        ]
        if show_higher_cols:
            palette = sns.diverging_palette(150, 275, s=80, l=55, n=9, as_cmap=True)
            df_st.background_gradient(
                axis=1, cmap=palette, vmin=-1, vmax=1, subset=show_higher_cols
            )

        # Lower is better
        show_lower_cols = [
            evaluation_measure.SHORT_NAME
            for evaluation_measure in evaluation_measures
            if evaluation_measure.BEST_SORTING_ASCENDING == True
            and evaluation_measure.SHORT_NAME in df_eval.columns
        ]

        if show_lower_cols:
            palette = sns.light_palette("blue", as_cmap=True, reverse=True)
            df_st.background_gradient(
                axis=1, cmap=palette, vmin=-1, vmax=1, subset=show_lower_cols
            )

        subset_columns_ranking = [
            f"{v}_r"
            for v in show_higher_cols + show_lower_cols
            if f"{v}_r" in df_eval.columns
        ]
        if subset_columns_ranking:
            df_st.background_gradient(
                axis=1,
                cmap=sns.light_palette("lightblue", as_cmap=True, reverse=True),
                vmin=1,
                vmax=len(df_eval),
                subset=subset_columns_ranking,
            )

        df_st.applymap(lambda x: color_nan_black(x)).applymap(
            lambda x: color_nan_white_background(x)
        )
        return df_st

    def _rank_explainers(self, df_eval, score_cols, th=1):

        from scipy import stats

        cols = list(df_eval.columns)

        evaluation_measures = (
            self.faithfulness_metrics
            + self.plausibility_metrics
            + self.class_metrics
            + self.other_metrics
        )

        # Higher is better
        show_higher_cols = [
            evaluation_measure.SHORT_NAME
            for evaluation_measure in evaluation_measures
            if evaluation_measure.BEST_SORTING_ASCENDING == False
            and evaluation_measure.SHORT_NAME in df_eval.columns
        ]

        for c in show_higher_cols:
            r = stats.rankdata(df_eval[c], method="dense")
            df_eval[f"{c}_r"] = min(len(df_eval[c]), max(r)) + 1 - r

        # Lower is better
        show_lower_cols = [
            evaluation_measure.SHORT_NAME
            for evaluation_measure in evaluation_measures
            if evaluation_measure.BEST_SORTING_ASCENDING == True
            and evaluation_measure.SHORT_NAME in df_eval.columns
        ]

        for c in show_lower_cols:
            df_eval[f"{c}_r"] = stats.rankdata(df_eval[c], method="dense")

        # Just to show the columns in order
        # First the scores, then faithfulness_metrics, plausibility_metrics, class_metrics and then the others.
        show_cols = []

        for metric_show in [m.SHORT_NAME for m in evaluation_measures]:

            if metric_show in show_higher_cols + show_lower_cols:
                show_cols.extend([metric_show, f"{metric_show}_r"])

        output_cols = (
            list(score_cols)
            + show_cols
            + [c for c in cols if c not in list(score_cols) + show_cols]
        )
        return df_eval[output_cols]

    def evaluate_explainers_globally(
        self,
        explainer_obj,
        texts,
        true_rationales=None,
        rank_explainer=True,
        style_df=True,
        classes=[0, 1],
        **evaluation_args,
    ):
        evaluation_args["accumulate_result"] = True

        accumulated_results = {}
        explanations_by_target = {}

        for e, text in enumerate(texts):
            true_rationale = None if true_rationales is None else true_rationales[e]

            target = self.modelw.get_predicted_label(text, self.tokenizer)
            for class_name in classes:
                explanations_by_target[class_name] = explainer_obj.compute_table(
                    text, target=class_name
                )
            explanations = explanations_by_target[target]

            for explainer_type in explanations.index:
                if explainer_type not in accumulated_results:
                    accumulated_results[explainer_type] = {}

                soft_score_explanation = explanations.loc[explainer_type].values

                for faithfulness_measure in self.faithfulness_metrics:
                    if (
                        faithfulness_measure.SHORT_NAME
                        not in accumulated_results[explainer_type]
                    ):
                        accumulated_results[explainer_type][
                            faithfulness_measure.SHORT_NAME
                        ] = copy.deepcopy(faithfulness_measure.INIT_VALUE)

                    accumulated_results[explainer_type][
                        faithfulness_measure.SHORT_NAME
                    ] += faithfulness_measure.evaluate_explanation(
                        text,
                        soft_score_explanation,
                        target=target,
                        **evaluation_args,
                    )

                if true_rationale is not None and len(self.plausibility_metrics) > 0:
                    # We can compute the plausibility metrics.

                    for plausibility_measure in self.plausibility_metrics:

                        if (
                            plausibility_measure.SHORT_NAME
                            not in accumulated_results[explainer_type]
                        ):
                            accumulated_results[explainer_type][
                                plausibility_measure.SHORT_NAME
                            ] = copy.deepcopy(plausibility_measure.INIT_VALUE)

                        accumulated_results[explainer_type][
                            plausibility_measure.SHORT_NAME
                        ] += plausibility_measure.evaluate_explanation(
                            text,
                            soft_score_explanation,
                            true_rationale,
                            **evaluation_args,
                        )

                for other_measure in self.other_metrics:
                    if (
                        other_measure.SHORT_NAME
                        not in accumulated_results[explainer_type]
                    ):
                        accumulated_results[explainer_type][
                            other_measure.SHORT_NAME
                        ] = copy.deepcopy(other_measure.INIT_VALUE)

                    accumulated_results[explainer_type][
                        other_measure.SHORT_NAME
                    ] += other_measure.evaluate_explanation(
                        text,
                        soft_score_explanation,
                        target=target,
                        **evaluation_args,
                    )

            for class_measure in self.class_metrics:
                for explainer_type in explanations.index:
                    if (
                        class_measure.SHORT_NAME
                        not in accumulated_results[explainer_type]
                    ):
                        accumulated_results[explainer_type][
                            class_measure.SHORT_NAME
                        ] = copy.deepcopy(class_measure.INIT_VALUE)

                result_by_explainer = class_measure.evaluate_class_explanation(
                    text,
                    explanations_by_target,
                    **evaluation_args,
                )
                for explainer_name, score in result_by_explainer.items():
                    accumulated_results[explainer_name][
                        class_measure.SHORT_NAME
                    ] += score

        # TODO

        evaluated_measures = copy.deepcopy(self.faithfulness_metrics)
        if true_rationales is not None and len(self.plausibility_metrics) > 0:
            evaluated_measures.extend(self.plausibility_metrics)
        evaluated_measures.extend(self.other_metrics + self.class_metrics)

        average_score = {}

        n_explanations = len(texts)
        for explainer_type, measure_score in accumulated_results.items():
            if explainer_type not in average_score:
                average_score[explainer_type] = {}
            for e, (measure, score) in enumerate(measure_score.items()):
                assert evaluated_measures[e].SHORT_NAME == measure
                average_score[explainer_type][measure] = evaluated_measures[
                    e
                ].aggregate_score(score, n_explanations, average="macro")

        average_score_df = pd.DataFrame(average_score).T

        if rank_explainer:
            average_score_df = self._rank_explainers(average_score_df, [])

        if style_df:
            df_style = self._style_result(average_score_df)

        return average_score_df, df_style
