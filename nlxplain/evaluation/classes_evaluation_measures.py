from cmath import exp
from nlxplain.evaluation.faithfulness_measures import (
    AOPC_Comprehensiveness_Evaluation,
    AOPC_Sufficiency_Evaluation,
)
from typing import List


class AOPC_Comprehensiveness_Evaluation_by_class:
    NAME = "aopc_class_comprehensiveness"
    SHORT_NAME = "aopc_class_compr"
    # Higher is better
    BEST_SORTING_ASCENDING = False
    TYPE_METRIC = "class_faithfulness"
    INIT_VALUE = 0

    def __init__(
        self,
        model=None,
        tokenizer=None,
        aopc_compr_eval: AOPC_Comprehensiveness_Evaluation = None,
    ):
        if aopc_compr_eval is None:
            if model is None or tokenizer is None:
                raise ValueError("Specify the tokenizer and the model")
            self.aopc_compr_eval = AOPC_Comprehensiveness_Evaluation(model, tokenizer)
        else:
            self.aopc_compr_eval = aopc_compr_eval

    def evaluate_class_explanation(
        self, text, explanations_by_target: List[List[float]], **evaluation_args
    ):

        evaluation_args["only_pos"] = True

        texts = [text] * len(explanations_by_target)
        auc_mean_std_class = {}
        score_explanations_by_explainer = {}
        for target in explanations_by_target:
            for explainer_type in explanations_by_target[target].index:
                if explainer_type not in score_explanations_by_explainer:
                    score_explanations_by_explainer[explainer_type] = []
                score_explanations_by_explainer[explainer_type].append(
                    explanations_by_target[target].loc[explainer_type].values
                )

        for (
            explainer_type,
            score_explanations,
        ) in score_explanations_by_explainer.items():

            auc_mean_std_class[
                explainer_type
            ] = self.aopc_compr_eval.evaluate_explanations(
                texts,
                score_explanations,
                list(explanations_by_target.keys()),
                **evaluation_args
            )

        auc_mean_class = {k: mean for k, (mean, std) in auc_mean_std_class.items()}
        return auc_mean_class

    def aggregate_score(self, score, total, **aggregation_args):
        return score / total


class AOPC_Sufficiency_Evaluation_by_class:
    NAME = "aopc_class_sufficiency"
    SHORT_NAME = "aopc_class_suff"
    # Higher is better
    BEST_SORTING_ASCENDING = True
    TYPE_METRIC = "class_faithfulness"
    INIT_VALUE = 0

    def __init__(
        self,
        model=None,
        tokenizer=None,
        aopc_suff_eval: AOPC_Sufficiency_Evaluation = None,
    ):
        if aopc_suff_eval is None:
            if model is None or tokenizer is None:
                raise ValueError("Specify the tokenizer and the model")
            self.aopc_suff_eval = AOPC_Sufficiency_Evaluation(model, tokenizer)
        else:
            self.aopc_suff_eval = aopc_suff_eval

    def evaluate_class_explanation(
        self, text, explanations_by_target: List[List[float]], **evaluation_args
    ):
        evaluation_args["only_pos"] = True

        texts = [text] * len(explanations_by_target)
        auc_mean_std_class = {}
        score_explanations_by_explainer = {}

        for target in explanations_by_target:
            for explainer_type in explanations_by_target[target].index:
                if explainer_type not in score_explanations_by_explainer:
                    score_explanations_by_explainer[explainer_type] = []
                score_explanations_by_explainer[explainer_type].append(
                    explanations_by_target[target].loc[explainer_type].values
                )

        for (
            explainer_type,
            score_explanations,
        ) in score_explanations_by_explainer.items():

            auc_mean_std_class[
                explainer_type
            ] = self.aopc_suff_eval.evaluate_explanations(
                texts,
                score_explanations,
                list(explanations_by_target.keys()),
                **evaluation_args
            )

        auc_mean_class = {k: mean for k, (mean, std) in auc_mean_std_class.items()}
        return auc_mean_class

    def aggregate_score(self, score, total, **aggregation_args):
        return score / total
