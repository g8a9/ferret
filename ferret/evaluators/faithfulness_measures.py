from . import BaseEvaluator
import copy
from ferret.modelw import Model
from ferret.explainers.explanation import Explanation, ExplanationWithRationale
import numpy as np
from typing import List
from .utils_from_soft_to_discrete import (
    _check_and_define_get_id_discrete_rationale_function,
)
from .evaluation import Evaluation


def _compute_aopc(scores):
    from statistics import mean

    return mean(scores)


class AOPC_Comprehensiveness_Evaluation(BaseEvaluator):
    NAME = "aopc_comprehensiveness"
    SHORT_NAME = "aopc_compr"
    # Higher is better
    BEST_SORTING_ASCENDING = False
    TYPE_METRIC = "faithfulness"

    def compute_evaluation(self, explanation: Explanation, target=1, **evaluation_args):
        remove_first_last = evaluation_args.get("remove_first_last", True)
        only_pos = evaluation_args.get("only_pos", False)

        text = explanation.text
        score_explanation = explanation.scores

        # TO DO - use tokens
        # Get prediction probability of the input sencence for the target
        baseline = self.model._get_class_predicted_probability(
            text, self.tokenizer, target
        )

        # Tokenized sentence
        item = self.tokenizer(text, return_tensors="pt")

        # Get token ids of the sentence
        input_len = item["attention_mask"].sum().item()
        input_ids = item["input_ids"][0][:input_len].tolist()

        # If remove_first_last, first and last token id (CLS and ) are removed
        if remove_first_last == True:
            input_ids = input_ids[1:-1]
            if self.tokenizer.cls_token == explanation.tokens[0]:
                score_explanation = score_explanation[1:-1]

        # Defaul parameters
        removal_args = {
            "remove_tokens": True,
            "based_on": "k",
            "thresholds": range(1, len(input_ids) + 1),
        }
        removal_args_input = evaluation_args.get("removal_args", None)
        if removal_args_input:
            removal_args.update(removal_args_input)

        discrete_expl_ths = []
        id_tops = []

        """
        We currently support multiple approaches to define the hard rationale from
        soft score rationales, based on:
        - th : token greater than a threshold
        - perc : more than x% of the tokens
        - k: top k values
        """

        get_discrete_rationale_function = (
            _check_and_define_get_id_discrete_rationale_function(
                removal_args["based_on"]
            )
        )

        thresholds = removal_args["thresholds"]
        last_id_top = None
        for v in thresholds:

            # Get rationale from score explanation
            id_top = get_discrete_rationale_function(score_explanation, v, only_pos)

            # If the rationale is the same, we do not include it. In this way, we will not consider in the average the same omission.
            if last_id_top is not None and set(id_top) == last_id_top:
                id_top = None

            id_tops.append(id_top)

            if id_top is None:
                continue

            last_id_top = set(id_top)

            # Comprehensiveness
            # The only difference between comprehesivenss and sufficiency is the computation of the removal.

            # For the comprehensiveness: we remove the terms in the discrete rationale.

            sample = np.array(copy.copy(input_ids))

            if removal_args["remove_tokens"]:
                discrete_expl_th_token_ids = np.delete(sample, id_top)
            else:
                sample[id_top] = self.tokenizer.mask_token_id
                discrete_expl_th_token_ids = sample

            discrete_expl_th = self.tokenizer.decode(discrete_expl_th_token_ids)

            discrete_expl_ths.append(discrete_expl_th)

        if discrete_expl_ths == []:
            return np.nan

        # Prediction probability for the target

        probs_removing = self.model._get_class_predicted_probabilities_texts(
            discrete_expl_ths, self.tokenizer, target
        ).numpy()

        # Compute probability difference
        removal_importance = baseline - probs_removing

        aopc_comprehesiveness = _compute_aopc(removal_importance)

        evaluation_output = Evaluation(self.SHORT_NAME, aopc_comprehesiveness)
        return evaluation_output

    def aggregate_score(self, score, total, **aggregation_args):
        return super().aggregate_score(score, total, **aggregation_args)


class AOPC_Sufficiency_Evaluation(BaseEvaluator):
    NAME = "aopc_sufficiency"
    SHORT_NAME = "aopc_suff"
    # Lower is better
    BEST_SORTING_ASCENDING = True
    TYPE_METRIC = "faithfulness"

    def compute_evaluation(self, explanation: Explanation, target=1, **evaluation_args):
        remove_first_last = evaluation_args.get("remove_first_last", True)
        only_pos = evaluation_args.get("only_pos", False)

        text = explanation.text
        score_explanation = explanation.scores

        # TO DO - use tokens
        # Get prediction probability of the input sencence for the target
        baseline = self.model._get_class_predicted_probability(
            text, self.tokenizer, target
        )

        # Tokenized sentence
        item = self.tokenizer(text, return_tensors="pt")
        # Get token ids of the sentence
        input_len = item["attention_mask"].sum().item()
        input_ids = item["input_ids"][0][:input_len].tolist()

        # If remove_first_last, first and last token id (CLS and ) are removed
        if remove_first_last == True:
            input_ids = input_ids[1:-1]
            if self.tokenizer.cls_token == explanation.tokens[0]:
                score_explanation = score_explanation[1:-1]

        # Defaul parameters
        removal_args = {
            "remove_tokens": True,
            "based_on": "k",
            "thresholds": range(1, len(input_ids) + 1),
        }
        removal_args_input = evaluation_args.get("removal_args", None)
        if removal_args_input:
            removal_args.update(removal_args_input)

        discrete_expl_ths = []
        id_tops = []

        """
        We currently support multiple approaches to define the hard rationale from
        soft score rationales, based on:
        - th : token greater than a threshold
        - perc : more than x% of the tokens
        - k: top k values
        """

        get_discrete_rationale_function = (
            _check_and_define_get_id_discrete_rationale_function(
                removal_args["based_on"]
            )
        )

        thresholds = removal_args["thresholds"]
        last_id_top = None
        for v in thresholds:

            # Get rationale
            id_top = get_discrete_rationale_function(score_explanation, v, only_pos)

            # If the rationale is the same, we do not include it. In this way, we will not consider in the average the same omission.
            if last_id_top is not None and set(id_top) == last_id_top:
                id_top = None

            id_tops.append(id_top)

            if id_top is None:
                continue

            last_id_top = set(id_top)

            # Sufficiency
            # The only difference between comprehesivenss and sufficiency is the computation of the removal.

            # For the sufficiency: we keep only the terms in the discrete rationale.

            sample = np.array(copy.copy(input_ids))

            if removal_args["remove_tokens"]:
                discrete_expl_th_token_ids = sample[id_top]
            else:
                sample[id_top] = self.tokenizer.mask_token_id
                discrete_expl_th_token_ids = sample
            ##############################################

            discrete_expl_th = self.tokenizer.decode(discrete_expl_th_token_ids)

            discrete_expl_ths.append(discrete_expl_th)

        if discrete_expl_ths == []:
            return np.nan

        # Prediction probability for the target

        probs_removing = self.model._get_class_predicted_probabilities_texts(
            discrete_expl_ths, self.tokenizer, target
        ).numpy()

        # Compute probability difference
        removal_importance = baseline - probs_removing

        aopc_sufficiency = _compute_aopc(removal_importance)

        evaluation_output = Evaluation(self.SHORT_NAME, aopc_sufficiency)
        return evaluation_output

    def aggregate_score(self, score, total, **aggregation_args):
        return super().aggregate_score(score, total, **aggregation_args)


class TauLOO_Evaluation(BaseEvaluator):
    NAME = "tau_leave-one-out_correlation"
    SHORT_NAME = "taucorr_loo"
    TYPE_METRIC = "faithfulness"
    BEST_SORTING_ASCENDING = False

    def compute_leave_one_out_occlusion(self, text, target=1, remove_first_last=True):

        baseline = self.model._get_class_predicted_probability(
            text, self.tokenizer, target
        )

        item = self.tokenizer(text, return_tensors="pt")
        input_len = item["attention_mask"].sum().item()
        input_ids = item["input_ids"][0][:input_len].tolist()
        if remove_first_last == True:
            input_ids = input_ids[1:-1]

        samples = list()
        for occ_idx in range(len(input_ids)):
            sample = copy.copy(input_ids)
            sample.pop(occ_idx)
            sample = self.tokenizer.decode(sample)
            samples.append(sample)

        leave_one_out_removal = self.model._get_class_predicted_probabilities_texts(
            samples, self.tokenizer, target
        )

        occlusion_importance = leave_one_out_removal - baseline

        return occlusion_importance

    def compute_evaluation(self, explanation: Explanation, target=1, **evaluation_args):

        remove_first_last = evaluation_args.get("remove_first_last", True)

        text = explanation.text
        score_explanation = explanation.scores

        if remove_first_last:
            if self.tokenizer.cls_token == explanation.tokens[0]:
                score_explanation = score_explanation[1:-1]

        loo_scores = (
            self.compute_leave_one_out_occlusion(
                text, target=target, remove_first_last=remove_first_last
            ).numpy()
            * -1
        )

        from scipy.stats import kendalltau

        kendalltau_score = kendalltau(loo_scores, score_explanation)[0]

        evaluation_output = Evaluation(self.SHORT_NAME, kendalltau_score)
        return evaluation_output

    def aggregate_score(self, score, total, **aggregation_args):
        return super().aggregate_score(score, total, **aggregation_args)
