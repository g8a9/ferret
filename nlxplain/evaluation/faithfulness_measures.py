from . import BaseEvaluator
import copy
from nlxplain.modelw import Model
import numpy as np
import torch
from typing import List
from .utils_from_soft_to_discrete import (
    _check_and_define_get_id_discrete_rationale_function,
)


def _compute_aopc(scores):
    from statistics import mean

    return mean(scores)


class AOPC_Comprehensiveness_Evaluation(BaseEvaluator):
    NAME = "aopc_comprehensiveness"
    SHORT_NAME = "aopc_compr"
    # Higher is better
    BEST_SORTING_ASCENDING = False
    TYPE_METRIC = "faithfulness"

    def __init__(self, model: Model, tokenizer):
        super().__init__(model, tokenizer)

    def evaluate_explanations(
        self,
        texts: List[str],
        score_explanations: List[List[float]],
        targets,
        **evaluation_args
    ):

        """
        Compute the aggregate evaluation score for a list of score explanations.
        Return the aggregate evaluation score as average and standard deviation of aopc_comprehensiveness
        """
        scores = []
        for text, score_explanation, target in zip(texts, score_explanations, targets):
            scores.append(
                self.evaluate_explanation(
                    text, score_explanation, target=target, **evaluation_args
                )
            )
        # https://github.com/jayded/eraserbenchmark/blob/36467f1662812cbd4fbdd66879946cd7338e08ec/rationale_benchmark/metrics.py#L263
        # Average of the average score AOPC
        average_score, std = np.average(scores), np.std(scores)
        return average_score, std

    def evaluate_explanation(
        self, text, score_explanation, target=1, **evaluation_args
    ):
        remove_first_last = evaluation_args.get("remove_first_last", True)
        only_pos = evaluation_args.get("only_pos", False)

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

        """
        detailed_result = {}
        r = 0
        for i in range(len(thresholds)):
            if id_tops[i] is not None:
                detailed_result[thresholds[i]] = (
                    id_tops[i],
                    removal_importance[r].item(),
                )
                r += 1

        """

        return _compute_aopc(removal_importance)

    def aggregate_score(self, score, total, **aggregation_args):
        return super().aggregate_score(score, total, **aggregation_args)


class AOPC_Sufficiency_Evaluation(BaseEvaluator):
    NAME = "aopc_sufficiency"
    SHORT_NAME = "aopc_suff"
    # Lower is better
    BEST_SORTING_ASCENDING = True
    TYPE_METRIC = "faithfulness"

    def __init__(self, model: Model, tokenizer):
        super().__init__(model, tokenizer)

    def evaluate_explanations(
        self,
        texts: List[str],
        score_explanations: List[List[float]],
        targets,
        **evaluation_args
    ):

        """
        Compute the aggregate evaluation score for a list of score explanations.
        Return the aggregate evaluation score as average and standard deviation of aopc_sufficiency
        """
        scores = []
        for text, score_explanation, target in zip(texts, score_explanations, targets):
            scores.append(
                self.evaluate_explanation(
                    text, score_explanation, target=target, **evaluation_args
                )
            )
        # https://github.com/jayded/eraserbenchmark/blob/36467f1662812cbd4fbdd66879946cd7338e08ec/rationale_benchmark/metrics.py#L263
        # Average of the average score AOPC
        average_score, std = np.average(scores), np.std(scores)
        return average_score, std

    def evaluate_explanation(
        self, text, score_explanation, target=1, **evaluation_args
    ):
        remove_first_last = evaluation_args.get("remove_first_last", True)
        only_pos = evaluation_args.get("only_pos", False)

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

        """
        detailed_result = {}
        r = 0
        for i in range(len(thresholds)):
            if id_tops[i] is not None:
                detailed_result[thresholds[i]] = (
                    id_tops[i],
                    removal_importance[r].item(),
                )
                r += 1

        """

        return _compute_aopc(removal_importance)

    def aggregate_score(self, score, total, **aggregation_args):
        return super().aggregate_score(score, total, **aggregation_args)


class TauLOO_Evaluation(BaseEvaluator):
    NAME = "tau_leave-one-out"
    SHORT_NAME = "tauloo"
    TYPE_METRIC = "faithfulness"
    BEST_SORTING_ASCENDING = None

    def __init__(self, model: Model, tokenizer, use_correlation=True):
        super().__init__(model, tokenizer)
        self.use_correlation = use_correlation
        if self.use_correlation:
            self.SHORT_NAME = "taucorr_loo"
            # Higher is better
            self.BEST_SORTING_ASCENDING = False
        else:
            self.SHORT_NAME = "taud_loo"
            # Lower is better
            self.BEST_SORTING_ASCENDING = False

    # As in Attention is not explanation
    # https://github.com/successar/AttentionExplanation/blob/425a89a49a8b3bffc3f5e8338287e2ecd0cf1fa2/common_code/kendall_top_k.py

    def evaluate_explanations(
        self,
        texts: List[str],
        score_explanations: List[List[float]],
        targets,
        **evaluation_args
    ):
        """
        Compute the aggregate evaluation score for a list of score explanations.
        Return the aggregate evaluation score as average and standard deviation of tau_leave-one-out scores
        """
        scores = []
        for text, score_explanation, target in zip(texts, score_explanations, targets):
            scores.append(
                self.evaluate_explanation(
                    text, score_explanation, target=target, **evaluation_args
                )
            )
        # https://arxiv.org/pdf/1902.10186.pdf: mean and standard deviation.
        average_score, std = np.average(scores), np.std(scores)
        return average_score, std

    def _kendalltau_distance(self, x, y):
        from scipy.stats import kendalltau

        """
        It returns a distance: 0 for identical lists and 1 if completely different.
        """
        # https://github.com/successar/AttentionExplanation/blob/425a89a49a8b3bffc3f5e8338287e2ecd0cf1fa2/common_code/kendall_top_k.py#L23
        if x.size != y.size:
            raise NameError("The two arrays need to have same lengths")
        return 1 - (kendalltau(x, y)[0] / 2 + 0.5)

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

    def evaluate_explanation(
        self, text, score_explanation, target=1, **evaluation_args
    ):

        remove_first_last = evaluation_args.get("remove_first_last", True)

        loo_scores = (
            self.compute_leave_one_out_occlusion(
                text, target=target, remove_first_last=remove_first_last
            ).numpy()
            * -1
        )

        if self.use_correlation:
            # Faithfulness - Kendall correlation w.r.t. leave one out
            from scipy.stats import kendalltau

            kendalltau_score = kendalltau(loo_scores, score_explanation)[0]

        else:
            # Faithfulness - Kendall tau distance w.r.t. leave one out
            kendalltau_score = self._kendalltau_distance(loo_scores, score_explanation)

        return kendalltau_score

    def aggregate_score(self, score, total, **aggregation_args):
        return super().aggregate_score(score, total, **aggregation_args)
