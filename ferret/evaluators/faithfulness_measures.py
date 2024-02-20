import copy
import pdb
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
from scipy.stats import kendalltau

from ..explainers.explanation import Explanation, ExplanationWithRationale
from . import BaseEvaluator, EvaluationMetricFamily
from .evaluation import EvaluationMetricOutput
from .perturbation import PertubationHelper
from .utils_from_soft_to_discrete import (
    _check_and_define_get_id_discrete_rationale_function,
    parse_evaluator_args,
)


def _compute_aopc(scores):
    from statistics import mean

    return mean(scores)


class AOPC_Comprehensiveness_Evaluation(BaseEvaluator):
    NAME = "aopc_comprehensiveness"
    SHORT_NAME = "aopc_compr"

    LOWER_IS_BETTER = False
    MIN_VALUE = -1.0
    MAX_VALUE = 1.0
    BEST_VALUE = 1.0
    METRIC_FAMILY = EvaluationMetricFamily.FAITHFULNESS

    def compute_evaluation(
        self, explanation: Explanation, **evaluation_args
    ) -> EvaluationMetricOutput:
        """Evaluate an explanation on the AOPC Comprehensiveness metric.

        Args:
            explanation (Explanation): the explanation to evaluate
            target (int): class label for which the explanation is evaluated
            evaluation_args (dict):  additional evaluation args.
                We currently support multiple approaches to define the hard rationale from
                soft score rationales, based on:
                - th : token greater than a threshold
                - perc : more than x% of the tokens
                - k: top k values

        Returns:
            Evaluation : the AOPC Comprehensiveness score of the explanation
        """

        remove_first_last, only_pos, removal_args, _ = parse_evaluator_args(
            evaluation_args
        )

        text = explanation.text
        target_pos_idx = explanation.target_pos_idx
        target_token_pos_idx = explanation.target_token_pos_idx
        score_explanation = explanation.scores
        helper_type = explanation.helper_type

        if (
            removal_args["remove_tokens"] == True
            and helper_type == "token-classification"
        ):
            removal_args["remove_tokens"] = False
            warnings.warn(
                "NER does not support token removal. 'remove_tokens' set to False"
            )

        # TODO - use tokens
        # Get prediction probability of the input sencence for the target
        _, logits = self.helper._forward(text, output_hidden_states=False)
        logits = self.helper._postprocess_logits(
            logits, target_token_pos_idx=target_token_pos_idx
        )

        baseline = logits.softmax(-1)[0, target_pos_idx].item()

        # TODO This part needs serious revision if metrics have to be general across modalities
        # Tokenized input
        item = self.helper._tokenize(text)
        input_len = item["attention_mask"].sum().item()
        input_ids = item["input_ids"][0][:input_len].tolist()

        # If remove_first_last, first and last token id (CLS and ) are removed
        if remove_first_last == True:
            input_ids = input_ids[1:-1]
            if self.tokenizer.cls_token == explanation.tokens[0]:
                score_explanation = score_explanation[1:-1]

        discrete_expl_ths = list()
        id_tops = list()
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
            if (
                id_top is not None
                and last_id_top is not None
                and set(id_top) == last_id_top
            ):
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

            discrete_expl_th = self.tokenizer.decode(
                discrete_expl_th_token_ids, skip_special_tokens=False
            )

            discrete_expl_ths.append(discrete_expl_th)

        if discrete_expl_ths == list():
            return EvaluationMetricOutput(self, 0)

        # Prediction probability for the target and post process logits
        _, logits = self.helper._forward(discrete_expl_ths, output_hidden_states=False)
        logits = self.helper._postprocess_logits(
            logits, target_token_pos_idx=target_token_pos_idx
        )

        probs_removing = logits.softmax(-1)[:, target_pos_idx].cpu().numpy()

        # compute probability difference
        removal_importance = baseline - probs_removing
        #  compute AOPC comprehensiveness
        aopc_comprehesiveness = _compute_aopc(removal_importance)

        evaluation_output = EvaluationMetricOutput(self, aopc_comprehesiveness)
        return evaluation_output

    # def aggregate_score(self, score, total, **aggregation_args):
    #     return super().aggregate_score(score, total, **aggregation_args)


class AOPC_Sufficiency_Evaluation(BaseEvaluator):
    NAME = "aopc_sufficiency"
    SHORT_NAME = "aopc_suff"
    LOWER_IS_BETTER = True
    MIN_VALUE = -1.0
    MAX_VALUE = 1.0
    BEST_VALUE = 0.0
    METRIC_FAMILY = EvaluationMetricFamily.FAITHFULNESS

    def compute_evaluation(
        self, explanation: Explanation, **evaluation_args
    ) -> EvaluationMetricOutput:
        """Evaluate an explanation on the AOPC Sufficiency metric.

        Args:
            explanation (Explanation): the explanation to evaluate
            target (int): class label for which the explanation is evaluated
            evaluation_args (dict):  additional evaluation args

        Returns:
            Evaluation : the AOPC Sufficiency score of the explanation
        """

        remove_first_last, only_pos, removal_args, _ = parse_evaluator_args(
            evaluation_args
        )

        text = explanation.text
        score_explanation = explanation.scores
        target_pos_idx = explanation.target_pos_idx
        target_token_pos_idx = explanation.target_token_pos_idx
        helper_type = explanation.helper_type

        if (
            removal_args["remove_tokens"] == True
            and helper_type == "token-classification"
        ):
            removal_args["remove_tokens"] = False
            warnings.warn(
                "NER does not support token removal. 'remove_tokens' set to False"
            )

        # TO DO - use tokens
        # Get prediction probability of the input sencence for the target
        _, logits = self.helper._forward(text, output_hidden_states=False)
        logits = self.helper._postprocess_logits(
            logits, target_token_pos_idx=target_token_pos_idx
        )
        baseline = logits.softmax(-1)[0, target_pos_idx].item()

        # Tokenized sentence
        item = self.helper._tokenize(text)
        # Get token ids of the sentence
        input_len = item["attention_mask"].sum().item()
        input_ids = item["input_ids"][0][:input_len].tolist()

        # If remove_first_last, first and last token id (CLS and ) are removed
        if remove_first_last == True:
            input_ids = input_ids[1:-1]
            if self.tokenizer.cls_token == explanation.tokens[0]:
                score_explanation = score_explanation[1:-1]

        discrete_expl_ths = list()
        id_tops = list()

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
            if (
                id_top is not None
                and last_id_top is not None
                and set(id_top) == last_id_top
            ):
                id_top = None

            id_tops.append(id_top)

            if id_top is None:
                continue

            last_id_top = set(id_top)

            # Sufficiency
            # The only difference between comprehesivenss and sufficiency is the computation of the removal.
            # For the sufficiency: we keep only the terms in the discrete rationale.

            sample = np.array(copy.copy(input_ids))

            # We take the tokens in the original order
            id_top = np.sort(id_top)

            if removal_args["remove_tokens"]:
                discrete_expl_th_token_ids = sample[id_top]
            else:
                mask_not_top = np.ones(sample.size, dtype=bool)
                mask_not_top[id_top] = False
                sample[mask_not_top] = self.tokenizer.mask_token_id
                discrete_expl_th_token_ids = sample
            ##############################################

            discrete_expl_th = self.tokenizer.decode(
                discrete_expl_th_token_ids, skip_special_tokens=False
            )
            discrete_expl_ths.append(discrete_expl_th)

        if discrete_expl_ths == []:
            return EvaluationMetricOutput(self, 1)

        # Prediction probability for the target
        _, logits = self.helper._forward(discrete_expl_ths, output_hidden_states=False)
        logits = self.helper._postprocess_logits(
            logits, target_token_pos_idx=target_token_pos_idx
        )
        probs_removing = logits.softmax(-1)[:, target_pos_idx].cpu().numpy()

        # Compute probability difference
        removal_importance = baseline - probs_removing

        aopc_sufficiency = _compute_aopc(removal_importance)

        evaluation_output = EvaluationMetricOutput(self, aopc_sufficiency)
        return evaluation_output

    # def aggregate_score(self, score, total, **aggregation_args):
    #     return super().aggregate_score(score, total, **aggregation_args)


class TauLOO_Evaluation(BaseEvaluator):
    NAME = "tau_leave-one-out_correlation"
    SHORT_NAME = "taucorr_loo"
    METRIC_FAMILY = EvaluationMetricFamily.FAITHFULNESS
    LOWER_IS_BETTER = False
    MAX_VALUE = 1.0
    MIN_VALUE = -1.0
    BEST_VALUE = 1.0

    def compute_evaluation(
        self, explanation: Explanation, **evaluation_args
    ) -> EvaluationMetricOutput:
        """Evaluate an explanation on the tau-LOO metric,
        i.e., the Kendall tau correlation between the explanation scores and leave one out (LOO) scores,
        computed by leaving one feature out and computing the change in the prediciton probability

        Args:
            explanation (Explanation): the explanation to evaluate
            target (int): class label for which the explanation is evaluated
            evaluation_args (dict):  additional evaluation args

        Returns:
            Evaluation : the tau-LOO score of the explanation
        """

        text = explanation.text
        score_explanation = explanation.scores
        target_pos_idx = explanation.target_pos_idx
        target_token_pos_idx = explanation.target_token_pos_idx
        helper_type = explanation.helper_type

        remove_first_last = evaluation_args.get("remove_first_last", True)

        if remove_first_last:
            if self.tokenizer.cls_token == explanation.tokens[0]:
                score_explanation = score_explanation[1:-1]

        _, logits = self.helper._forward(text, output_hidden_states=False)
        logits = self.helper._postprocess_logits(
            logits, target_token_pos_idx=target_token_pos_idx
        )

        baseline = logits.softmax(-1)[0, target_pos_idx].item()

        item = self.helper._tokenize(text)
        input_len = item["attention_mask"].sum().item()
        input_ids = item["input_ids"][0][:input_len].tolist()
        if remove_first_last == True:
            input_ids = input_ids[1:-1]

        # TODO: for a very long input these end up being many samples we need to process. Think about showing a progress bar here
        perturbation_helper = PertubationHelper(self.helper.tokenizer)
        samples_ids = perturbation_helper.edit_one_token(
            input_ids,
            strategy="remove" if helper_type != "token-classification" else "mask",
        )
        samples = self.helper.tokenizer.batch_decode(
            samples_ids, skip_special_tokens=False
        )

        _, logits = self.helper._forward(samples, output_hidden_states=False)
        logits = self.helper._postprocess_logits(
            logits, target_token_pos_idx=target_token_pos_idx
        )
        leave_one_out_removal = logits.softmax(-1)[:, target_pos_idx].cpu()

        occlusion_importance = leave_one_out_removal - baseline
        loo_scores = -1 * occlusion_importance.numpy()
        kendalltau_score = kendalltau(loo_scores, score_explanation)[0]

        evaluation_output = EvaluationMetricOutput(self, kendalltau_score)
        return evaluation_output

    # def aggregate_score(self, score, total, **aggregation_args):
    #     return super().aggregate_score(score, total, **aggregation_args)
