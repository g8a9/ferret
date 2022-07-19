from pickle import FALSE
from . import BaseEvaluator
from ..modelw import Model
import numpy as np
from typing import List
from .utils_from_soft_to_discrete import (
    get_discrete_explanation_topK,
)


class AUPRC_PlausibilityEvaluation(BaseEvaluator):
    NAME = "AUPRC_soft_plausibility"
    SHORT_NAME = "auprc_plau"
    # Higher is better
    BEST_SORTING_ASCENDING = False
    TYPE_METRIC = "plausibility"

    def __init__(self, model: Model, tokenizer):
        super().__init__(model, tokenizer)

    def _compute_auprc_soft_scoring(self, true_rationale, soft_scores):
        from sklearn.metrics import auc, precision_recall_curve

        precision, recall, _ = precision_recall_curve(true_rationale, soft_scores)
        auc_score = auc(recall, precision)
        return auc_score

    def evaluate_explanations(
        self,
        texts: List[str],
        score_explanations: List[List[float]],
        human_rationales,
        targets=None,
        **evaluation_args
    ):
        scores = []
        for text, score_explanation, human_rationale in zip(
            texts, score_explanations, human_rationales
        ):
            scores.append(
                self.evaluate_explanation(
                    text,
                    score_explanation,
                    human_rationale,
                    # target=target,
                    **evaluation_args
                )
            )
        # https://github.com/jayded/eraserbenchmark/blob/36467f1662812cbd4fbdd66879946cd7338e08ec/rationale_benchmark/metrics.py#L222        # Average of aucs
        average_score, std = np.average(scores), np.std(scores)
        return average_score, std

    def evaluate_explanation(
        self, text, score_explanation, human_rationale, target=1, **evaluation_args
    ):
        # Plausibility - Area Under the Precision- Recall curve (AUPRC) - ERASER

        only_pos = evaluation_args.get("only_pos", False)

        # TODO.
        if only_pos:
            # Only positive terms of explanations.
            # https://github.com/hate-alert/HateXplain/blob/daa7955afbe796b00e79817f16621469a38820e0/testing_with_lime.py#L276
            score_explanation = [v if v > 0 else 0 for v in score_explanation]

        auprc_soft_plausibility = self._compute_auprc_soft_scoring(
            human_rationale, score_explanation
        )
        return auprc_soft_plausibility

    def aggregate_score(self, score, total, **aggregation_args):
        return super().aggregate_score(score, total, **aggregation_args)


class Tokenf1_PlausibilityEvaluation(BaseEvaluator):
    NAME = "token_f1_hard_plausibility"
    SHORT_NAME = "token_f1_plau"
    # Higher is better
    BEST_SORTING_ASCENDING = False
    TYPE_METRIC = "plausibility"
    INIT_VALUE = np.zeros(6)

    def __init__(self, model: Model, tokenizer):
        super().__init__(model, tokenizer)

    def _instance_tp_pos_pred_pos(self, true_expl, pred_expl):

        true_expl = np.array(true_expl)
        pred_expl = np.array(pred_expl)

        assert true_expl.shape[0] == pred_expl.shape[0]

        tp = (true_expl & pred_expl).sum()
        pos = (true_expl).sum()
        pred_pos = (pred_expl).sum()

        """
        Alternative, in the case the rationales are representate by the positional id
        e.g., "i hate you" --> [1,2]
        
        true_expl = set(true_expl)
        pred_expl = set(pred_expl)

        tp =  len(true_expl & pred_expl)
        pos = len(true_expl)
        pred_pos = len(pred_expl)
        """
        return tp, pos, pred_pos

    def _precision_recall_fmeasure(self, tp, positive, pred_positive):
        precision = tp / pred_positive
        recall = tp / positive
        fmeasure = self._f1(precision, recall)
        return precision, recall, fmeasure

    def _f1(self, _p, _r):
        if _p == 0 or _r == 0:
            return 0
        return 2 * _p * _r / (_p + _r)

    def _score_hard_rationale_predictions_dataset(self, list_true_expl, list_pred_expl):

        """Computes instance micro/macro averaged F1s
        ERASER: https://github.com/jayded/eraserbenchmark/blob/36467f1662812cbd4fbdd66879946cd7338e08ec/rationale_benchmark/metrics.py#L168

        """

        """ Each explanations is provided as one hot encoding --> True if the word is in the explanation, False otherwise
        I hate you --> --> [0, 1, 1]
        One for each instance.
        """
        tot_tp, tot_pos, tot_pred_pos = 0, 0, 0
        macro_prec_sum, macro_rec_sum, macro_f1_sum = 0, 0, 0

        for true_expl, pred_expl in zip(list_true_expl, list_pred_expl):
            tp, pos, pred_pos = self._instance_tp_pos_pred_pos(true_expl, pred_expl)

            instance_prec, instance_rec, instance_f1 = self._precision_recall_fmeasure(
                tp, pos, pred_pos
            )

            # Update for macro computation
            macro_prec_sum += instance_prec
            macro_rec_sum += instance_rec
            macro_f1_sum += instance_f1

            # Update for micro computation
            tot_tp += tp
            tot_pos += pos
            tot_pred_pos += pred_pos

        # Macro computation

        n_explanations = len(list_true_expl)
        macro = {
            "p": macro_prec_sum / n_explanations,
            "r": macro_rec_sum / n_explanations,
            "f1": macro_f1_sum / n_explanations,
        }

        # Micro computation

        micro_prec, micro_rec, micro_f1 = self._precision_recall_fmeasure(
            tot_tp, tot_pos, tot_pred_pos
        )
        micro = {"p": micro_prec, "r": micro_rec, "f1": micro_f1}

        return {"micro": micro, "macro": macro}

    def _score_hard_rationale_predictions_accumulate(self, true_expl, pred_expl):

        """Computes instance micro/macro averaged F1s
        ERASER: https://github.com/jayded/eraserbenchmark/blob/36467f1662812cbd4fbdd66879946cd7338e08ec/rationale_benchmark/metrics.py#L168

        """

        """ Each explanations is provided as one hot encoding --> True if the word is in the explanation, False otherwise
        I hate you --> --> [0, 1, 1]
        One for each instance.
        """

        # For macro computation
        tp, pos, pred_pos = self._instance_tp_pos_pred_pos(true_expl, pred_expl)

        # For micro computation
        instance_prec, instance_rec, instance_f1 = self._precision_recall_fmeasure(
            tp, pos, pred_pos
        )
        return instance_prec, instance_rec, instance_f1, tp, pos, pred_pos

    def evaluate_explanations(
        self,
        texts: List[str],
        score_explanations: List[List[float]],
        human_rationales,
        targets=None,
        **evaluation_args
    ):

        only_pos = evaluation_args.get("only_pos", False)
        top_k_hard_rationale = evaluation_args.get("top_k_rationale", 5)

        discrete_explanations = []
        for score_explanation in score_explanations:

            topk_score_explanations = get_discrete_explanation_topK(
                score_explanation, top_k_hard_rationale, only_pos=only_pos
            )

            discrete_explanations.append(topk_score_explanations)

        scores = self._score_hard_rationale_predictions_dataset(
            human_rationales, discrete_explanations
        )

        # Micro and macro f1, precision, recall
        return scores

    def evaluate_explanation(
        self, text, score_explanation, human_rationale, target=1, **evaluation_args
    ):
        # Token fpr - hard rationale predictions. token-level F1 scores
        only_pos = evaluation_args.get("only_pos", False)
        top_k_hard_rationale = evaluation_args.get("top_k_rationale", 5)

        topk_score_explanations = get_discrete_explanation_topK(
            score_explanation, top_k_hard_rationale, only_pos=only_pos
        )

        tp, pos, pred_pos = self._instance_tp_pos_pred_pos(
            human_rationale, topk_score_explanations
        )

        (
            instance_prec,
            instance_rec,
            instance_f1_micro,
        ) = self._precision_recall_fmeasure(tp, pos, pred_pos)

        accumulate_result = evaluation_args.get("accumulate_result", False)

        if accumulate_result:
            return tp, pos, pred_pos, instance_prec, instance_rec, instance_f1_micro
        else:
            return instance_f1_micro

    def aggregate_score(self, score, total, **aggregation_args):
        average = aggregation_args.get("average", "macro")
        (
            total_tp,
            total_pos,
            total_pred_pos,
            macro_prec_sum,
            macro_rec_sum,
            macro_f1_sum,
        ) = tuple(score)

        # Macro computation
        macro = {
            "p": macro_prec_sum / total,
            "r": macro_rec_sum / total,
            "f1": macro_f1_sum / total,
        }

        # Micro computation

        micro_prec, micro_rec, micro_f1 = self._precision_recall_fmeasure(
            total_tp, total_pos, total_pred_pos
        )
        micro = {"p": micro_prec, "r": micro_rec, "f1": micro_f1}

        if average == "macro":
            return macro["f1"]
        elif average == "micro":
            return micro["f1"]
        else:
            raise ValueError()


class TokenIOU_PlausibilityEvaluation(BaseEvaluator):
    NAME = "token_IOU_hard_plausibility"
    SHORT_NAME = "token_iou_plau"
    # Higher is better
    BEST_SORTING_ASCENDING = False
    TYPE_METRIC = "plausibility"

    def __init__(self, model: Model, tokenizer):
        super().__init__(model, tokenizer)

    def _token_iou(self, true_expl, pred_expl):
        """From ERASER
        We define IOU on a token level:  for two spans,
            it is the size of the overlap of the tokens they cover divided by the size of their union.
        """

        if type(true_expl) is list:
            true_expl = np.array(true_expl)
        if type(pred_expl) is list:
            pred_expl = np.array(pred_expl)

        assert true_expl.shape[0] == pred_expl.shape[0]

        num = (true_expl & pred_expl).sum()
        denom = (true_expl | pred_expl).sum()

        iou = 0 if denom == 0 else num / denom
        return iou

    def evaluate_explanation(
        self, text, score_explanation, human_rationale, target=1, **evaluation_args
    ):

        """From ERASER
        'We define IOU on a token level:  for two spans,
        it is the size of the overlap of the tokens they cover divided by the size of their union.''

        Same process as in _token_f1_hard_rationales
        rationale: one hot encoding of the rationale
        soft_score_explanation: soft scores, len = #tokens, floats
        """

        only_pos = evaluation_args.get("only_pos", False)
        top_k_hard_rationale = evaluation_args.get("top_k_rationale", 5)

        topk_score_explanations = get_discrete_explanation_topK(
            score_explanation, top_k_hard_rationale, only_pos=only_pos
        )

        return self._token_iou(human_rationale, topk_score_explanations)

    # TODO
    def evaluate_explanations(
        self,
        texts: List[str],
        score_explanations: List[List[float]],
        human_rationales,
        targets=None,
        **evaluation_args
    ):
        scores = []
        for text, score_explanation, human_rationale in zip(
            texts, score_explanations, human_rationales
        ):
            scores.append(
                self.evaluate_explanation(
                    text,
                    score_explanation,
                    human_rationale,
                    # target=target,
                    **evaluation_args
                )
            )
        # https://github.com/jayded/eraserbenchmark/blob/36467f1662812cbd4fbdd66879946cd7338e08ec/rationale_benchmark/metrics.py#L222        # Average of aucs
        average_score, std = np.average(scores), np.std(scores)
        return average_score, std

    def aggregate_score(self, score, total, **aggregation_args):
        return super().aggregate_score(score, total, **aggregation_args)
