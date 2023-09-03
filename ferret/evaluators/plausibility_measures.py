import numpy as np
from sklearn.metrics import auc, precision_recall_curve

from ..explainers.explanation import ExplanationWithRationale
from . import BaseEvaluator, EvaluationMetricFamily
from .evaluation import EvaluationMetricOutput
from .utils_from_soft_to_discrete import (
    get_discrete_explanation_topK,
    parse_evaluator_args,
)


class AUPRC_PlausibilityEvaluation(BaseEvaluator):
    NAME = "AUPRC_soft_plausibility"
    SHORT_NAME = "auprc_plau"
    LOWER_IS_BETTER = False
    MAX_VALUE = 1.0
    MIN_VALUE = 0.0
    METRIC_FAMILY = EvaluationMetricFamily.PLAUSIBILITY

    def _compute_auprc_soft_scoring(self, true_rationale, soft_scores):
        precision, recall, _ = precision_recall_curve(true_rationale, soft_scores)
        auc_score = auc(recall, precision)
        return auc_score

    def compute_evaluation(
        self,
        explanation_with_rationale: ExplanationWithRationale,
        target=1,
        **evaluation_args
    ):
        """Evaluate an explanation on the Area Under the Precision- Recall (AUPRC) Plausibility metric.

        Args:
            explanation (ExplanationWithRationale): the explanation to evaluate
            evaluation_args (dict):  additional evaluation args

        Returns:
            Evaluation : the AUPRC Plausibility score of the explanation
        """

        # Plausibility - Area Under the Precision- Recall curve (AUPRC) - ERASER
        if isinstance(explanation_with_rationale, ExplanationWithRationale) == False:
            return None
        remove_first_last, only_pos, _, _ = parse_evaluator_args(evaluation_args)

        score_explanation = explanation_with_rationale.scores
        human_rationale = explanation_with_rationale.rationale

        if remove_first_last == True:
            human_rationale = human_rationale[1:-1]
            if self.tokenizer.cls_token == explanation_with_rationale.tokens[0]:
                score_explanation = score_explanation[1:-1]

        # TODO.
        if only_pos:
            # Only positive terms of explanations.
            # https://github.com/hate-alert/HateXplain/blob/daa7955afbe796b00e79817f16621469a38820e0/testing_with_lime.py#L276
            score_explanation = [v if v > 0 else 0 for v in score_explanation]

        auprc_soft_plausibility = self._compute_auprc_soft_scoring(
            human_rationale, score_explanation
        )
        evaluation_output = EvaluationMetricOutput(self, auprc_soft_plausibility)
        return evaluation_output


class Tokenf1_PlausibilityEvaluation(BaseEvaluator):
    NAME = "token_f1_hard_plausibility"
    SHORT_NAME = "token_f1_plau"
    METRIC_FAMILY = EvaluationMetricFamily.PLAUSIBILITY
    LOWER_IS_BETTER = False
    MIN_VALUE = 0.0
    MAX_VALUE = 1.0

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

    def compute_evaluation(
        self,
        explanation_with_rationale: ExplanationWithRationale,
        target=1,
        **evaluation_args
    ):

        """Evaluate an explanation on the Token-f1 score Plausibility metric.

        Args:
            explanation (ExplanationWithRationale): the explanation to evaluate
            evaluation_args (dict):  additional evaluation args

        Returns:
            Evaluation : the Token-f1 Plausibility score of the explanation
        """

        if isinstance(explanation_with_rationale, ExplanationWithRationale) == False:
            return None

        # Token fpr - hard rationale predictions. token-level F1 scores
        remove_first_last, only_pos, _, top_k_hard_rationale = parse_evaluator_args(
            evaluation_args
        )
        accumulate_result = evaluation_args.get("accumulate_result", False)

        score_explanation = explanation_with_rationale.scores
        human_rationale = explanation_with_rationale.rationale

        if remove_first_last == True:
            human_rationale = human_rationale[1:-1]
            if self.tokenizer.cls_token == explanation_with_rationale.tokens[0]:
                score_explanation = score_explanation[1:-1]

        topk_score_explanations = get_discrete_explanation_topK(
            score_explanation, top_k_hard_rationale, only_pos=only_pos
        )

        if topk_score_explanations is None:
            # Return default scores
            if accumulate_result:
                return EvaluationMetricOutput(self, [0, 0, 0, 0, 0, 0])
            else:
                return EvaluationMetricOutput(self, 0)

        tp, pos, pred_pos = self._instance_tp_pos_pred_pos(
            human_rationale, topk_score_explanations
        )

        (
            instance_prec,
            instance_rec,
            instance_f1_micro,
        ) = self._precision_recall_fmeasure(tp, pos, pred_pos)

        if accumulate_result:

            output_score = np.array(
                [tp, pos, pred_pos, instance_prec, instance_rec, instance_f1_micro]
            )

            evaluation_output = EvaluationMetricOutput(self.SHORT_NAME, output_score)
        else:
            evaluation_output = EvaluationMetricOutput(
                self.SHORT_NAME, instance_f1_micro
            )

        return evaluation_output

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
    METRIC_FAMILY = EvaluationMetricFamily.PLAUSIBILITY
    LOWER_IS_BETTER = False
    MIN_VALUE = 0.0
    MAX_VALUE = 1.0

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

    def compute_evaluation(
        self,
        explanation_with_rationale: ExplanationWithRationale,
        target=1,
        **evaluation_args
    ):

        """Evaluate an explanation on the Intersection Over Union (IOU) Plausibility metric.

        Args:
            explanation (ExplanationWithRationale): the explanation to evaluate
            evaluation_args (dict):  additional evaluation args

        Returns:
            Evaluation : the IOU Plausibility score of the explanation
        """

        """From ERASER
        'We define IOU on a token level:  for two spans,
        it is the size of the overlap of the tokens they cover divided by the size of their union.''

        Same process as in _token_f1_hard_rationales
        rationale: one hot encoding of the rationale
        soft_score_explanation: soft scores, len = #tokens, floats
        """

        if isinstance(explanation_with_rationale, ExplanationWithRationale) == False:
            return None

        remove_first_last, only_pos, _, top_k_hard_rationale = parse_evaluator_args(
            evaluation_args
        )

        score_explanation = explanation_with_rationale.scores
        human_rationale = explanation_with_rationale.rationale

        if remove_first_last == True:
            human_rationale = human_rationale[1:-1]
            if self.tokenizer.cls_token == explanation_with_rationale.tokens[0]:
                score_explanation = score_explanation[1:-1]

        topk_score_explanations = get_discrete_explanation_topK(
            score_explanation, top_k_hard_rationale, only_pos=only_pos
        )
        if topk_score_explanations is None:
            # Return default scores
            return EvaluationMetricOutput(self, 0)

        token_iou = self._token_iou(human_rationale, topk_score_explanations)

        evaluation_output = EvaluationMetricOutput(self, token_iou)
        return evaluation_output
