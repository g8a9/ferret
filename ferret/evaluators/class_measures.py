from ferret.explainers.explanation import Explanation, ExplanationWithRationale
from .faithfulness_measures import (
    AOPC_Comprehensiveness_Evaluation,
)
from .evaluation import Evaluation
import numpy as np
from typing import List, Union


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

    def compute_evaluation(
        self,
        class_explanation: List[Union[Explanation, ExplanationWithRationale]],
        **evaluation_args
    ):

        """
        Each element of the list is the explanation for a target class
        """

        evaluation_args["only_pos"] = True

        aopc_values = []
        for target, explanation in enumerate(class_explanation):
            aopc_values.append(
                self.aopc_compr_eval.compute_evaluation(
                    explanation, target, **evaluation_args
                ).score
            )
        aopc_class_score = np.mean(aopc_values)
        evaluation_output = Evaluation(self.SHORT_NAME, aopc_class_score)
        return evaluation_output

    def aggregate_score(self, score, total, **aggregation_args):
        return score / total
