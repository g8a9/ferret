from typing import Dict, Text

import numpy as np
import shap
from shap.maskers import Text as TextMasker

from . import BaseExplainer
from .explanation import Explanation
from .utils import parse_explainer_args


class SHAPExplainer(BaseExplainer):
    NAME = "Partition SHAP"

    def __init__(
        self,
        model,
        tokenizer,
        task_name: str = "text-classification",
        silent: bool = True,
        algorithm: str = "partition",
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(model, tokenizer, task_name, **kwargs)
        self.init_args["silent"] = silent
        self.init_args["algorithm"] = algorithm
        self.init_args["seed"] = seed

    def compute_feature_importance(self, text, target=1, **kwargs):
        # sanity checks
        target = self.helper._check_target(target)
        text = self.helper._check_sample(text)

        def func(texts: np.array):
            _, logits = self.helper._forward(texts.tolist())
            return logits.softmax(-1).cpu().numpy()

        masker = TextMasker(self.tokenizer)
        explainer_partition = shap.Explainer(model=func, masker=masker, **self.init_args)
        shap_values = explainer_partition(text, **kwargs)
        attr = shap_values.values[0][:, target]

        output = Explanation(text, self.get_tokens(text), attr, self.NAME, target)
        return output
