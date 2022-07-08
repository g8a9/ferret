from . import BaseExplainer
from .utils import parse_explainer_args

from shap import Explainer as ShapExplainer
from transformers import pipeline


class SHAPExplainer(BaseExplainer):
    NAME = "SHAP"

    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer)

    def compute_feature_importance(self, text, target=1, **explainer_args):
        init_args, call_args = parse_explainer_args(explainer_args)

        pipe = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            return_all_scores=True,
        )
        explainer_partition = ShapExplainer(pipe, **init_args)
        shap_values = explainer_partition(text, **call_args)
        return shap_values.values[0][:, target]
