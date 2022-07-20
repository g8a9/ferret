from . import BaseExplainer
from .utils import parse_explainer_args
from .explanation import Explanation
from shap import Explainer as ShapExplainer
from transformers import pipeline


class SHAPExplainer(BaseExplainer):
    NAME = "Partition SHAP"

    def compute_feature_importance(self, text, target=1, **explainer_args):
        init_args, call_args = parse_explainer_args(explainer_args)

        pipe = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            return_all_scores=True,
        )
        explainer_partition = ShapExplainer(pipe, **init_args)
        shap_values = explainer_partition([text], **call_args)
        attr = shap_values.values[0][:, target]

        output = Explanation(text, self.get_tokens(text), attr, self.NAME, target)
        return output
