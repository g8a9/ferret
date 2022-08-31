from . import BaseExplainer
from .utils import parse_explainer_args
from .explanation import Explanation
import shap
from shap.maskers import Text as TextMasker
from typing import Dict, Text


class SHAPExplainer(BaseExplainer):
    NAME = "Partition SHAP"

    def compute_feature_importance(self, text, target=1, **explainer_args):
        init_args, call_args = parse_explainer_args(explainer_args)

        # SHAP silent mode
        init_args["silent"] = init_args.get("silent", True)
        # Default to 'Partition' algorithm
        init_args["algorithm"] = init_args.get("algorithm", "partition")
        #  seed for reproducibility
        init_args["seed"] = init_args.get("seed", 42)

        def func(texts):
            _, logits = self.helper._forward(texts)
            return logits.softmax(-1).cpu().numpy()

        masker = TextMasker(self.tokenizer)
        explainer_partition = shap.Explainer(model=func, masker=masker, **init_args)
        shap_values = explainer_partition([text], **call_args)
        attr = shap_values.values[0][:, target]

        output = Explanation(text, self.get_tokens(text), attr, self.NAME, target)
        return output
