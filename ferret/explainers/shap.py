from . import BaseExplainer
from .utils import parse_explainer_args
from .explanation import Explanation
from shap import Explainer as ShapExplainer
from typing import Dict
from transformers import TextClassificationPipeline


class TextClassificationPipelineWithTruncation(TextClassificationPipeline):
    def preprocess(self, inputs, **tokenizer_kwargs) -> Dict:
        return_tensors = self.framework
        tokenizer_kwargs["truncation"] = True
        if isinstance(inputs, dict):
            return self.tokenizer(
                **inputs, return_tensors=return_tensors, **tokenizer_kwargs
            )
        elif (
            isinstance(inputs, list)
            and len(inputs) == 1
            and isinstance(inputs[0], list)
            and len(inputs[0]) == 2
        ):
            # It used to be valid to use a list of list of list for text pairs, keeping this path for BC
            return self.tokenizer(
                text=inputs[0][0],
                text_pair=inputs[0][1],
                return_tensors=return_tensors,
                **tokenizer_kwargs
            )
        elif isinstance(inputs, list):
            # This is likely an invalid usage of the pipeline attempting to pass text pairs.
            raise ValueError(
                "The pipeline received invalid inputs, if you are trying to send text pairs, you can try to send a"
                ' dictionnary `{"text": "My text", "text_pair": "My pair"}` in order to send a text pair.'
            )
        return self.tokenizer(inputs, return_tensors=return_tensors, **tokenizer_kwargs)


class SHAPExplainer(BaseExplainer):
    NAME = "Partition SHAP"

    def compute_feature_importance(self, text, target=1, **explainer_args):
        init_args, call_args = parse_explainer_args(explainer_args)

        # SHAP silent mode
        init_args["silent"] = init_args.get("silent", True)

        pipe = TextClassificationPipelineWithTruncation(
            model=self.helper.model,
            tokenizer=self.helper.tokenizer,
            return_all_scores=True,
        )
        explainer_partition = ShapExplainer(pipe, **init_args)
        shap_values = explainer_partition([text], **call_args)
        attr = shap_values.values[0][:, target]

        output = Explanation(
            text, self.get_tokens(text), attr, self.NAME, target
        )
        return output
