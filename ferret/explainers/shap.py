import pdb
from typing import Dict, Optional, Text, Union

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
        model_helper: Optional[str] = None,
        silent: bool = True,
        algorithm: str = "partition",
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(model, tokenizer, model_helper, **kwargs)
        self.init_args["silent"] = silent
        self.init_args["algorithm"] = algorithm
        self.init_args["seed"] = seed

    def compute_feature_importance(
        self,
        text,
        target: Union[int, Text] = 1,
        target_token: Optional[Union[int, Text]] = None,
        **kwargs,
    ):
        # sanity checks
        target_pos_idx = self.helper._check_target(target)
        target_token_pos_idx = self.helper._check_target_token(text, target_token)
        text = self.helper._check_sample(text)

        kwargs.pop('target_option', None)

        def func(texts: np.array):
            _, logits = self.helper._forward(texts.tolist())
            logits = self.helper._postprocess_logits(
                logits, target_token_pos_idx=target_token_pos_idx
            )
            return logits.softmax(-1).cpu().numpy()

        masker = TextMasker(self.tokenizer)
        explainer_partition = shap.Explainer(model=func, masker=masker, **self.init_args)
        full_text = text
        if isinstance(text, list):
            if isinstance(text[0], tuple):
                full_text = list(text[0])
        shap_values = explainer_partition(full_text, **kwargs)
        attr = shap_values.values[0][:, target_pos_idx]

        item = self._tokenize(full_text, return_special_tokens_mask=True)
        token_ids = item['input_ids'][0].tolist()
        token_scores = np.zeros_like(token_ids, dtype=float)
        for i, (shap_value, is_special_token) in enumerate(zip(attr, item['special_tokens_mask'][0])):
            if not is_special_token:
                token_scores[i] = shap_value

        output = Explanation(
            text=text,
            tokens=self.get_tokens(text),
            scores=token_scores,
            explainer=self.NAME,
            helper_type=self.helper.HELPER_TYPE,
            target_pos_idx=target_pos_idx,
            target_token_pos_idx=target_token_pos_idx,
            target=self.helper.model.config.id2label[target_pos_idx],
            target_token=self.helper.tokenizer.decode(
                item["input_ids"][0, target_token_pos_idx].item()
            )
            if self.helper.HELPER_TYPE == "token-classification"
            else None,
        )
        return output
