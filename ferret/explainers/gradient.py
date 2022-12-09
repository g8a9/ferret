from typing import Tuple, Union

import torch
from captum.attr import InputXGradient, IntegratedGradients, Saliency
from cv2 import multiply

from . import BaseExplainer
from .explanation import Explanation
from .utils import parse_explainer_args


class GradientExplainer(BaseExplainer):
    NAME = "Gradient"

    def __init__(
        self,
        model,
        tokenizer,
        task_name: str = "text-classification",
        multiply_by_inputs: bool = True,
        **kwargs,
    ):
        super().__init__(model, tokenizer, task_name, **kwargs)

        self.multiply_by_inputs = multiply_by_inputs
        if self.multiply_by_inputs:
            self.NAME += " (x Input)"

    def compute_feature_importance(
        self,
        text: Union[str, Tuple[str, str]],
        target: Union[int, str] = 1,
        **kwargs,
    ):
        # sanity checks
        target = self.helper.check_format_target(target)
        text = self.helper.check_format_input(text)

        item = self._tokenize(text)
        item = {k: v.to(self.device) for k, v in item.items()}
        input_len = item["attention_mask"].sum().item()

        def func(input_embeds):
            outputs = self.helper.model(
                inputs_embeds=input_embeds, attention_mask=item["attention_mask"]
            )
            return outputs.logits

        dl = (
            InputXGradient(func, **self.init_args)
            if self.multiply_by_inputs
            else Saliency(func, **self.init_args)
        )

        inputs = self.get_input_embeds(text)
        attr = dl.attribute(inputs, target=target, **kwargs)
        attr = attr[0, :input_len, :].detach().cpu()

        # pool over hidden size
        attr = attr.sum(-1).numpy()

        output = Explanation(text, self.get_tokens(text), attr, self.NAME, target)
        return output


class IntegratedGradientExplainer(BaseExplainer):
    NAME = "Integrated Gradient"

    def __init__(
        self,
        model,
        tokenizer,
        task_name: str = "text-classification",
        multiply_by_inputs: bool = True,
        **kwargs,
    ):
        super().__init__(model, tokenizer, task_name, **kwargs)

        self.multiply_by_inputs = multiply_by_inputs
        if self.multiply_by_inputs:
            self.NAME += " (x Input)"

    def _generate_baselines(self, input_len):
        ids = (
            [self.helper.tokenizer.cls_token_id]
            + [self.helper.tokenizer.pad_token_id] * (input_len - 2)
            + [self.helper.tokenizer.sep_token_id]
        )
        embeddings = self.helper._get_input_embeds_from_ids(
            torch.tensor(ids, device=self.device)
        )
        return embeddings.unsqueeze(0)

    def compute_feature_importance(
        self,
        text: Union[str, Tuple[str, str]],
        target: Union[int, str] = 1,
        show_progress: bool = False,
        **kwargs,
    ):
        # sanity checks
        target = self.helper.check_format_target(target)
        text = self.helper.check_format_input(text)

        # init_args, call_args = parse_explainer_args(explainer_args)
        item = self._tokenize(text)
        input_len = item["attention_mask"].sum().item()

        def func(input_embeds):
            attention_mask = torch.ones(
                *input_embeds.shape[:2], dtype=torch.uint8, device=self.device
            )
            _, logits = self.helper._forward_with_input_embeds(
                input_embeds, attention_mask, show_progress=show_progress
            )
            return logits

        dl = IntegratedGradients(
            func, multiply_by_inputs=self.multiply_by_inputs, **self.init_args
        )
        inputs = self.get_input_embeds(text)
        baselines = self._generate_baselines(input_len)

        attr = dl.attribute(inputs, baselines=baselines, target=target, **kwargs)
        attr = attr[0, :input_len, :].detach().cpu()

        # pool over hidden size
        attr = attr.sum(-1).numpy()

        # norm_attr = self._normalize_input_attributions(attr.detach())
        output = Explanation(text, self.get_tokens(text), attr, self.NAME, target)
        return output
