"""Explainers API"""

import warnings
from abc import ABC, abstractmethod
from typing import Optional, Union

from ..modeling import create_helper
from ..modeling.base_helpers import BaseTaskHelper


class BaseExplainer(ABC):
    @property
    @abstractmethod
    def NAME(self):
        pass

    def __init__(
        self, model, tokenizer, model_helper: Optional[BaseTaskHelper] = None, **kwargs
    ):
        # We use the task_name parameter to specify the correct helper via the create_helper() function
        task_name = kwargs.pop('task_name', None)

        if model is None or tokenizer is None:
            raise ValueError("Please specify a model and a tokenizer.")

        self.init_args = kwargs

        # The user can now specify the task name even for explainers, and that will set the correct helper
        # even if no model_helper is specified. If the user does not specify anything, we show the Warning.
        if model_helper is None:
            if task_name is None:
                task_name = "text-classification"
                warnings.warn(
                    "No helper provided. Using default 'text-classification' helper."
                )
            self.helper = create_helper(model, tokenizer, task_name)
        else:
            self.helper=model_helper

    @property
    def device(self):
        return self.helper.model.device

    @property
    def model(self):
        return self.helper.model

    @property
    def tokenizer(self):
        return self.helper.tokenizer

    def _tokenize(self, text, **tok_kwargs):
        return self.helper._tokenize(text, **tok_kwargs)

    def get_tokens(self, text):
        return self.helper.get_tokens(text)

    def get_input_embeds(self, text):
        return self.helper.get_input_embeds(text)

    @abstractmethod
    def compute_feature_importance(
        self, text: str, target: int, target_token: Optional[str], **explainer_args
    ):
        pass

    def __call__(
        self,
        text: str,
        target: Union[str,int],
        target_token: Optional[str] = None,
        **explainer_args
    ):
        return self.compute_feature_importance(
            text, target, target_token, **explainer_args
        )
