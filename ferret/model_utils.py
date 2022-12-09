import math
import pdb
from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import numpy as np
import torch
from tqdm.autonotebook import tqdm
from transformers.tokenization_utils_base import BatchEncoding


class BaseTaskHelper(ABC):
    """
    Base helper class to handle basic steps of the pipeline (e.g., tokenization, inference).
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @abstractmethod
    def check_format_target(self, target):
        """Validate the specific target requested for the explanaiton"""
        pass

    @abstractmethod
    def check_format_input(self, input):
        """Validate the specific input requested for the explanaiton"""
        pass

    def format_target(self, target):
        """Format the target variable

        In all our current explainers, 'target' must be a positional integer for the
        logits matrix. Default: leave target unchanged.
        """
        return target

    def get_input_embeds(self, text: str) -> torch.Tensor:
        """Extract input embeddings

        :param text str: the string to extract embeddings from.
        """
        item = self._tokenize(text)
        item = {k: v.to(self.model.device) for k, v in item.items()}
        embeddings = self._get_input_embeds_from_ids(item["input_ids"][0])
        embeddings = embeddings.unsqueeze(0)
        return embeddings

    def _tokenize(self, text: str, **tok_kwargs) -> BatchEncoding:
        """
        Base tokenization strategy for a single text.

        Note that we truncate to the maximum length supported by the model.

        :param text str: the string to tokenize
        """
        return self.tokenizer(text, return_tensors="pt", truncation=True, **tok_kwargs)

    def _get_input_embeds_from_ids(self, ids) -> torch.Tensor:
        return self.model.get_input_embeddings()(ids)

    def get_tokens(self, text: str, **tok_kwargs) -> List[str]:
        """Extract a list of tokens

        :param text str: the string to extract tokens from.
        """
        item = self._tokenize(text)
        input_len = item["attention_mask"].sum()
        ids = item["input_ids"][0][:input_len]
        return self.tokenizer.convert_ids_to_tokens(ids, **tok_kwargs)

    def _forward_with_input_embeds(
        self,
        input_embeds,
        attention_mask,
        batch_size=8,
        show_progress=False,
        output_hidden_states=False,
    ):
        input_len = input_embeds.shape[0]
        n_batches = math.ceil(input_len / batch_size)
        input_batches = torch.tensor_split(input_embeds, n_batches)
        mask_batches = torch.tensor_split(attention_mask, n_batches)

        outputs = list()
        for emb, mask in tqdm(
            zip(input_batches, mask_batches),
            total=n_batches,
            desc="Batch",
            leave=False,
            disable=not show_progress,
        ):
            out = self.model(
                inputs_embeds=emb,
                attention_mask=mask,
                output_hidden_states=output_hidden_states,
            )
            outputs.append(out)

        logits = torch.cat([o.logits for o in outputs])
        return outputs, logits

    @torch.no_grad()
    def _forward(
        self,
        text: Union[str, List[str], Tuple[str, str]],
        batch_size=8,
        show_progress=False,
        use_input_embeddings=False,
        output_hidden_states=True,
        **tok_kwargs,
    ):
        if not isinstance(text, list):
            text = [text]

        n_batches = math.ceil(len(text) / batch_size)
        batches = np.array_split(text, n_batches)

        outputs = list()

        for batch in tqdm(
            batches,
            total=n_batches,
            desc="Batch",
            leave=False,
            disable=not show_progress,
        ):
            item = self._tokenize(batch.tolist(), padding="longest", **tok_kwargs)
            item = {k: v.to(self.model.device) for k, v in item.items()}

            if use_input_embeddings:
                ids = item.pop("input_ids")  # (B,S,d_model)
                input_embeddings = self._get_input_embeds_from_ids(ids)
                out = self.model(
                    inputs_embeds=input_embeddings,
                    **item,
                    output_hidden_states=output_hidden_states,
                )
            else:
                out = self.model(**item, output_hidden_states=output_hidden_states)
            outputs.append(out)

        logits = torch.cat([o.logits for o in outputs])
        return outputs, logits

    def _score(self, text: str, return_dict: bool = True):
        """Compute prediction scores for a single query

        :param text str: query to compute the logits from
        :param return_dict bool: return a dict in the format Class Label -> score. Otherwise, return softmaxed logits as torch.Tensor. Default True
        """
        _, logits = self._forward(text, output_hidden_states=False)
        scores = logits[0].softmax(-1)

        if return_dict:
            scores = {
                self.model.config.id2label[idx]: value.item()
                for idx, value in enumerate(scores)
            }
        return scores


def create_helper(model, tokenizer, task_name):
    helper = SUPPORTED_TASKS_TO_HELPERS.get(task_name, None)
    if helper is None:
        raise ValueError(f"Task {task_name} is not supported.")
    else:
        return helper(model, tokenizer)


class TaskClassificationHelper(BaseTaskHelper):
    def check_format_target(self, target):
        if isinstance(target, str) and target not in self.model.config.label2id:
            raise ValueError(
                f"Target {target} is not a valid target. Use a string among: {list(self.model.config.label2id.keys())}"
            )
        if isinstance(target, int) and target not in self.model.config.id2label:
            raise ValueError(
                f"Target {target} is not a valid target. Use an integer amond: {list(self.model.config.id2label.keys())}"
            )

        if isinstance(target, str):
            target = self.model.config.label2id[target]
        return target

    def check_format_input(self, text):
        if not any(
            (isinstance(text, str), isinstance(text, tuple), isinstance(text, list))
        ):
            raise ValueError("Input sample type is not supported")

        if isinstance(text, str) or isinstance(text, tuple):
            return [text]
        else:
            return text


SUPPORTED_TASKS_TO_HELPERS = {"text-classification": TaskClassificationHelper}
