import math
import pdb
from typing import List, Optional, Tuple, Union
import logging

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.autonotebook import tqdm
from transformers.tokenization_utils_base import BatchEncoding

from .base_helpers import BaseTaskHelper


class BaseTextTaskHelper(BaseTaskHelper):
    """
    Base helper class to handle basic steps of the pipeline (e.g., tokenization, inference).
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @property
    def targets(self) -> List[int]:
        return self.model.config.id2label

    def list_tokens(self, example: str, as_dict: bool = False) -> List[str]:
        """List tokens for a given example

        :param example str: the example to tokenize
        :param as_dict bool: whether to return a dictionary with the relevant information. If True, the dictionary will be in the format {positional_index: (input_id, token)}. Default: False
        """
        item = self._tokenize(example)
        text_tokens = self.tokenizer.convert_ids_to_tokens(item["input_ids"][0])

        if as_dict:
            return {
                pos: (iid, token)
                for pos, (iid, token) in enumerate(
                    zip(item["input_ids"][0], text_tokens)
                )
            }
        else:
            return text_tokens

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

        # Process each batch
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

        # Concatenate logits
        logits = torch.cat([o.logits for o in outputs], dim=0)

        return outputs, logits

    def _check_target(self, target, **kwargs):
        return target

    def _check_sample(self, input, **kwargs):
        return input

    def _check_target_token(self, text, target_token, **kwargs):
        return None


class SequenceClassificationHelper(BaseTextTaskHelper):
    HELPER_TYPE = "sequence-classification"

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

    def _check_target(self, target):
        if isinstance(target, str) and target not in self.model.config.label2id:
            raise ValueError(
                f"Target {target} is not a valid target. Use a string among: {list(self.model.config.label2id.keys())}"
            )
        if isinstance(target, int) and target not in self.model.config.id2label:
            raise ValueError(
                f"Target {target} is not a valid target. Use an integer among: {list(self.model.config.id2label.keys())}"
            )

        if isinstance(target, str):
            target = self.model.config.label2id[target]
        return target

    def _check_sample(self, text):
        if not any(
            (isinstance(text, str), isinstance(text, tuple), isinstance(text, list))
        ):
            raise ValueError("Input sample type is not supported")
        
        sep_token = self.tokenizer.sep_token if hasattr(self.tokenizer, 'sep_token') else "[SEP]"
        if sep_token == "[SEP]":
            logging.warning("Using hardcoded '[SEP]' as separator token.")

        # The SequenceClassificationHelper is only used for the text-classification and Natural Language Inference tasks.
        # The following condition takes care of the NLI task (which was causing problems) in the SHAP explainer. 
        # The expected input for the NLI task is constructed as follows:

        # >>> premise = "I first thought that I liked the movie, but upon second thought it was actually disappointing."
        # >>> hypothesis = "The movie was good."
        # >>> sample = (premise, hypothesis)

        # where the tuple given by "sample" is indeed the expected text input to the explainer.
        # As currently designed, the SHAP explainer expects the input instead to be either a string or a list with a single string, 
        # and *not* a list of a tuple. We her construct the text input for the explainers as "premise <separator_token> hypothesis"
        if isinstance(text, str):
            return [text]
        
        elif isinstance(text, tuple) and len(text) == 2 and all(isinstance(t, str) for t in text):
            return [text[0] + f" {sep_token} " + text[1]]
        else:
            return text


class ZeroShotTextClassificationHelper(BaseTextTaskHelper):
    HELPER_TYPE = "zero-shot-text-classification"
    DEFAULT_TEMPLATE = "This is {}"

    def _score(
        self,
        sample,
        return_dict: bool,
        options: List[str] = None,
        template: str = None,
        class_label: str = "entailment",
        return_probs: bool = False,
    ):
        """Compute prediction scores for a single query

        :param text str: query to compute the logits from
        :param return_dict bool: return a dict in the format Class Label -> score.
        Otherwise, return softmaxed logits as torch.Tensor. Default True
        """
        if template is None:
            template = self.DEFAULT_TEMPLATE

        texts = [(sample, template.format(opt)) for opt in options]

        # pdb.set_trace()
        _, logits = self._forward(texts, output_hidden_states=False)
        scores = logits.softmax(-1)

        ent_idx = self.model.config.label2id[class_label]
        scores = scores[:, ent_idx]

        if return_probs:
            scores = scores.softmax(-1)

        if return_dict:
            scores = {opt: s.item() for opt, s in zip(options, scores)}
        return scores

    def _prepare_sample(self, sample, **kwargs):
        target_option = kwargs["target_option"]
        # Simikarly to what done for the NLI task above we combine sample and target option through a separator token token
        sep_token = self.tokenizer.sep_token if hasattr(self.tokenizer, 'sep_token') else "[SEP]"
        if sep_token == "[SEP]":
            logging.warning("Using hardcoded '[SEP]' as separator token.")
        return [sample + f" {sep_token} " + self.DEFAULT_TEMPLATE.format(target_option)]

    def _check_target(self, target):
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


class TokenClassificationHelper(BaseTextTaskHelper):
    HELPER_TYPE = "token-classification"

    def _score(self, text: str, return_dict: bool = True):
        """Compute prediction scores for a single query

        :param text str: query to compute the logits from
        :param return_dict bool: return a dict in the format Class Label -> score. Otherwise, return softmaxed logits as torch.Tensor. Default True
        """
        _, logits = self._forward(text, output_hidden_states=False)
        scores = logits[0].softmax(-1)  # sequence length x num_labels

        if return_dict:
            # TODO We do not perform a clever aggregation here. Might be worth introducing it.
            tokens_dict = self.list_tokens(text, as_dict=True)

            scores_dict = dict()
            for pos_idx, (input_id, token) in tokens_dict.items():

                scores_dict[pos_idx] = (
                    token,
                    {
                        self.model.config.id2label[idx]: value.item()
                        for idx, value in enumerate(scores[pos_idx])
                    },
                )

            return scores_dict
        return scores

    def _check_target(self, target):
        if isinstance(target, str) and target not in self.model.config.label2id:
            raise ValueError(
                f"Target {target} is not a valid target. Use a string among: {list(self.model.config.label2id.keys())}"
            )
        if isinstance(target, int) and target not in self.model.config.id2label:
            raise ValueError(
                f"Target {target} is not a valid target. Use an integer among: {list(self.model.config.id2label.keys())}"
            )

        if isinstance(target, str):
            target = self.model.config.label2id[target]
        return target

    def _check_sample(self, text):
        if not any(
            (isinstance(text, str), isinstance(text, tuple), isinstance(text, list))
        ):
            raise ValueError("Input sample type is not supported")

        if isinstance(text, str) or isinstance(text, tuple):
            return [text]
        else:
            return text

    def _check_target_token(
        self, text: Optional[str] = None, target_token: Optional[str] = None
    ):
        if target_token is None:
            raise ValueError(
                "Target token must be specified for a TokenClassification task"
            )

        tokens_list = self.list_tokens(text)
        if isinstance(target_token, str):
            try:
                target_token_idx = tokens_list.index(target_token)
                return target_token_idx
            except:
                raise ValueError(
                    f"Target token {target_token} is not in tokens {tokens_list}"
                )

        if isinstance(target_token, int):
            if target_token >= len(tokens_list):
                raise ValueError(
                    f"Target token index {target_token} is out of range of tokens {tokens_list}. (Choose a number between 0 and {len(tokens_list) - 1})"
                )
            return target_token

    def _postprocess_logits(self, logits, **kwargs):
        target_token_pos_idx = kwargs["target_token_pos_idx"]
        return logits[:, target_token_pos_idx, :]
