import numpy as np
import torch
from typing import Union, List
import math
from tqdm.autonotebook import tqdm
from transformers.tokenization_utils_base import BatchEncoding
import pdb


class ModelHelper:
    """
    Wrapper class to interface with HuggingFace models
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def _tokenize(self, text: str, **tok_kwargs) -> BatchEncoding:
        """
        Base tokenization strategy for a single text.

        Note that we truncate to the maximum length supported by the model.

        :param text str: the string to tokenize
        """
        return self.tokenizer(text, return_tensors="pt", truncation=True, **tok_kwargs)

    def get_input_embeds(self, text: str) -> torch.Tensor:
        """Extract input embeddings

        :param text str: the string to extract embeddings from.
        """
        item = self._tokenize(text)
        item = {k: v.to(self.model.device) for k, v in item.items()}
        embeddings = self._get_input_embeds_from_ids(item["input_ids"][0])
        embeddings = embeddings.unsqueeze(0)
        return embeddings

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

        if show_progress:
            pbar = tqdm(total=n_batches, desc="Batch", leave=False)

        outputs = list()
        for emb, mask in zip(input_batches, mask_batches):
            out = self.model(
                inputs_embeds=emb,
                attention_mask=mask,
                output_hidden_states=output_hidden_states,
            )
            outputs.append(out)

            if show_progress:
                pbar.update(1)

        if show_progress:
            pbar.close()

        logits = torch.cat([o.logits for o in outputs])
        return outputs, logits

    def _forward(
        self,
        text: Union[str, List[str]],
        batch_size=8,
        show_progress=False,
        use_input_embeddings=False,
        output_hidden_states=True,
        **tok_kwargs
    ):
        if isinstance(text, str):
            text = [text]

        n_batches = math.ceil(len(text) / batch_size)
        batches = np.array_split(text, n_batches)

        outputs = list()
        with torch.no_grad():

            if show_progress:
                pbar = tqdm(total=n_batches, desc="Batch", leave=False)

            for batch in batches:
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

                if show_progress:
                    pbar.update(1)

        if show_progress:
            pbar.close()

        logits = torch.cat([o.logits for o in outputs])
        return outputs, logits

    # def _get_class_predicted_probability(self, text, tokenizer, target):
    #     outputs = self._forward(text, tokenizer)
    #     logits = outputs.logits[0]
    #     class_prob = logits.softmax(-1)[target].item()
    #     return class_prob

    # def _get_tokenizer(self, tokenizer=None):
    #     tokenizer = tokenizer if tokenizer else self.tokenizer
    #     if tokenizer is None:
    #         raise ValueError("Tokenizer is not specified")
    #     return tokenizer

    # def get_predicted_label(self, text, tokenizer=None):
    #     tokenizer = self._get_tokenizer(tokenizer)
    #     outputs = self._forward(text, tokenizer)
    #     logits = outputs.logits

    #     prediction = logits.argmax(-1).item()
    #     return prediction

    # # TODO - Uniformate
    # def _get_class_predicted_probabilities_texts(self, texts, tokenizer, target):
    #     # TODO
    #     tokenizer = tokenizer if tokenizer else self.tokenizer
    #     if tokenizer is None:
    #         raise ValueError("Tokenizer is not specified")
    #     inputs = tokenizer(texts, return_tensors="pt", padding="longest")

    #     with torch.no_grad():
    #         outputs = self.model(**inputs)

    #     return outputs.logits.softmax(-1)[:, target]

    # def _forward(self, idx, tokenizer=None, no_grad=True, use_inputs=False):
    #     self.model.eval()
    #     tokenizer = tokenizer if tokenizer else self.tokenizer
    #     if tokenizer is None:
    #         raise ValueError("Tokenizer is not specified")

    #     item = tokenizer(idx, return_tensors="pt")

    #     def _foward_pass(use_inputs=False):

    #         if use_inputs:
    #             embeddings = self._get_input_embeds(idx)
    #             outputs = self.model(
    #                 inputs_embeds=embeddings,
    #                 **item,
    #                 output_hidden_states=True,
    #             )

    #             return outputs, embeddings

    #         else:
    #             outputs = self.model(
    #                 **item, output_attentions=True, output_hidden_states=True
    #             )
    #             return outputs

    #     if no_grad:
    #         with torch.no_grad():
    #             outputs = _foward_pass(use_inputs)
    #     else:
    #         outputs = _foward_pass(use_inputs)

    #     return outputs
