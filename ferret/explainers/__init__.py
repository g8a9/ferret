"""Explainers API"""

from abc import ABC, abstractmethod
from einops import rearrange


class BaseExplainer(ABC):
    @property
    @abstractmethod
    def NAME(self):
        pass

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @abstractmethod
    def compute_feature_importance(self, text: str, target: int):
        pass

    def __call__(self, text: str, target: int = 1):
        return self.compute_feature_importance(text, target)

    def _tokenize(self, text):
        """Base tokenization strategy for a single text.

        Note that we truncate to the maximum length supported by the model.
        """
        return self.tokenizer(text, return_tensors="pt", truncation=True)

    def get_input_embeds(self, text):
        item = self._tokenize(text)
        embeddings = self._get_input_embeds_from_ids(item["input_ids"][0])
        embeddings = rearrange(embeddings, "s h -> () s h")
        return embeddings

    def get_tokens(self, text):
        item = self._tokenize(text)
        input_len = item["attention_mask"].sum()
        return self.tokenizer.convert_ids_to_tokens(item["input_ids"][0][:input_len])

    def _get_input_embeds_from_ids(self, ids):
        embeddings = self.model.get_input_embeddings()(ids)
        return embeddings
