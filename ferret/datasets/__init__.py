"""Datasets API"""

from abc import ABC, abstractmethod
from typing import List

TRAIN_SET = "TRAIN_SET"
VALIDATION_SET = "VALIDATION_SET"
TEST_SET = "TEST_SET"


class BaseDataset(ABC):
    @property
    @abstractmethod
    def NAME(self):
        pass

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @abstractmethod
    def get_instance(self, idx: int, split_type: str = TEST_SET):
        pass

    @abstractmethod
    def _get_item(self, idx: int, split_type: str = TEST_SET):
        pass

    @abstractmethod
    def _get_text(self, idx, split_type: str = TEST_SET):
        pass

    @abstractmethod
    def _get_rationale(self, idx, split_type: str = TEST_SET):
        pass

    @abstractmethod
    def _get_ground_truth(self, idx, split_type: str = TEST_SET):
        pass

    def get_true_rationale_from_words_to_tokens(
        self, word_based_tokens: List[str], words_based_rationales: List[int]
    ) -> List[int]:
        # original_tokens --> list of words.
        # rationale_original_tokens --> 0 or 1, if the token belongs to the rationale or not
        # Typically, the importance is associated with each word rather than each token.
        # We convert each word in token using the tokenizer. If a word is in the rationale,
        # we consider as important all the tokens of the word.
        token_rationale = []
        for t, rationale_t in zip(word_based_tokens, words_based_rationales):
            converted_token = self.tokenizer.encode(t)[1:-1]

            for token_i in converted_token:
                token_rationale.append(rationale_t)
        return token_rationale
