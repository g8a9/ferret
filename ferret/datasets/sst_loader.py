"""Datasets API"""


from . import BaseDataset
import numpy as np
import os
from typing import List
import pickle

TRAIN_SET = "TRAIN_SET"
VALIDATION_SET = "VALIDATION_SET"
TEST_SET = "TEST_SET"


class SSTDataset(BaseDataset):

    NAME = "SST"

    def __init__(self, tokenizer, rationales=None, rationales_input_dir=None):
        from datasets import load_dataset

        dataset = load_dataset("sst")
        self.train_dataset = dataset["train"]
        self.validation_dataset = dataset["validation"]
        self.test_dataset = dataset["test"]
        self.tokenizer = tokenizer
        if rationales is None:
            if rationales_input_dir is None:
                raise ValueError(
                    "Specify the rationales or the dir in which they are stored. Consider run first: data/sst_process.py"
                )
            rationales = {}
            for set_name, set_name_key in zip(
                ["train", "validation", "test"], [TRAIN_SET, VALIDATION_SET, TEST_SET]
            ):
                with open(
                    os.path.join(
                        rationales_input_dir, f"sst_rationales_{set_name}.pickle"
                    ),
                    "rb",
                ) as handle:
                    rationales[set_name_key] = pickle.load(handle)
        self.rationales = rationales

    def __len__(self):
        # We use the TEST_SET as default
        return self.len()

    def len(self, split_type: str = TEST_SET):
        if split_type == TRAIN_SET:
            return len(self.train_dataset)
        elif split_type == VALIDATION_SET:
            return len(self.validation_dataset)
        elif split_type == TEST_SET:
            return len(self.test_dataset)
        else:
            raise ValueError(
                f"{split_type} not supported as split_type. Specify one among: TRAIN_SET, VALIDATION_SET or TEST_SET."
            )

    def _get_item(self, idx: int, split_type: str = TEST_SET):
        if isinstance(idx, int):
            if split_type == TRAIN_SET:
                item_idx = self.train_dataset[idx]
            elif split_type == VALIDATION_SET:
                item_idx = self.validation_dataset[idx]
            elif split_type == TEST_SET:
                item_idx = self.test_dataset[idx]
            else:
                raise ValueError(
                    f"{split_type} not supported as split_type. Specify one among: TRAIN_SET, VALIDATION_SET or TEST_SET."
                )
            return item_idx
        elif isinstance(idx, dict):
            return idx
        else:
            raise ValueError()

    def __getitem__(self, idx, target=1):
        # We use the TEST_SET as default
        return self.get_instance(idx, target=target)

    def get_instance(self, idx, split_type: str = TEST_SET, target=1):
        item_idx = self._get_item(idx, split_type)
        text = self._get_text(item_idx)
        tokens = (
            [self.tokenizer.cls_token]
            + self.tokenizer.tokenize(text)
            + [self.tokenizer.sep_token]
        )
        rationale = self._get_rationale(
            idx, item_idx=item_idx, split_type=split_type, target=target
        )
        true_label = self._get_ground_truth(item_idx, split_type)
        return {
            "text": text,
            "tokens": tokens,
            "rationale": rationale,
            "label": true_label,
        }

    def _get_text(self, idx, split_type: str = TEST_SET):
        item_idx = self._get_item(idx, split_type)
        text = item_idx["sentence"]
        return text

    def _get_rationale(self, idx, item_idx=None, split_type: str = TEST_SET, target=1):
        if item_idx is None:
            item_idx = self._get_item(idx, split_type)
        word_based_tokens = item_idx["tokens"].split("|")

        rationale = self.rationales[split_type][idx]
        # Convert rationale in one hot encoding.

        if target == 1:
            # If we want the rationale for the POSITIVE class,
            # if the score is greater than 0, we say that the word is important
            rationale = [1 if r > 0 else 0 for r in rationale]
        else:
            # If we want the rationale for the NEGATIVE class,
            # if the score is lower than 0, we say that the word is important
            rationale = [1 if r < 0 else 0 for r in rationale]

        # We convert from word importance to token importance.
        # If a word is relevant, we say that all tokens of the word are relevant.
        rationale = self.get_true_rationale_from_words_to_tokens(
            word_based_tokens, rationale
        )
        return rationale

    def _get_ground_truth(self, idx, split_type: str = TEST_SET, threshold=0.5):
        item_idx = self._get_item(idx, split_type)
        label_score = item_idx["label"]
        if label_score > threshold:
            label = 1
        else:
            label = 0
        return label

    def get_true_rationale_from_words_to_tokens(
        self, word_based_tokens: List[str], words_based_rationales: List[int]
    ):
        return super().get_true_rationale_from_words_to_tokens(
            word_based_tokens, words_based_rationales
        )
