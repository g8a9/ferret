"""Datasets API"""


from . import BaseDataset
from typing import List

TRAIN_SET = "TRAIN_SET"
VALIDATION_SET = "VALIDATION_SET"
TEST_SET = "TEST_SET"


class HateXplainDataset(BaseDataset):

    NAME = "HateXplain"

    def __init__(self, tokenizer):
        from datasets import load_dataset

        dataset = load_dataset("hatexplain")
        self.train_dataset = dataset["train"]
        self.validation_dataset = dataset["validation"]
        self.test_dataset = dataset["test"]
        self.tokenizer = tokenizer

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

    def __getitem__(self, idx):
        # We use the TEST_SET as default
        return self.get_instance(idx)

    def get_instance(self, idx, split_type: str = TEST_SET, rationale_union=True):
        item_idx = self._get_item(idx, split_type)
        text = self._get_text(item_idx)
        tokens = (
            [self.tokenizer.cls_token]
            + self.tokenizer.tokenize(text)
            + [self.tokenizer.sep_token]
        )
        rationale = self._get_rationale(item_idx, split_type, rationale_union)
        true_label = self._get_ground_truth(item_idx, split_type)
        return {
            "text": text,
            "tokens": tokens,
            "rationale": rationale,
            "label": true_label,
        }

    def _get_text(self, idx, split_type: str = TEST_SET):
        item_idx = self._get_item(idx, split_type)
        post_tokens = item_idx["post_tokens"]
        text = self.tokenizer.convert_tokens_to_string(post_tokens)
        return text

    def _get_rationale(self, idx, split_type: str = TEST_SET, rationale_union=True):
        item_idx = self._get_item(idx, split_type)
        word_based_tokens = item_idx["post_tokens"]
        rationale = []
        if "rationales" in item_idx:
            rationales = item_idx["rationales"]
            if len(rationales) > 0 and isinstance(rationales[0], list):
                # It is a list of lists
                if rationale_union:
                    # We get the union of the rationales.
                    rationale = [any(each) for each in zip(*rationales)]
                    rationale = [int(each) for each in rationale]
                else:
                    # We return all of them (deprecated)
                    rationales = [
                        self.get_true_rationale_from_words_to_tokens(
                            word_based_tokens, rationale
                        )
                        for rationale in rationales
                    ]
                    return rationales
            else:
                rationale = rationales
        rationale = self.get_true_rationale_from_words_to_tokens(
            word_based_tokens, rationale
        )
        return rationale

    def _get_ground_truth(self, idx, split_type: str = TEST_SET):
        item_idx = self._get_item(idx, split_type)
        labels = item_idx["annotators"]["label"]
        # Label by majority voting
        return max(set(labels), key=labels.count)

    def get_true_rationale_from_words_to_tokens(
        self, word_based_tokens: List[str], words_based_rationales: List[int]
    ):
        return super().get_true_rationale_from_words_to_tokens(
            word_based_tokens, words_based_rationales
        )
