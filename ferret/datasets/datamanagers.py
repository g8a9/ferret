"""Datasets API"""


from typing import List

import numpy as np
import pytreebank

from . import BaseDataset

TRAIN_SET = "train"
VALIDATION_SET = "validation"
TEST_SET = "test"

NONE_RATIONALE = []

from .utils_sst_rationale_generation import get_sst_rationale


class HateXplainDataset(BaseDataset):

    NAME = "HateXplain"
    avg_rationale_size = 7
    # np.mean([sum(self._get_rationale(i, split_type="train")[self._get_ground_truth(i, split_type="train")]) for i in range(self.len("train"))])

    def __init__(self, tokenizer):
        from datasets import load_dataset

        dataset = load_dataset("hatexplain")
        self.train_dataset = dataset["train"]
        self.validation_dataset = dataset["validation"]
        self.test_dataset = dataset["test"]
        self.tokenizer = tokenizer
        self.top_k_hard_rationale = 7
        self.classes = [0, 1, 2]

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
                f"{split_type} not supported as split_type. Specify one among: train, validation or test."
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
                    f"{split_type} not supported as split_type. Specify one among:  train, validation or test."
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
        text = " ".join(post_tokens)
        return text

    def _get_rationale(self, idx, split_type: str = TEST_SET, rationale_union=True):
        item_idx = self._get_item(idx, split_type)
        word_based_tokens = item_idx["post_tokens"]

        # All hatexplain rationales are defined for the label, only for hatespeech or offensive classes
        rationale_label = self._get_ground_truth(idx, split_type)

        rationale_by_label = [NONE_RATIONALE for c in self.classes]
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
                    rationale_by_label[rationale_label] = [
                        self.get_true_rationale_from_words_to_tokens(
                            word_based_tokens, rationale
                        )
                        for rationale in rationales
                    ]
                    return rationale_by_label
            else:
                rationale = rationales
        rationale_by_label[
            rationale_label
        ] = self.get_true_rationale_from_words_to_tokens(word_based_tokens, rationale)

        return rationale_by_label

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


class MovieReviews(BaseDataset):

    NAME = "MovieReviews"
    avg_rationale_size = 78

    def __init__(self, tokenizer):
        from datasets import load_dataset

        dataset = load_dataset("movie_rationales")
        self.train_dataset = dataset["train"]
        self.validation_dataset = dataset["validation"]
        self.test_dataset = dataset["test"]
        self.tokenizer = tokenizer
        self.classes = [0, 1]

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
                f"{split_type} not supported as split_type. Specify one among:  train, validation or test."
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
        text = item_idx["review"]
        text = text.replace("\n", " ")
        return text

    def _get_offset_rationale(self, text, text_rationales):
        tokenizer = self.tokenizer
        rationale_offsets = []

        for text_rationale in text_rationales:

            start_i = text.index(text_rationale)
            end_i = start_i + len(text_rationale)
            rationale_encoded_text = tokenizer.encode_plus(
                text[start_i:end_i],
                return_offsets_mapping=True,
                return_attention_mask=False,
            )
            rationale_token_offset = [
                (s + start_i, e + start_i)
                for (s, e) in rationale_encoded_text["offset_mapping"]
                if (s == 0 and e == 0) == False
            ]
            rationale_offsets.append(rationale_token_offset)
        return rationale_offsets

    def _get_rationale_one_hot_encoding(self, offsets, rationale_offsets, len_tokens):
        rationale = np.zeros(len_tokens)

        for rationale_offset in rationale_offsets:
            if rationale_offset in offsets:
                rationale[offsets.index(rationale_offset)] = 1

        return rationale

    def _get_rationale(self, idx, split_type: str = TEST_SET, rationale_union=True):

        item_idx = self._get_item(idx, split_type)
        text = self._get_text(item_idx)

        tokenizer = self.tokenizer
        encoded_text = tokenizer.encode_plus(
            text, return_offsets_mapping=True, return_attention_mask=False
        )
        tokens = tokenizer.convert_ids_to_tokens(encoded_text["input_ids"])
        offsets = encoded_text["offset_mapping"]

        rationale_field_name = "evidences"

        # Movie rationales are defined for the ground truth label
        rationale_label = self._get_ground_truth(idx, split_type)

        rationale_by_label = [NONE_RATIONALE for c in self.classes]

        if rationale_field_name in item_idx:
            text_rationales = item_idx[rationale_field_name]

            rationale_offsets = self._get_offset_rationale(text, text_rationales)
            if len(text_rationales) > 0 and isinstance(text_rationales, list):
                # It is a list of lists
                if rationale_union:
                    # We get the union of the rationales.
                    rationale_offsets = [t1 for t in rationale_offsets for t1 in t]
                    rationale_by_label[
                        rationale_label
                    ] = self._get_rationale_one_hot_encoding(
                        offsets, rationale_offsets, len(tokens)
                    ).astype(
                        int
                    )

                else:
                    # We return all of them (deprecated)
                    rationales = [
                        self._get_rationale_one_hot_encoding(
                            offsets, rationale_offset, len(tokens)
                        ).astype(int)
                        for rationale_offset in rationale_offsets
                    ]
                    rationale_by_label[rationale_label] = rationales
                    return rationale_by_label
            else:

                rationale_by_label[
                    rationale_label
                ] = self._get_rationale_one_hot_encoding(
                    offsets, rationale_offsets, len(tokens)
                ).astype(
                    int
                )

        return rationale_by_label

    def _get_ground_truth(self, idx, split_type: str = TEST_SET):
        item_idx = self._get_item(idx, split_type)
        label = item_idx["label"]
        return label

    def get_true_rationale_from_words_to_tokens(
        self, word_based_tokens: List[str], words_based_rationales: List[int]
    ):
        return super().get_true_rationale_from_words_to_tokens(
            word_based_tokens, words_based_rationales
        )


class SSTDataset(BaseDataset):

    NAME = "SST"
    avg_rationale_size = 10

    def __init__(self, tokenizer):
        from datasets import load_dataset

        dataset = load_dataset("sst")
        self.train_dataset = dataset["train"]
        self.validation_dataset = dataset["validation"]
        self.test_dataset = dataset["test"]
        self.sst_ptb = load_dataset("sst", "ptb")
        self.tokenizer = tokenizer
        self.classes = [0, 1]

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

    def get_instance(self, idx, split_type: str = TEST_SET):
        item_idx = self._get_item(idx, split_type)
        text = self._get_text(item_idx)
        tokens = (
            [self.tokenizer.cls_token]
            + self.tokenizer.tokenize(text)
            + [self.tokenizer.sep_token]
        )
        rationale = self._get_rationale(idx, item_idx=item_idx, split_type=split_type)
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

    def _get_rationale(self, idx, item_idx=None, split_type: str = TEST_SET):
        if item_idx is None:
            item_idx = self._get_item(idx, split_type)
        word_based_tokens = item_idx["tokens"].split("|")

        # Rationales are defined for the ground truth
        rationale_label = self._get_ground_truth(idx, split_type)

        rationale_by_label = [NONE_RATIONALE for c in self.classes]

        # Get rationale from tree
        tree_str = self.sst_ptb[split_type]["ptb_tree"][idx]
        tree = pytreebank.create_tree_from_string(tree_str)
        rationale = get_sst_rationale(tree)

        # Convert rationale in one hot encoding.
        rationale = [1 if r > 0 else 0 for r in rationale]

        # We convert from word importance to token importance.
        # If a word is relevant, we say that all tokens of the word are relevant.
        rationale_by_label[
            rationale_label
        ] = self.get_true_rationale_from_words_to_tokens(word_based_tokens, rationale)
        return rationale_by_label

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
