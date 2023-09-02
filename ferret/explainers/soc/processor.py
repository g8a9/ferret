import csv
import os

import pandas as pd
import torch
from datasets import Dataset, DatasetDict, Value
from regex import D
from torch.utils.data import DataLoader, Dataset
from transformers import DataCollatorWithPadding

from .common import *


class Processor:
    def __init__(
        self, tokenizer, train_texts, train_labels, validation_texts, validation_labels
    ):
        self.tokenizer = tokenizer
        self.raw_datasets = DatasetDict(
            train=Dataset.from_dict({"text": train_texts, "label": train_labels}),
            valid=Dataset.from_dict(
                {"text": validation_texts, "label": validation_labels}
            ),
        )

        def preprocess_function(examples):
            return tokenizer(examples["text"], truncation=True)

        self.proc_datasets = self.raw_datasets.map(
            preprocess_function, batched=True, remove_columns=["text"]
        ).cast_column("label", Value(dtype="int16"))

    def get_train_dataloader(self):
        collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        return DataLoader(self.proc_datasets["train"], collate_fn=collator)

    def get_valid_dataloader(self):
        collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        return DataLoader(self.proc_datasets["valid"], collate_fn=collator)


# class ProcessorOld(DataProcessor):
#     """
#     Data processor using DataProcessor class provided by BERT
#     """

#     def __init__(self, configs, tokenizer, train_file, valid_file):
#         super().__init__()
#         self.label_groups = [0, 1]
#         self.tokenizer = tokenizer
#         self.max_seq_length = 128
#         self.configs = configs

#         self.train_file = train_file
#         self.valid_file = valid_file

#     def _create_examples(self, split, label=None):
#         """
#         Create a list of InputExample, where .text_a is raw text and .label is specified
#         as configs.label_groups
#         :param split:
#         :param label:
#         :return:
#         """
#         if split == "train":
#             f = self.train_file
#         else:
#             f = self.valid_file

#         df = pd.read_csv(f, sep="\t")
#         examples = list()
#         for row in df.itertuples():
#             print(row)
#             example = InputExample(text_a=row.text, guid=row.id)
#             example.label = int(row.misogynous)
#             examples.append(example)
#         return examples

#     def get_train_examples(self, label=None):
#         return self._create_examples("train", label)

#     def get_dev_examples(self, label=None):
#         return self._create_examples("dev", label)

#     def get_test_examples(self, label=None):
#         return self._create_examples("test", label)

#     def get_example_from_tensor_dict(self, tensor_dict):
#         raise NotImplementedError

#     def get_labels(self):
#         return [0, 1]

#     def get_features(self, split):
#         """
#         Return a list of dict, where each dict contains features to be fed into the BERT model
#         for each instance. ['text'] is a LongTensor of length configs.max_seq_length, either truncated
#         or padded with 0 to match this length.
#         :param split: 'train' or 'dev'
#         :return:
#         """
#         examples = self._create_examples(split)
#         features = []
#         for example in examples:
#             tokens = self.tokenizer.tokenize(example.text_a)
#             if len(tokens) > self.max_seq_length - 2:
#                 tokens = tokens[: (self.max_seq_length - 2)]
#             tokens = ["[CLS]"] + tokens + ["[SEP]"]
#             input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
#             padding = [0] * (self.max_seq_length - len(input_ids))
#             input_ids += padding
#             input_ids = torch.LongTensor(input_ids)
#             features.append({"text": input_ids, "length": len(tokens)})
#         return features

#     def get_dataloader(self, split, batch_size=1):
#         """
#         return a torch.utils.DataLoader instance, mainly used for training the language model.
#         :param split:
#         :param batch_size:
#         :return:
#         """
#         features = self.get_features(split)
#         dataset = SimpleDataset(features)
#         dataloader = DataLoader(dataset, batch_size=batch_size)
#         return dataloader

#     def set_tokenizer(self, tokenizer):
#         self.tokenizer = tokenizer


class SimpleDataset(Dataset):
    """
    torch.utils.Dataset instance for building torch.utils.DataLoader, for training the language model.
    """

    def __init__(self, features):
        super().__init__()
        self.features = features

    def __getitem__(self, item):
        return self.features[item]

    def __len__(self):
        return len(self.features)
