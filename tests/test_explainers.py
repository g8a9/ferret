#!/usr/bin/env python

"""Tests for `ferret` package."""


from re import T
import unittest
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from ferret import Benchmark, SHAPExplainer, LIMEExplainer
import numpy as np


DEFAULT_EXPLAINERS_NUM = 6


class TestExplainers(unittest.TestCase):
    """Tests for `ferret` package."""

    def setUp(self):
        self.m = AutoModelForSequenceClassification.from_pretrained(
            "lvwerra/distilbert-imdb"
        )
        self.t = AutoTokenizer.from_pretrained("lvwerra/distilbert-imdb")
        self.bench = Benchmark(self.m, self.t)

    def test_initialization(self):
        self.assertEqual(len(self.bench.explainers), DEFAULT_EXPLAINERS_NUM)

    def test_shap(self):
        text = "You look stunning!"
        exp = SHAPExplainer(self.m, self.t)
        explanation = exp(text)
        self.assertListEqual(
            explanation.tokens, ["[CLS]", "you", "look", "stunning", "!", "[SEP]"]
        )
        self.assertTrue(
            np.allclose(
                explanation.scores,
                np.array([0.0, 0.05189601, -0.0196495, 0.37571134, 0.06520349, 0.0]),
            )
        )
        self.assertEqual(explanation.target, 1)

    def test_lime(self):
        text = "You look so stunning!"
        exp = LIMEExplainer(self.m, self.t)
        explanation = exp(text, call_args={"num_samples": 30, "show_progress": True})
        self.assertListEqual(
            explanation.tokens, ["[CLS]", "you", "look", "so", "stunning", "!", "[SEP]"]
        )
        #  LIME samples randomnly points in the neighborhood, attribution scores are not deteministic.
        self.assertEqual(explanation.target, 1)