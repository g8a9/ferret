#!/usr/bin/env python

"""Tests for `ferret` package."""

import unittest
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer
)
from ferret import Benchmark, LIMEExplainer, SHAPExplainer, GradientExplainer, IntegratedGradientExplainer, TokenClassificationHelper


DEFAULT_EXPLAINERS_NUM = 6
class TestExplainers(unittest.TestCase):
    """Tests for `ferret` package."""

    def setUp(self):
        #Text-Classification
        self.model_text_class = AutoModelForSequenceClassification.from_pretrained("lvwerra/distilbert-imdb")
        self.tokenizer_text_class = AutoTokenizer.from_pretrained("lvwerra/distilbert-imdb")

        # NLI
        self.model_nli = AutoModelForSequenceClassification.from_pretrained("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
        self.tokenizer_nli = AutoTokenizer.from_pretrained("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")

        # Zero-Shot Text Classification
        self.model_zero_shot = AutoModelForSequenceClassification.from_pretrained("MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
        self.tokenizer_zero_shot = AutoTokenizer.from_pretrained("MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")

        # Named Entity Recognition
        self.model_ner = AutoModelForTokenClassification.from_pretrained("Babelscape/wikineural-multilingual-ner").to("cpu")
        self.tokenizer_ner = AutoTokenizer.from_pretrained("Babelscape/wikineural-multilingual-ner")

        # Benchmark instances
        self.bench_text_class = Benchmark(self.model_text_class, self.tokenizer_text_class, task_name="text-classification")
        self.bench_nli = Benchmark(self.model_nli, self.tokenizer_nli, task_name="nli")
        self.bench_zero_shot = Benchmark(self.model_zero_shot, self.tokenizer_zero_shot, task_name="zero-shot-text-classification")
        self.bench_ner = Benchmark(self.model_ner, self.tokenizer_ner, task_name="ner")

    def test_initialization_text_class(self):
        self.assertEqual(len(self.bench_text_class.explainers), DEFAULT_EXPLAINERS_NUM)

    def test_initialization_nli(self):
        self.assertEqual(len(self.bench_nli.explainers), DEFAULT_EXPLAINERS_NUM)

    def test_initialization_zero_shot(self):
        self.assertEqual(len(self.bench_zero_shot.explainers), DEFAULT_EXPLAINERS_NUM)

    def test_initialization_ner(self):
        self.assertEqual(len(self.bench_ner.explainers), DEFAULT_EXPLAINERS_NUM)

    def test_shap_text_classification(self):
        text = "You look stunning!"
        exp = SHAPExplainer(self.model_text_class, self.tokenizer_text_class)
        explanation = exp(text, target=1)
        self.assertListEqual(
            explanation.tokens, ["[CLS]", "you", "look", "stunning", "!", "[SEP]"]
        )
        self.assertEqual(explanation.target_pos_idx, 1)

    def test_lime_text_classification(self):
        text = "You look so stunning!"
        exp = LIMEExplainer(self.model_text_class, self.tokenizer_text_class)
        explanation = exp(text, target=1, num_samples=30, show_progress=True)
        self.assertListEqual(
            explanation.tokens, ["[CLS]", "you", "look", "so", "stunning", "!", "[SEP]"]
        )
        self.assertEqual(explanation.target_pos_idx, 1)
        
    def test_gradient_text_classification(self):
        text = "The new movie is awesome!"
        exp = GradientExplainer(self.model_text_class, self.tokenizer_text_class, multiply_by_inputs=True)
        explanation = exp(text, target=1)
        self.assertListEqual(
            explanation.tokens, ["[CLS]", "the", "new", "movie", "is", "awesome", "!", "[SEP]"]
        )
        self.assertEqual(explanation.target_pos_idx, 1)

    def test_integrated_gradient_text_classification(self):
        text = "The new movie is awesome!"
        exp = IntegratedGradientExplainer(self.model_text_class, self.tokenizer_text_class, multiply_by_inputs=True)
        explanation = exp(text, target=1)
        self.assertListEqual(
            explanation.tokens, ["[CLS]", "the", "new", "movie", "is", "awesome", "!", "[SEP]"]
        )
        self.assertEqual(explanation.target_pos_idx, 1)
        
    def test_shap_nli(self):
        premise = "A soccer game with multiple males playing."
        hypothesis = "A sports activity."
        # adding task_name = "nli" is in this cas optional, as nli uses the default helper SequenceClassificationHelper
        exp = SHAPExplainer(self.model_nli, self.tokenizer_nli, task_name="nli")
        explanation = exp((premise, hypothesis), target="contradiction")
        self.assertListEqual(
            explanation.tokens[:5], ["[CLS]", "▁A", "▁soccer", "▁game", "▁with"]
        )
        self.assertEqual(explanation.target_pos_idx, 2)

    def test_lime_nli(self):
        premise = "A soccer game with multiple males playing."
        hypothesis = "A sports activity."
        exp = LIMEExplainer(self.model_nli, self.tokenizer_nli)
        explanation = exp((premise, hypothesis), target="entailment", num_samples=30, show_progress=True)
        self.assertListEqual(
            explanation.tokens[:5], ["[CLS]", "▁A", "▁soccer", "▁game", "▁with"]
        )
        self.assertEqual(explanation.target_pos_idx, 0)
        
    def test_gradient_nli(self):
        premise = "A soccer game with multiple males playing."
        hypothesis = "A sports activity."
        exp = GradientExplainer(self.model_nli, self.tokenizer_nli)
        explanation = exp((premise, hypothesis), target="neutral")
        self.assertListEqual(
            explanation.tokens[:5], ["[CLS]", "▁A", "▁soccer", "▁game", "▁with"]
        )
        self.assertEqual(explanation.target_pos_idx, 1)

    def test_integrated_gradient_nli(self):
        premise = "A soccer game with multiple males playing."
        hypothesis = "A sports activity."
        exp = IntegratedGradientExplainer(self.model_nli, self.tokenizer_nli)
        explanation = exp((premise, hypothesis), target="contradiction")
        self.assertListEqual(
            explanation.tokens[:5], ["[CLS]", "▁A", "▁soccer", "▁game", "▁with"]
        )
        self.assertEqual(explanation.target_pos_idx, 2)
    
    def test_shap_zero_shot(self):
        bench = Benchmark(self.model_zero_shot, self.tokenizer_zero_shot, task_name="zero-shot-text-classification")
        sequence_to_classify = "A new Tesla model was unveiled."
        candidate_labels = ["technology", "economy", "sports"]
        exp = SHAPExplainer(self.model_zero_shot, self.tokenizer_zero_shot)
        scores = bench.score(sequence_to_classify, options=candidate_labels, return_probs=True)
        most_probable_label = max(scores, key=scores.get)
        explanation = exp(sequence_to_classify, target="entailment", target_option=most_probable_label)
        self.assertIn('Tesla', explanation.tokens)
        self.assertEqual(explanation.target, "entailment")

    def test_lime_zero_shot(self):
        bench = Benchmark(self.model_zero_shot, self.tokenizer_zero_shot, task_name="zero-shot-text-classification")
        sequence_to_classify = "A new Tesla model was unveiled."
        candidate_labels = ["technology", "economy", "sports"]
        exp = LIMEExplainer(self.model_zero_shot, self.tokenizer_zero_shot)
        scores = bench.score(sequence_to_classify, options=candidate_labels, return_probs=True)
        most_probable_label = max(scores, key=scores.get)
        explanation = exp(sequence_to_classify, target="entailment", target_option=most_probable_label, num_samples=30, show_progress=True)
        self.assertIn('Tesla', explanation.tokens)
        self.assertEqual(explanation.target, "entailment")
    
    def test_gradient_zero_shot(self):
        sequence_to_classify = "A new Tesla model was unveiled."
        candidate_labels = ["technology", "economy", "sports"]
        scores = self.bench_zero_shot.score(sequence_to_classify, options=candidate_labels, return_probs=True)
        most_probable_label = max(scores, key=scores.get)
        exp = GradientExplainer(self.model_zero_shot, self.tokenizer_zero_shot)
        explanation = exp([(sequence_to_classify, "This is " + str(most_probable_label))], target="entailment")
        self.assertIn('Tesla', explanation.tokens)
        self.assertEqual(explanation.target, "entailment")

    def test_integrated_gradient_zero_shot(self):
        sequence_to_classify = "A new Tesla model was unveiled."
        candidate_labels = ["technology", "economy", "sports"]
        scores = self.bench_zero_shot.score(sequence_to_classify, options=candidate_labels, return_probs=True)
        most_probable_label = max(scores, key=scores.get)
        exp = IntegratedGradientExplainer(self.model_zero_shot, self.tokenizer_zero_shot)
        explanation = exp([(sequence_to_classify, "This is " + str(most_probable_label))], target="entailment")
        self.assertIn('Tesla', explanation.tokens)
        self.assertEqual(explanation.target, "entailment")

    def test_shap_ner(self):
        text = "My name is John and I live in New York"
        exp = SHAPExplainer(self.model_ner, self.tokenizer_ner, task_name="ner")
        explanation = exp(text, target="I-LOC", target_token="York")
        self.assertTrue("York" in explanation.tokens)
        self.assertEqual(explanation.target_pos_idx, 6)

    def test_lime_ner(self):
        text = "My name is John and I live in New York"
        exp = LIMEExplainer(self.model_ner, self.tokenizer_ner, task_name="ner")
        exp.helper = TokenClassificationHelper(self.model_ner, self.tokenizer_ner)
        explanation = exp(text, target="I-LOC", target_token="York")
        self.assertTrue("York" in explanation.tokens)
        self.assertEqual(explanation.target_pos_idx, 6)
        
    def test_integrated_gradient_ner(self):
        text = "My name is John and I live in New York"
        exp = IntegratedGradientExplainer(self.model_ner, self.tokenizer_ner, multiply_by_inputs=True, task_name="ner")
        explanation = exp(text, target="I-LOC", target_token="York")
        self.assertTrue("york" in [token.lower() for token in explanation.tokens])
        self.assertEqual(explanation.target_pos_idx, 6)

    def test_gradient_ner(self):
        text = ["My name is John and I live in New York"]
        exp = GradientExplainer(self.model_ner, self.tokenizer_ner, multiply_by_inputs=True, task_name="ner")
        explanation = exp(text, target="I-LOC", target_token="York")
        self.assertTrue("york" in [token.lower() for token in explanation.tokens])
        self.assertEqual(explanation.target_pos_idx, 6)
