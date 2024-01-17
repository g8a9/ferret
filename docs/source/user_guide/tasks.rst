=====================
Tasks Documentation
=====================

This document provides a comprehensive guide to the tasks available in the Ferret project. Each task is detailed with its purpose, usage, and associated parameters.

.. contents::
   :local:
   :depth: 2

Sequence Classification
=======================

.. _sequence-classification:

Introduction
------------
Sequence Classification is a task that involves categorizing text sequences into predefined labels or classes. This task is commonly used for sentiment analysis, topic labeling, and similar applications where text needs to be classified according to its content or sentiment.

Usage
-----
.. code-block:: python


    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from ferret import Benchmark
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment")
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment")
    bench = Benchmark(model, tokenizer)
    text = "You look stunning!"
    exp = bench.explain(text, target=1)
    bench.show_table(exp)
    # 'explanation' contains SHAP values for each token in the text.

Natural Language Inference (NLI)
=================================

.. _natural-language-inference:

Introduction
------------
Natural Language Inference focuses on determining the relationship between a premise and a hypothesis, categorizing the relationship as entailment, contradiction, or neutral. 

Usage
-----
.. code-block:: python


    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from ferret import Benchmark
    model = AutoModelForSequenceClassification.from_pretrained("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
    tokenizer = AutoTokenizer.from_pretrained("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
    premise = "A soccer game with multiple males playing."
    hypothesis = "A sports activity."
    sample = (premise, hypothesis)
    bench = Benchmark(model, tokenizer, task_name="nli")
    exp = bench.explain(sample, target="contradiction")
    bench.show_table(exp)

Zero-Shot Classification
========================

.. _zero-shot-classification:

Introduction
------------
Zero-Shot Classification refers to classifying text into categories that were not seen during training. It's used for tasks where predefined categories are not available.

Usage
-----
.. code-block:: python


    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from ferret import Benchmark
    
    tokenizer = AutoTokenizer.from_pretrained("MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
    model = AutoModelForSequenceClassification.from_pretrained("MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
    sequence_to_classify = "A new Tesla model was unveiled."
    candidate_labels = ["technology", "economy", "sports"]
    bench = Benchmark(model, tokenizer, task_name="zero-shot-text-classification")
    scores = bench.score(sequence_to_classify, options=candidate_labels, return_probs=True)
    # get the label with the highest score, and use it as 'target_option'
    most_probable_label = max(scores, key=scores.get)
    exp = bench.explain(sequence_to_classify, target="entailment", target_option=most_probable_label)
    # 'explanation' shows how the model associates the text with the categories.

Named Entity Recognition (NER)
==============================

.. _named-entity-recognition:

Introduction
------------
Named Entity Recognition involves identifying and categorizing key information (entities) in text, such as names of people, places, organizations, etc.

Usage
-----
.. code-block:: python


    from transformers import AutoModelForTokenClassification, AutoTokenizer
    from ferret import Benchmark
    tokenizer = AutoTokenizer.from_pretrained("Babelscape/wikineural-multilingual-ner")
    model = AutoModelForTokenClassification.from_pretrained("Babelscape/wikineural-multilingual-ner")
    text = "My name is John and I live in New York"
    bench = Benchmark(model, tokenizer, task_name="ner")
    exp = bench.explain(text, target="I-LOC", target_token="York")
    bench.show_table(exp)
.. note::
   The usage examples provided in this document are intended to guide users through the various tasks. For detailed explanations of the different explainers, please refer to the respective documentation files.