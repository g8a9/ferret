.. _quickstart:

**********
Quickstart
**********

Here is a code snipped to show **ferret** integrated with your existing **transformers** models.

.. code-block:: python

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from ferret import Benchmark

    name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    model = AutoModelForSequenceClassification.from_pretrained(name)
    tokenizer = AutoTokenizer.from_pretrained(name)

    bench = Benchmark(model, tokenizer)
    explanations = bench.explain("You look stunning!", target=1)
    evaluations = bench.evaluate_explanations(explanations, target=1)

    bench.show_evaluation_table(evaluations)

