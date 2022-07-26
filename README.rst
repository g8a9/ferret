ferret
========

|pypi badge| |docs badge|

|banner|

.. |pypi badge| image:: https://img.shields.io/pypi/v/ferret-xai.svg
    :target: https://pypi.python.org/pypi/ferret-xai
    :alt: Latest PyPI version

.. |Docs Badge| image:: https://readthedocs.org/projects/ferret/badge/?version=latest
    :alt: Documentation Status
    :scale: 100%
    :target: https://ferret.readthedocs.io/en/latest/?version=latest

.. |banner| image:: ./images/banner.png
    :alt: Ferret circular logo with the name to the right
    :scale: 100%

A python package for benchmarking interpretability techniques.

* Free software: MIT license
* Documentation: https://ferret.readthedocs.io.

.. code-block:: python

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from ferret import Benchmark

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    bench = Benchmark(model, tokenizer)
    explanations = bench.explain("You look stunning!")
    evaluations = bench.evaluate_explanations(explanations)

    print(evaluations)


Features
--------

**ferret** builds on top of the transformers library. The library supports explanations using:

* Gradients
* Integrated Gradinets
* Gradient x Input word embeddings
* SHAP
* LIME

and evaluate explanations via:

Faithfulness measures.

* AOPC Comprehensiveness
* AOPC Sufficiency
* Kendall’s tau correlation with leave-one-feature out


Plausibility measures.

* AUPRC soft score plausibility
* Token f1 hard score plausibility
* Token IOU hard score plausibility

**TODOs**

* Possibility to run on select device ("cpu", "cuda")
* Sample-And-Occlusion explanations
* Discretized Integrated Gradients: https://arxiv.org/abs/2108.13654

Visualization
-------------

.. code-block:: python

    bench = Benchmark(...)

    explanations = ...
    bench.show_table(explanations)

    evaluations = bench.evaluate_explanations(explanations)
    bench.show_evaluation_table(evaluations)


Datasets evaluations
--------------------

.. code-block:: python

    bench = Benchmark(...)

    hatexdata = bench.load_dataset("hatexplain")
    dataset_explanations = bench.generate_dataset_explanations(hatexdata)
    dataset_evaluations = bench.evaluate_dataset_explanations(dataset_explanations)
    bench.show_dataset_evaluation_table(dataset_evaluations)


Credits
-------

This package was created with Cookiecutter and the *audreyr/cookiecutter-pypackage* project template.

- Cookiecutter: https://github.com/audreyr/cookiecutter
- `audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

Logo and graphical assets made by `Luca Attanasio <https://www.behance.net/attanasiol624d>`_.
