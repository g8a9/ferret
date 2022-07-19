nlxplain
========

|pypi badge| |docs badge|

.. |pypi badge| image:: https://img.shields.io/pypi/v/nlxplain.svg
    :target: https://pypi.python.org/pypi/nlxplain
    :alt: Latest PyPI version

.. |Docs Badge| image:: https://readthedocs.org/projects/nlxplain/badge/?version=latest
    :alt: Documentation Status
    :scale: 100%
    :target: https://nlxplain.readthedocs.io/en/latest/?version=latest

A python package for NLP explainability.

* Free software: MIT license
* Documentation: https://nlxplain.readthedocs.io.

.. code-block:: python

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from nlxplain import Benchmark

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    bench = Benchmark(model, tokenizer)
    explanations = bench.explain("You look stunning!")
    bench.show_table(explanations)


Features
--------

**Nlxplain** (read "nellxplain") builds on top of the transformers library. The library supports explanations using:

* Gradients
* Integrated Gradinets
* Gradient x Input word embeddings
* SHAP
* LIME


**TODOs**

* Possibility to run on select device ("cpu", "cuda")
* Sample-And-Occlusion explanations
* Discretized Integrated Gradients: https://arxiv.org/abs/2108.13654

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

- Cookiecutter: https://github.com/audreyr/cookiecutter
- `audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
