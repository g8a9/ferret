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
    from nlxplain import Explainer

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    exp = Explainer(model, tokenizer)
    exp.compute_table("You look stunning!")


Features
--------

**Nlxplain** (read "nellxplain") builds on top of the transformers library. The library supports explanations using:

* Gradients
* Integrated Gradinets
* Gradient x Input word embeddings
* SHAP (partition approximation)
* LIME

Feature we expect to include:

* Possibility to run on select device ("cpu", "cuda")
* Sample-And-Occlusion explanations

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

- Cookiecutter: https://github.com/audreyr/cookiecutter
- `audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
