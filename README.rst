========
nlxplain
========


.. image:: https://img.shields.io/pypi/v/nlxplain.svg
        :target: https://pypi.python.org/pypi/nlxplain

.. image:: https://img.shields.io/travis/g8a9/nlxplain.svg
        :target: https://travis-ci.com/g8a9/nlxplain

.. image:: https://readthedocs.org/projects/nlxplain/badge/?version=latest
        :target: https://nlxplain.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status



A python package for NLP explainability.



* Free software: MIT license
* Documentation: https://nlxplain.readthedocs.io.

Usage
^^^
.. code-block:: python
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from nlxplain import Explainer

        model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased")
        tokenizer = AutoTokenizer.from_pretrainedb("bert-base-cased")

        exp = Explainer(model, tokenizer)

        exp.compute_table("You look stunning!")



Features
--------

Nlxplain (read "nellxplain") builds on top of the transformers library. The library supports explanations using:
* Gradients
* Integrated Gradinets
* Gradient x Input word embeddings
* SHAP (partition approximation)

Feature we expect to include:
* GPU based inference
* TBD

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
