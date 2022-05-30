# nlxplain
        
[![pypi](https://img.shields.io/pypi/v/nlxplain.svg)](https://pypi.python.org/pypi/nlxplain)
[![travis](https://img.shields.io/travis/g8a9/nlxplain.svg)](https://travis-ci.com/g8a9/nlxplain)
[![docs](https://readthedocs.org/projects/nlxplain/badge/?version=latest)](https://nlxplain.readthedocs.io/en/latest/?version=latest)

A python package for NLP explainability.

* Free software: MIT license
* Documentation: https://nlxplain.readthedocs.io.

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from nlxplain import Explainer

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

exp = Explainer(model, tokenizer)
exp.compute_table("You look stunning!")
```

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

- Cookiecutter: https://github.com/audreyr/cookiecutter
- `audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
