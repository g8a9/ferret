ferret
========

|pypi badge| |docs badge| |demo badge| |youtube badge|

|banner|

.. |pypi badge| image:: https://img.shields.io/pypi/v/ferret-xai.svg
    :target: https://pypi.python.org/pypi/ferret-xai
    :alt: Latest PyPI version

.. |Docs Badge| image:: https://readthedocs.org/projects/ferret/badge/?version=latest
    :alt: Documentation Status
    :scale: 100%
    :target: https://ferret.readthedocs.io/en/latest/?version=latest

.. |demo badge| image:: https://img.shields.io/badge/HF%20Spaces-Demo-yellow
    :alt: HuggingFace Spaces Demo 
    :scale: 100%
    :target: https://huggingface.co/spaces/g8a9/ferret

.. |youtube badge| image:: https://img.shields.io/badge/youtube-video-red
    :alt: YouTube Video
    :scale: 100%
    :target: 

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
    sample_evaluations =  bench.evaluate_samples(hatexdata, [1,2])
    bench.show_samples_evaluation_table(sample_evaluations)


Credits
-------

This package was created with Cookiecutter and the *audreyr/cookiecutter-pypackage* project template.

- Cookiecutter: https://github.com/audreyr/cookiecutter
- `audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

Logo and graphical assets made by `Luca Attanasio <https://www.behance.net/attanasiol624d>`_.
