[![Latest PyPI version](https://img.shields.io/pypi/v/ferret-xai.svg)](https://pypi.python.org/pypi/ferret-xai)
[![Documentation Status](https://readthedocs.org/projects/ferret/badge/?version=latest)](https://ferret.readthedocs.io/en/latest/?version=latest)
[![HuggingFace Spaces Demo](https://img.shields.io/badge/HF%20Spaces-Demo-yellow)](https://huggingface.co/spaces/g8a9/ferret)
[![YouTube Video](https://img.shields.io/badge/youtube-video-red)](https://www.youtube.com/watch?v=kX0HcSah_M4)
[![arxiv preprint](https://img.shields.io/badge/arXiv-2208.01575-b31b1b.svg)](https://arxiv.org/abs/2208.01575)
[![downloads badge](https://pepy.tech/badge/ferret-xai/month)](https://pepy.tech/project/ferret-xai)

![Ferret circular logo with the name to the right](/docs/source/_static/banner.png)

ferret is Python library that streamlines the use and benchmarking of interpretability techniques on Transformers models.

- Documentation: https://ferret.readthedocs.io
- Paper: https://arxiv.org/abs/2208.01575
- Demo: https://huggingface.co/spaces/g8a9/ferret

ferret is meant to integrate seamlessly with 🤗 transformers models, among which it currently supports text models only.
We provide:
- 🔍 Four established token-level feature attribution explainability techniques. Use these to get quickly token-level importance.
- ⚖️ Six Faithfulness and Plausibility evaluation protocols. Benchmark any token-level explanation against these tests to guide your choice toward the most reliable explainer.  

**📝 Examples**

Text Classification

- Sentiment Classification on Twitter Data: [Colab](https://colab.research.google.com/github/g8a9/ferret/blob/main/examples/benchmark.ipynb)
- Intent Detection with Multilingual XLM RoBERTa: [Colab](https://colab.research.google.com/drive/17AXeA9-u7lOLlE_DWtUixMg7Mi0NFPIp?usp=sharing)


## Getting Started

### Installation

```bash
pip install -U ferret-xai
```

Our main dependencies are 🤗 `tranformers` and `datasets`.

### Explain & Benchmark 

The code below provides a minimal example to run all the feature-attribution explainers supported by ferret and benchmark them on faithfulness metrics.

We start from a common text classification pipeline

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from ferret import Benchmark

name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(name)
tokenizer = AutoTokenizer.from_pretrained(name)
```

Using *ferret* is as simple as: 

```python
bench = Benchmark(model, tokenizer)
explanations = bench.explain("You look stunning!", target=1)
evaluations = bench.evaluate_explanations(explanations, target=1)

bench.show_evaluation_table(evaluations)
```

Be sure to run the code in a Jupyter Notebook/Colab: the cell above will produce a nicely-formatted table to analyze the saliency maps.

## Features

**ferret** offers a *painless* integration with Hugging Face models and
naming conventions. If you are already using the
[transformers](https://github.com/huggingface/transformers) library, you
immediately get access to our **Explanation and Evaluation API**.

### Post-Hoc Explainers

- Gradient (plain gradients or multiplied by input token embeddings) ([Simonyan et al., 2014](https://arxiv.org/abs/1312.6034))
- Integrated Gradient (plain gradients or multiplied by input token embeddings) ([Sundararajan et al., 2017](http://proceedings.mlr.press/v70/sundararajan17a.html))
- SHAP (via Partition SHAP approximation of Shapley values) ([Lundberg and Lee, 2017](https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html))
- LIME ([Ribeiro et al., 2016](https://dl.acm.org/doi/abs/10.1145/2939672.2939778))

### Evaluation Metrics

**Faithfulness** measures:

- AOPC Comprehensiveness ([DeYoung et al.,
    2020](https://doi.org/10.18653/v1/2020.acl-main.408))
- AOPC Sufficiency ([DeYoung et al.,
    2020](https://doi.org/10.18653/v1/2020.acl-main.408))
- Kendall's Tau correlation with Leave-One-Out token removal. ([Jain
    and Wallace, 2019](https://aclanthology.org/N19-1357/))

**Plausibility** measures:

- Area-Under-Precision-Recall-Curve (soft score) ([DeYoung et al., 2020](https://doi.org/10.18653/v1/2020.acl-main.408))
- Token F1 (hard score) ([DeYoung et al., 2020](https://doi.org/10.18653/v1/2020.acl-main.408))
- Token Intersection Over Union (hard score) ([DeYoung et al., 2020](https://doi.org/10.18653/v1/2020.acl-main.408))

See our [paper](https://arxiv.org/abs/2208.01575) for details.

## Visualization

The `Benchmark` class exposes easy-to-use table
visualization methods (e.g., within Jupyter Notebooks)

```python
bench = Benchmark(model, tokenizer)

# Pretty-print feature attribution scores by all supported explainers
explanations = bench.explain("You look stunning!")
bench.show_table(explanations)

# Pretty-print all the supported evaluation metrics
evaluations = bench.evaluate_explanations(explanations)
bench.show_evaluation_table(evaluations)
```

## Dataset Evaluations

The `Benchmark` class has a handy method to compute and
average our evaluation metrics across multiple samples from a dataset.

```python
import numpy as np
bench = Benchmark(model, tokenizer)

# Compute and average evaluation scores one of the supported dataset
samples = np.arange(20)
hatexdata = bench.load_dataset("hatexplain")
sample_evaluations =  bench.evaluate_samples(hatexdata, samples)

# Pretty-print the results
bench.show_samples_evaluation_table(sample_evaluations)
```

## Planned Developement

See [the changelog file](https://github.com/g8a9/ferret/blob/main/HISTORY.rst) for further
details.

- ✅ GPU acceleartion support for inference (**v0.4.0**)
- ✅ Batched Inference for internal methods\'s approximation steps (e.g., LIME or SHAP) (**v0.4.0**)
- ⚙️ Simplified Task API to support NLI, Zero-Shot Text Classification, Language Modeling ([branch](https://github.com/g8a9/ferret/tree/task-API)).
- ⚙️ Multi-sample explanation generation and evaluation
- ⚙️ Support to explainers for seq2seq and autoregressive generation through [inseq](https://github.com/inseq-team/inseq).
- ⚙️ New evaluation measure: Sensitivity, Stability ([Yin et al.](https://aclanthology.org/2022.acl-long.188/))
- ⚙️ New evaluation measure: Area Under the Threshold-Performance Curve (AUC-TP) ([Atanasova et al.](https://aclanthology.org/2020.emnlp-main.263/))
- ⚙️ New explainer: Sampling and Occlusion (SOC) ([Jin et al., 2020](https://arxiv.org/abs/1911.06194))
- ⚙️ New explainer: Discretized Integrated Gradient (DIG) ([Sanyal and Ren, 2021](https://aclanthology.org/2021.emnlp-main.805/))
- ⚙️ Support additional form of aggregation over embeddings' hidden dimension.


## Authors

- [Giuseppe Attanasio](https://gattanasio.cc)
- [Eliana Pastor](mailto:eliana.pastor@centai.eu)
- [Debora Nozza](https://deboranozza.com/)
- Chiara Di Bonaventura

## Credits

This package was created with Cookiecutter and the
*audreyr/cookiecutter-pypackage* project template.

- Cookiecutter: https://github.com/audreyr/cookiecutter
- [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage)

Logo and graphical assets made by [Luca Attanasio](https://www.behance.net/attanasiol624d).

If you are using *ferret* for your work, please consider citing us!

```bibtex
@article{attanasio2022ferret,
  title={ferret: a Framework for Benchmarking Explainers on Transformers},
  author={Attanasio, Giuseppe and Pastor, Eliana and Di Bonaventura, Chiara and Nozza, Debora},
  journal={arXiv preprint arXiv:2208.01575},
  year={2022}
}
```
