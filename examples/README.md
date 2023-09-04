# Task Support Matrix

*ferret* integrates seamlessly with a wide range of tasks. Please refer to the matrix below
to see which task we currently support off-the-shelf (note: **ferret-xai >= 0.5.0 is required**).


| Task (`HF Class`) | G | IG | SHAP | LIME | Tutorial |
|-------------------------------|:-:|:--:|:----:|:----:|----------|
| Sequence Classification (`AutoModelForSequenceClassification`)     | ✅ |  ✅ |   ✅  |   ✅  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/g8a9/ferret/blob/task-API/examples/sentiment_classification.ipynb) |
| Natural Language Inference (`AutoModelForSequenceClassification`)   | ✅ |  ✅ |   ⚙️  |   ✅  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/g8a9/ferret/blob/task-API/examples/nli.ipynb) |
| Zero-Shot Text Classification (`AutoModelForSequenceClassification`) | ✅ |  ✅ |   ⚙️  |   ✅  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/g8a9/ferret/blob/task-API/examples/zeroshot_text_classification.ipynb) |
| Named Entity Recognition (`AutoModelForTokenClassification`)  | ✅  |  ✅  |  ✅️  |  ✅  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/g8a9/ferret/blob/task-API/examples/ner.ipynb) |
| _Multiple Choice_    |   |    |      |      |          |
| _Masked Language Modeling_    |   |    |      |      |          |
| _Casual Language Modeling_    |   |    |      |      |          |

Where:
- ✅: we got you covered!
- ⚙️: working on it...
