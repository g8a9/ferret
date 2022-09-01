=====
Usage
=====

Inference on GPU
^^^^^^^^^^^^^^^^

Starting from version 0.4.0, ferret supports inference on GPU.
In practice, ferret will use the device of the `model` with no changes of explicit calls.

Assuming that your model is currently on CPU, to run explanations on GPU you just need to move it before:

.. code-block:: python

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from ferret import LIMEExplainer

    name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    m = AutoModelForSequenceClassification.from_pretrained(name).to("cuda:0")
    t = AutoTokenizer.from_pretrained(name)

    exp = LIMEExplainer(model, tokenizer)
    explanation = expl("You look stunning!", target=1)

Batched Inference
^^^^^^^^^^^^^^^^^

Some explainers (e.g., LIME or IntegratedGradients) require to run inference on a large number
of data points, which might be computationally unfeasible for transformer models.

Since verson 0.4.0, ferret supports automatically batched inference (on both CPU and GPU).
When the batched inference is available, you can specify both `batch_size` and
`show_progress`.

.. code-block:: python

    exp = LIMEExplainer(m, t)
    call_args={"num_samples": 5000, "show_progress": True, "batch_size": 16}
    explanation = expl("You look stunning!", call_args=call_args)