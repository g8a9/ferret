.. _notions.explainers:

************************************
Post-Hoc Feature Attribution Methods
************************************

Post-hoc feature attribution methods explain why a model made a specific prediction for a given text. 
These methods assign an importance score to each input. In the context of text data, we typically assign a score to each token, and so in ferret.
Given a model, a target class, and a prediction, ferret lets you measure how much each token contributed to that prediction.


.. _overview-explainers:

Overview of explainers
----------------------------

ferret integrates thee following post-hoc attribution methods:

- :ref:`LIME <explainers-lime>` 
- :ref:`SHAP <explainers-shap>` 
- :ref:`Gradient <explainers-gradient>`, plain gradients or multiplied by input token embeddings
- :ref:`Integrated Gradient <explainers-integratedgradient>`, plain gradients or multiplied by input token embeddings


.. _explainers-lime:

LIME
----------------------------
LIME (Local Interpretable Model-agnostic Explanations) is a model-agnostic method for explaining individual predictions by learning an interpretable model in the locality of the prediction.
The interpretable model is a local surrogate model that mimics the behavior of the original model locally. 

LIME learns an interpretable model only in the locality of the instance to derive the relevant feature for the individual label assignment. The approach derives the locality by generating perturbed samples of the instance, weighting the samples by their proximity. LIME optimizes the fidelity of the local surrogate model to the original one while preserving its understandability.

More details can be found in the `LIME paper <https://arxiv.org/abs/1602.04938>`_.


ferret uses the Captum implementation of `LIME <https://captum.ai/api/lime.html>`_.


.. _explainers-shap:

SHAP
----------------------------

SHAP (SHapley Additive exPlanations) is a game theoretic approach to explain individual predictions.

SHAP is based on the notion of Shapley values. The Shapley value is a concept from coalition game theory to assign a score to the players who cooperate to achieve a specific total score. In the context of prediction explanations, the attribute values of the instance to explain are the players, and the prediction probability is the score.
The exact estimation requires the computation of the omission effect for the power set of the attributes. Hence, multiple solutions have been proposed to overcome the problem of its exponential complexity.

SHAP's authors propose practical approximations for estimating Shapley values as KernelSHAP and PartitionSHAP. KernelSHAP is an approximation approach based on local surrogate models. The estimation is based on weighted linear regression models. 
More recently, the authors proposed PartitionSHAP, which uses hierarchical data clustering to define feature coalitions. The approach, as KernelSHAP, is model agnostic but makes the computation more tractable and typically requires less time.
More details can be found in the `SHAP paper <https://arxiv.org/abs/1602.04938>`_ and SHAP library `documentation  <https://shap.readthedocs.io/en/latest/index.html>`_.


ferret integrates `SHAP library implementation <https://github.com/slundberg/shap>`_ and PartitionSHAP as the default algorithm, which is also the default for textual data in the SHAP library.


.. _explainers-gradient:

Gradient (Saliency) and GradientXInput
----------------------------------------

Gradient approach, also known as Saliency, is one of the first gradient-based approaches. This class of approaches computes the gradient of the prediction score with respect to the input features and the methods differ on how the gradient is computed.
Gradient approach directly computes the gradient of the loss function for the target class with respect to the input.
More details can be found in the `corresponding paper <hhttps://arxiv.org/abs/1312.6034>`_.

The GradientXInput approach multiplies the gradient with respect to input with the input itself. More details can be found `here <https://arxiv.org/abs/1605.01713>`_.

ferret uses Captum implementations of `Gradient <https://captum.ai/api/saliency.html>`_  and `GradientXInput <https://captum.ai/api/input_x_gradient.html>`_.


.. _explainers-integratedgradient:

Integrated Gradients and Integrated Gradient X Input
-------------------------------------------------------

Integrated Gradients is a gradient-based approach. 
The approach considers a baseline input that consist in an informationless input. In the case of text, it could corresponds to an empty text or zero embedding vector.
The approach consider the straightline path from the baseline to the input, and compute the gradients along the path. 
Integrated gradients are obtained by cumulating these gradients. 

The method description can be found in the original `paper <https://arxiv.org/abs/1703.01365>`_.

ferret adopts the `Captum implementation <https://captum.ai/api/integrated_gradients.html>`_  and also includes the version multiplied for the input.