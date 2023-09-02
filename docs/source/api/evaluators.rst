.. _api.evaluators:

==========
Evaluators
==========

.. currentmodule:: ferret

Abstract Classes
----------------

.. autosummary::
    :toctree: api/

    BaseEvaluator


Evaluation Methods
------------------

.. autosummary::
    :toctree: api/
    :caption: Faithfulness

    AOPC_Comprehensiveness_Evaluation
    AOPC_Sufficiency_Evaluation
    TauLOO_Evaluation

.. autosummary::
    :toctree: api/
    :caption: Plausibility

    AUPRC_PlausibilityEvaluation
    Tokenf1_PlausibilityEvaluation
    TokenIOU_PlausibilityEvaluation