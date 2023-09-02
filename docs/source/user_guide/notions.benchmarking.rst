.. _notions.benchmarking:

***********************
Evaluating Explanations
***********************

Benchmarking Metrics
=======================

We evaluate explanations on the faithfulness and plausibility properties. Specifically, *ferret* implements three state-of-the-art metrics to measure faithfulness and three for plausibility [1]_ [2]_.

.. [1] Towards Faithfully Interpretable NLP Systems: How Should We Define and Evaluate Faithfulness? (Jacovi & Goldberg, ACL 2020)
.. [2] ERASER: A Benchmark to Evaluate Rationalized NLP Models (DeYoung et al., ACL 2020)


.. _explanations-type:

Type of explanations
=======================

Before describing the faithfulness and plausibility metrics, we first define the types of explanations we handle: continuous score explanations, discrete explanations and human rationale.

.. glossary::

    Continuous score explanations 
      Continuous score explanations assign a continuous score to each token. All the post-hoc feature attribution methods of ferret generate continuous score explanations.
      Continuous score explanations are also called soft scores or continuous token attribution scores.

    Discrete explanations
      Discrete explanations or rationale indicates the set of tokens supporting the prediction. 

    Human rationales
      Human rationales are annotations highlighting the most relevant words (phrases, or sentences) a human annotator attributed to a given class label. 
      Typically, human rationales are discrete explanations, indicating the set of words relevant for a human.


=======================
.. _faithfulness-overview:

Faithfulness measures
=======================
Faithfulness evaluates how accurately the explanation reflects the inner working of the model (Jacovi and Goldberg, 2020).

ferret offers the following measures of faithfulness:

- :ref:`AOPC Comprehensiveness <faithfulness-aopc_compr>` - (aopc_compr, ↑) -  goes from 0 to 1 (best)
- :ref:`AOPC Sufficiency <faithfulness-aopc_suff>` - (aopc_suff, ↓)) -  goes from 0 (best) to 1;
- :ref:`Kendall's Tau correlation with Leave-One-Out token removal <faithfulness-taucorr_loo>` - (taucorr_loo, ↑) - goes from -1 to 1 (best).



.. _faithfulness-aopc_compr:

AOPC Comprehensiveness
---------------------------


Comprehensiveness evaluates whether the explanation captures the tokens the model used to make the prediction. 

Given a set of relevant token that defines a discrete explanation, comprehensiveness measures the drop in the model probability if the relevant tokens are removed.
A high value of comprehensiveness indicates that the tokens in rj are relevant for the prediction.


More formally, let :math:`x` be a sentence and let :math:`f_j` be the prediction probability of the model :math:`f` for a target class :math:`j`.  
Let :math:`r_j` be a discrete explanation indicating the set of tokens supporting the prediction :math:`f_j`.
Comprehensiveness is defined as 

.. math::
    \textnormal{comprehensiveness} = f_j(x)−f_j(x \setminus r_j )

where :math:`x \setminus r_j` is the sentence :math:`x` were tokens in :math:`r_j` are removed. 

The higher the value, the more the explainer is able to select the relevant tokens for the prediction.

While comprehensiveness is defined for discrete explanations, ferret explanations assign a continuous score to each token. The selection of the most important tokens from continuous score explanations impact the results.
Hence, following (DeYoung et al., 2020), we measure comprehensiveness via the Area Over the Perturbation Curve.
First, we filter out tokens with a negative contribution (i.e., they pull the prediction away from the chosen label). 
Then, we progressively consider th *k* most important tokens, with k ranging from 10% to 100% (step of 10%). Finally, we average the result. 

See `DeYoung et al. (2020) <https://aclanthology.org/2020.acl-main.408.pdf>`_ for its detailed definition.


.. _faithfulness-aopc_suff:

AOPC Sufficiency
---------------------------

Sufficiency captures if the tokens in the explanation are sufficient for the model to make the prediction. 


Let :math:`x` be a sentence and let :math:`f_j` be the prediction probability of the model :math:`f` for a target class :math:`j`.  
Let :math:`r_j` be a discrete explanation indicating the set of tokens supporting the prediction :math:`f_j`.
Sufficiency is defined as 

.. math::
    \textnormal{sufficiency} = f_j(x)− f_j(r_j)

where :math:`r_j` is the sentence :math:`x` were only tokens in :math:`r_j` are considered. 

It goes from 0 (best) to 1.
A low score indicates that tokens in the discrete explanation in :math:`r_j` are indeed the ones driving the prediction. 


As for comprehensiveness, we compute the Area Over the Perturbation Curve by varying the number of the relevant tokens :math:`r_j`.
Specifically, we first filter out tokens with a negative contribution for the chosen target class.
Then, we compute sufficiency varying the *k* most important tokens in :math:`r_j` (as default for 10% to 100% with step 10) and we average the result. 


See `DeYoung et al. (2020) <https://aclanthology.org/2020.acl-main.408.pdf>`_ for its detailed definition.


.. _faithfulness-taucorr_loo:

Correlation with Leave-One-Out scores
------------------------------------------------------
The correlation with Leave-One-Out (taucorr_loo) measures the correlation between the explanation and a baseline explanation referred to as leave-one-out scores.
The leave-one-out (LOO) scores are the prediction difference when one feature at the time is omitted. 
The taucorr_loo measure the Spearman correlation between the explanation and the leave-one-out scores.

It goes from -1 to 1; a value closer to 1 means higher faithfulness to LOO.


See `Jain and Wallace, (2019) <https://aclanthology.org/N19-1357/>`_ for its detailed definition.
 


.. _plausibility:

Plausibility measures
=======================

Plausibility evaluates how well the explanation agree with human rationale. 


- Token Intersection Over Union (hard score)  - (token_iou_plau) 
- Token F1 (hard score) - (token_f1_plau) 
- Area-Under-Precision-Recall-Curve - (auprc_plau) 


.. _plausibility-token_iou_plau:

Intersection-Over-Union (IOU)
------------------------------------------------------

Given a human rationale and a discrete explanation, the Intersection-Over-Union (IOU) is the size of the overlap of the tokens they cover divided by the size of their union.

We derive the discrete rationale from continuous score explanations by taking the top-K values with positive contribution.

When the set of human rationales for the dataset is available, K is set as the average rationale length (as in ERASER).
Otherwise, K is set as default to 5.

See `DeYoung et al. (2020) <https://aclanthology.org/2020.acl-main.408.pdf>`_ for its detailed definition.


.. _plausibility-token_f1_plau:

Token-level f1-score
------------------------------------------------------

Token-level F1 scores (↑) is the F1 score computed from the precision and recall at the token level considering the human rationale as ground truth explanation and the discrete explanation as the predicted one.

As for the IOU, we derive the discrete rationale from explanations by taking the top-K values with positive contribution.

When the set of human rationales for the dataset is available, K is set as the average rationale length.
Otherwise, K is set as default to 5.

See `DeYoung et al. (2020) <https://aclanthology.org/2020.acl-main.408.pdf>`_ for its detailed definition.


.. _plausibility-auprc_plau:

Area Under the Precision Recall curve (AUPRC)
------------------------------------------------------
Area Under the Precision Recall curve (AUPRC) is computed by varying a threshold over token importance scores, using the human rationale as ground truth.

The advantage of AUPRC with respect to the IOU and Token-level F1 is that it directly consider continuous score explanation. 
Hence, it takes into account  tokens’ relative ranking and degree of importance.

See `DeYoung et al. (2020) <https://aclanthology.org/2020.acl-main.408.pdf>`_ for its detailed definition.
