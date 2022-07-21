import numpy as np


def _get_id_tokens_greater_th(soft_score_explanation, th, only_pos=None):
    id_top = np.where(soft_score_explanation > th)[0]
    return id_top


def _get_id_tokens_top_k(soft_score_explanation, k, only_pos=True):
    if only_pos:
        id_top_k = [
            i
            for i in np.array(soft_score_explanation).argsort()[-k:][::-1]
            if soft_score_explanation[i] > 0
        ]
    else:
        id_top_k = np.array(soft_score_explanation).argsort()[-k:][::-1]
    # if len(id_top_k) != k:
    #    return None
    return id_top_k


def _get_id_tokens_percentage(soft_score_explanation, percentage, only_pos=True):
    v = int(percentage * len(soft_score_explanation))
    # Only if we remove at least instance. TBD
    if v > 0 and v <= len(soft_score_explanation):
        return _get_id_tokens_top_k(soft_score_explanation, v, only_pos=only_pos)
    else:
        return None


def get_discrete_explanation_topK(score_explanation, topK, only_pos=False):

    # Indexes in the top k. If only pos is true, we only consider scores>0
    topk_indices = _get_id_tokens_top_k(score_explanation, topK, only_pos=only_pos)

    # topk_score_explanations: one hot encoding: 1 if the token is in the rationale, 0 otherwise
    # i hate you [0, 1, 1]

    topk_score_explanations = [
        1 if i in topk_indices else 0 for i in range(len(score_explanation))
    ]
    return topk_score_explanations


def _check_and_define_get_id_discrete_rationale_function(based_on):
    if based_on == "th":
        get_discrete_rationale_function = _get_id_tokens_greater_th
    elif based_on == "k":
        get_discrete_rationale_function = _get_id_tokens_top_k
    elif based_on == "perc":
        get_discrete_rationale_function = _get_id_tokens_percentage
    else:
        raise ValueError(f"{based_on} type not supported. Specify th, k or perc.")
    return get_discrete_rationale_function
