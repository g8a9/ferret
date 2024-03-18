import numpy as np
from pydub import AudioSegment
import warnings
from typing import List
from IPython.display import display
from . import SpeechBaseEvaluator, EvaluationMetricFamily
from ..evaluators.utils_from_soft_to_discrete import (
    parse_evaluator_args,
    _check_and_define_get_id_discrete_rationale_function,
)
from ..evaluators.faithfulness_measures import _compute_aopc
from ..explainers.explanation_speech.explanation_speech import (
    ExplanationSpeech, EvaluationSpeech)
from ..explainers.explanation_speech.utils_removal import remove_specified_words
from ..speechxai_utils import pydub_to_np


class AOPC_Comprehensiveness_Evaluation_Speech(SpeechBaseEvaluator):
    NAME = "aopc_comprehensiveness_speech"
    SHORT_NAME = "aopc_compr_speech"
    LOWER_IS_BETTER = False  # Higher is better
    METRIC_FAMILY = EvaluationMetricFamily.FAITHFULNESS

    def compute_evaluation(
        self,
        explanation: ExplanationSpeech,
        target=None,
        words_trascript: List = None,
        **evaluation_args,
    ) -> EvaluationSpeech:
        """Evaluate an explanation on the AOPC Comprehensiveness metric.

        Args:
            explanation (ExplanationSpeech): the explanation to evaluate
            target: class labels for which the explanation is evaluated - deprecated
            evaluation_args (dict): arguments for the evaluation

        Returns:
            Evaluation : the AOPC Comprehensiveness score of the explanation
        """

        _, only_pos, removal_args, _ = parse_evaluator_args(evaluation_args)

        assert (
            "perturb_paraling" not in explanation.explainer
        ), f"{explanation.explainer} not supported"

        if "+" in explanation.explainer:
            # 'The explainer name contain "+" to specify the removal type'
            removal_type = explanation.explainer.split("+")[1]
        else:
            # Default
            removal_type = "silence"

        if target is not None:
            warnings.warn(
                'The "target" argument is deprecated and will be removed in a future version. The explanation target are used as default.'
            )

        target = explanation.target

        # Get audio as array.
        audio_np = explanation.audio.array

        # Get prediction probability of the input sencence for the target
        ground_truth_probs = self.model_helper.predict([audio_np])

        # Get the probability of the target classes
        if self.model_helper.n_labels > 1:
            # Multi-label setting - probability of the target classes for each label (list of size = number of labels)
            ground_truth_probs_target = [
                ground_truth_probs[e][:, tidx][0] for e, tidx in enumerate(target)
            ]
        else:
            # Single probability
            ground_truth_probs_target = [ground_truth_probs[0][target[0]]]

        # TODO: modify to accept a `FerretAudio` object as input.
        # Split the audio into word-level audio segments
        from ..speechxai_utils import transcribe_audio

        if words_trascript is None:
            words_trascript = explanation.audio.transcription

        get_discrete_rationale_function = (
            _check_and_define_get_id_discrete_rationale_function(
                removal_args["based_on"]
            )
        )

        thresholds = removal_args["thresholds"]

        score_explanation = explanation.scores

        # In this way, we allow for multi-label explanations
        # We have a list of score_explanations, one for each label
        # Each score explanation has length equal to the number of features: number of words in the case of word-level explanations or 1 in the case of paralinguistic level explanation (one paralinguistic feature at the time)
        if score_explanation.ndim == 1:
            score_explanations = [score_explanation]
        else:
            score_explanations = score_explanation

        aopc_comprehesiveness_multi_label = list()

        # We iterate over the target classes for a multi-label setting
        # In the case of single label, we iterate only once
        for target_class_idx, score_explanation in enumerate(score_explanations):
            removal_importances = list()
            id_tops = list()
            last_id_top = None

            # Ground truth probabilities of the target label (target_class_idx) and target class (target[target_class_idx]])
            # It is the output probability of the target class itself in the case of single label
            original_prob = ground_truth_probs_target[target_class_idx]

            # We compute the difference between in probability for all the thresholds
            for v in thresholds:
                # Get rationale from score explanation
                id_top = get_discrete_rationale_function(score_explanation, v, only_pos)

                # If the rationale is the same, we do not include it. In this way, we will not consider in the average the same omission.
                if (
                    id_top is not None
                    and last_id_top is not None
                    and set(id_top) == last_id_top
                ):
                    id_top = None

                id_tops.append(id_top)

                if id_top is None:
                    continue

                last_id_top = set(id_top)

                id_top.sort()

                # Comprehensiveness
                # The only difference between comprehesivenss and sufficiency is the computation of the removal.

                # For the comprehensiveness: we remove the terms in the discrete rationale.

                words_removed = [words_trascript[i] for i in id_top]

                audio_removed = remove_specified_words(
                    explanation.audio.to_pydub(),
                    words_removed,
                    removal_type=removal_type
                )

                audio_removed_np = pydub_to_np(audio_removed)[0]

                # Probability of the modified audio
                audio_modified_probs = self.model_helper.predict([audio_removed_np])

                # Probability of the target class (and label) for the modified audio
                if self.model_helper.n_labels > 1:
                    # In the multi-label setting, we have a list of probabilities for each label

                    # We first take the probability of the corresponding target label target_class_idx
                    # Then we take the probability of the target class for that label target[target_class_idx]
                    modified_prob = audio_modified_probs[target_class_idx][
                        :, target[target_class_idx]
                    ][0]

                else:
                    # Single probability
                    # We take the probability of the target class target[target_class_idx]
                    modified_prob = audio_modified_probs[0][target[target_class_idx]]

                # compute probability difference
                removal_importance = original_prob - modified_prob
                removal_importances.append(removal_importance)

            if removal_importances == []:
                return EvaluationSpeech(self.SHORT_NAME, 0, target)

            #  compute AOPC comprehensiveness
            aopc_comprehesiveness = _compute_aopc(removal_importances)
            aopc_comprehesiveness_multi_label.append(aopc_comprehesiveness)

        evaluation_output = EvaluationSpeech(
            self.SHORT_NAME, aopc_comprehesiveness_multi_label, target
        )

        return evaluation_output


class AOPC_Sufficiency_Evaluation_Speech(SpeechBaseEvaluator):
    NAME = "aopc_sufficiency"
    SHORT_NAME = "aopc_suff"
    LOWER_IS_BETTER = True
    METRIC_FAMILY = EvaluationMetricFamily.FAITHFULNESS

    def compute_evaluation(
        self,
        explanation: ExplanationSpeech,
        target: List = None,
        words_trascript: List = None,
        **evaluation_args,
    ) -> EvaluationSpeech:
        """Evaluate an explanation on the AOPC Sufficiency metric.

        Args:
            explanation (ExplanationSpeech): the explanation to evaluate
            target: class labels for which the explanation is evaluated - deprecated
            evaluation_args (dict): arguments for the evaluation

        Returns:
            Evaluation : the AOPC Sufficiency score of the explanation
        """

        _, only_pos, removal_args, _ = parse_evaluator_args(evaluation_args)

        assert (
            "perturb_paraling" not in explanation.explainer
        ), f"{explanation.explainer} not supported"

        if "+" in explanation.explainer:
            # 'The explainer name contain "+" to specify the removal type'
            removal_type = explanation.explainer.split("+")[1]
        else:
            # Default
            removal_type = "silence"

        if target is not None:
            warnings.warn(
                'The "target" argument is deprecated and will be removed in a future version. The explanation target are used as default.'
            )

        target = explanation.target

        # Get audio as an array.
        audio_np = explanation.audio.array

        # Get prediction probability of the input sencence for the target
        ground_truth_probs = self.model_helper.predict([audio_np])

        # Get the probability of the target classes
        if self.model_helper.n_labels > 1:
            # Multi-label setting - probability of the target classes for each label (list of size = number of labels)
            ground_truth_probs_target = [
                ground_truth_probs[e][:, tidx][0] for e, tidx in enumerate(target)
            ]
        else:
            # Single probability
            ground_truth_probs_target = [ground_truth_probs[0][target[0]]]

        # TODO: as above, probably a `FerretAudio` object should we passed as
        # input.
        # Split the audio into word-level audio segments
        from ..speechxai_utils import transcribe_audio

        if words_trascript is None:
            words_trascript = explanation.audio.transcription

        get_discrete_rationale_function = (
            _check_and_define_get_id_discrete_rationale_function(
                removal_args["based_on"]
            )
        )

        thresholds = removal_args["thresholds"]

        score_explanation = explanation.scores

        # In this way, we allow for multi-label explanations
        # We have a list of score_explanations, one for each label
        # Each score explanation has length equal to the number of features: number of words in the case of word-level explanations or 1 in the case of paralinguistic level explanation (one paralinguistic feature at the time)
        if score_explanation.ndim == 1:
            score_explanations = [score_explanation]
        else:
            score_explanations = score_explanation

        aopc_comprehesiveness_multi_label = list()

        # We iterate over the target classes for a multi-label setting
        # In the case of single label, we iterate only once
        for target_class_idx, score_explanation in enumerate(score_explanations):
            removal_importances = list()
            id_tops = list()
            last_id_top = None

            # Ground truth probabilities of the target label (target_class_idx) and target class (target[target_class_idx]])
            # It is the output probability of the target class itself in the case of single label
            original_prob = ground_truth_probs_target[target_class_idx]

            # We compute the difference between in probability for all the thresholds
            for v in thresholds:
                # Get rationale from score explanation
                id_top = get_discrete_rationale_function(score_explanation, v, only_pos)

                # If the rationale is the same, we do not include it. In this way, we will not consider in the average the same omission.
                if (
                    id_top is not None
                    and last_id_top is not None
                    and set(id_top) == last_id_top
                ):
                    id_top = None

                id_tops.append(id_top)

                if id_top is None:
                    continue

                last_id_top = set(id_top)

                id_top.sort()

                # Sufficiency
                # The only difference between comprehesivenss and sufficiency is the computation of the removal.

                # For the sufficiency: we keep only the terms in the discrete rationale.
                # Hence, we remove all the other terms.
                words_removed = [
                    words_trascript[i]
                    for i in range(len(words_trascript))
                    if i not in id_top
                ]

                audio_removed = remove_specified_words(
                    explanation.audio.to_pydub(),
                    words_removed,
                    removal_type=removal_type
                )

                audio_removed_np = pydub_to_np(audio_removed)[0]

                # Probability of the modified audio
                audio_modified_probs = self.model_helper.predict([audio_removed_np])

                # Probability of the target class (and label) for the modified audio
                if self.model_helper.n_labels > 1:
                    # In the multi-label setting, we have a list of probabilities for each label

                    # We first take the probability of the corresponding target label target_class_idx
                    # Then we take the probability of the target class for that label target[target_class_idx]
                    modified_prob = audio_modified_probs[target_class_idx][
                        :, target[target_class_idx]
                    ][0]

                else:
                    # Single probability
                    # We take the probability of the target class target[target_class_idx]
                    modified_prob = audio_modified_probs[0][target[target_class_idx]]

                # compute probability difference
                removal_importance = original_prob - modified_prob
                removal_importances.append(removal_importance)

            if removal_importances == []:
                return EvaluationSpeech(self.SHORT_NAME, 0, target)

            #  compute AOPC comprehensiveness
            aopc_comprehesiveness = _compute_aopc(removal_importances)
            aopc_comprehesiveness_multi_label.append(aopc_comprehesiveness)

        evaluation_output = EvaluationSpeech(
            self.SHORT_NAME, aopc_comprehesiveness_multi_label, target
        )

        return evaluation_output
