from ..benchmark import lp_normalize
from . import BaseDataset
import thermostat


TRAIN_SET = "train"
VALIDATION_SET = "validation"
TEST_SET = "test"

NONE_RATIONALE = []


def check_termostat_config(name):
    """
    Check if the input name is among the Thermostat admitted ones
    """
    assert any(
        [name in c.name for c in thermostat.data.thermostat_configs.builder_configs]
    ), f"{name} not in the admitted thermostat configuration of thermostat"


def get_supported_explainer_by_config(name):
    """
    Given the configuration name as a tuple 'dataset-model', we get the admitted explainer"
    """
    thermostat_configs = thermostat.data.thermostat_configs.builder_configs
    # We direclty use lime, omitting lime-100, if available
    return [
        c.name.replace(f"{name}-", "")
        for c in thermostat_configs
        if name in c.name and "lime-100" not in c.name
    ]


def _check_explainers_are_supported(name_explainers, supported_explainers):
    not_supported = [e for e in name_explainers if e not in supported_explainers]
    if not_supported:
        raise ValueError(
            f"{not_supported}  in not the supported explainers: {supported_explainers}"
        )


def _is_complete_thermostat_input_config(name):
    n_configs = len(name.split("-"))
    assert n_configs >= 2

    return True if n_configs >= 3 else False


class ThermostatDataset(BaseDataset):

    NAME = "Thermostat"
    avg_rationale_size = None

    def __init__(self, name, name_explainers=None):

        """
        Load of thermostat dataset(s)

        name: dataset-model-explainer or dataset-model
        A thermostat dataset is based on the triple ("dataset", "model", "explainer")
        In the latter case, we load the thermostat datasets (for the specified dataset and model)
        for all the explainers specified in name_explainers (all admitted ones if name_explainers is None)

        name_explainers: list of admitted explainers. If none, we use all the admitted explainers
        """

        check_termostat_config(name)

        # Load the termostat dataset
        # A thermostat dataset is based on the triple ("dataset", "model", "explainer")
        # We first check if the entire triple is provided dataset-model-explainer
        if _is_complete_thermostat_input_config(name):
            # If true, we load the thermostat dataset
            self.test_datasets = [thermostat.load(name)]
            # The explainer name is the third element of the config
            explainer_name = "-".join(name.split("-")[2:])
            self.explainers = [explainer_name]
        else:
            # We load all thermostat datasets with config name ("dataset-model") for multiple explainers

            supported_explainers = get_supported_explainer_by_config(name)
            if not name_explainers:
                # We use all supported explainers for the input consideration
                self.explainers = supported_explainers
            else:

                _check_explainers_are_supported(name_explainers, supported_explainers)

                # We use the explainers provided as input
                self.explainers = name_explainers

            # We load all the thermostat datasets for the given config (dataset-model) for all the self.explainers
            self.test_datasets = list()
            for explainer_name in self.explainers:
                self.test_datasets.append(thermostat.load(f"{name}-{explainer_name}"))

        self.test_dataset = self.test_datasets[0]
        self.tokenizer = self.test_dataset.tokenizer
        self.classes = range(0, len(self.test_dataset.label_names))
        self.model_name = self._get_model_name()
        self.tokenizer_name = self._get_tokenizer_name()

    def __len__(self):
        return self.len()

    def len(self):
        return len(self.test_dataset)

    def _get_item(self, idx: int):
        if isinstance(idx, int):
            item_idx = self.test_dataset[idx]
            return item_idx
        elif isinstance(idx, thermostat.data.dataset_utils.Thermounit):
            return idx
        else:
            raise ValueError()

    def __getitem__(self, idx):
        # We use the TEST_SET as default
        return self.get_instance(idx)

    def get_instance(self, idx, normalize_scores: bool = True):

        """
        Get the instance at index idx.
        Args:
            idx
            normalize_score: if set to True, explanations scores are normalized to ease the comparison

        Returns:
            dict representing the instance.

            An instance of the dataset is composed by the
            - text
            - tokens
            - label: the ground truth
            - predicted_label: the predicted label by the Thermostat model under analysis
            - explanations: the list of explanation (List[Explanation])
        """

        item_idx = self._get_item(idx)

        text = self._get_text(item_idx)
        tokens = self._get_tokens(item_idx)

        true_label = self._get_ground_truth(item_idx)
        predicted_label = self._get_predicted_label(item_idx)

        # Thermostat explanations are by default built w.r.t the predicted class
        explanations = self.get_explanations(
            idx, text, tokens, predicted_label, normalize_scores=normalize_scores
        )

        instance = {
            "text": text,
            "tokens": tokens,
            "label": true_label,
            "predicted_label": predicted_label,
            "explanations": explanations,
        }
        return instance


    def _get_tokens(self, idx):

        item_idx = self._get_item(idx)
        # item_idx.tokens is a dict of ids (from 0 to #tokens in item_idx) and tokens. We directly extract the tokens
        tokens = list(item_idx.tokens.values())

        pad_token = self.tokenizer.pad_token
        if pad_token in tokens:
            # The tokens may also include pad tokens
            # If the pad token is present, we truncate the tokens til the first pad (not included)
            idx_first_pad_token = tokens.index(pad_token)
            tokens = tokens[:idx_first_pad_token]
        return tokens

    def _get_text(self, idx):
        item_idx = self._get_item(idx)

        token_ids = item_idx.input_ids

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id in token_ids:
            # The token_ids may also include pad tokens
            # If the pad token is present, we take the tokens til the first pad (not included)
            idx_first_pad_token = token_ids.index(pad_token_id)
            token_ids = token_ids[:idx_first_pad_token]

        # 1 -1 to remove CLS and SEP tokens
        text = self.tokenizer.decode(token_ids[1:-1])
        return text

    def _get_rationale(self, idx):
        # Thermostat dataset does not include human rationales
        return None

    def _get_ground_truth(self, idx, split_type: str = TEST_SET):
        item_idx = self._get_item(idx)
        return item_idx.label

    def _get_predicted_label(self, idx):
        item_idx = self._get_item(idx)
        return item_idx.predicted_label_index

    def _get_model_name(self):
        return self.test_dataset.model_name

    def _get_tokenizer_name(self):
        return self.test_dataset.tokenizer.name_or_path


    def get_target_explanations(self, idx):
        """
        Thermostat explanations are by default built w.r.t the predicted class
        """    
        return self._get_predicted_label(idx) 


    def get_explanations(
        self, idx, text=None, tokens=None, target=None, normalize_scores: bool = True
    ):
        from ..explainers.explanation import Explanation

        if text is None:
            text = self._get_text(idx)
        if tokens is None:
            tokens = self._get_tokens(idx)
        if target is None:
            target = self.get_target_explanations(idx)

        explanations = []
        for data, explainer_name in zip(self.test_datasets, self.explainers):

            # Thermostat explanations are in form (token, explanation scores, id).
            # We take only the explanation scores (1)
            # We keep the importance of CLS and SEP
            scores = [e[1] for e in data[idx].explanation]

            explanation = Explanation(text, tokens, scores, explainer_name, target)
            explanations.append(explanation)

        if normalize_scores:
            explanations = lp_normalize(explanations)

        return explanations
