from . import BaseExplainer
from .utils import parse_explainer_args
from .explanation import Explanation
from lime.lime_text import LimeTextExplainer
import torch
import numpy as np


class LIMEExplainer(BaseExplainer):
    NAME = "LIME"

    def compute_feature_importance(self, text, target=1, **explainer_args):
        init_args, call_args = parse_explainer_args(explainer_args)
        item = self.tokenizer(text, return_tensors="pt")
        token_ids = item["input_ids"][0].tolist()

        # https://github.com/copenlu/xai-benchmark/blob/1cb264c21fb2c0b036127cf3bb8e035c5c5e95da/saliency_gen/interpret_lime.py
        def fn_prediction_token_ids(token_ids_sentences):
            token_ids = [
                [int(i) for i in instance_ids.split(" ") if i != "" and i != "UNKWORDZ"]
                for instance_ids in token_ids_sentences
            ]
            max_batch_id = max([len(_l) for _l in token_ids])
            padded_batch_ids = [
                _l + [self.tokenizer.pad_token_id] * (max_batch_id - len(_l))
                for _l in token_ids
            ]
            tokens_tensor = torch.tensor(padded_batch_ids)
            logits = (
                self.model(tokens_tensor, attention_mask=tokens_tensor.long() > 0)
                .logits.softmax(-1)
                .detach()
                .cpu()
                .numpy()
            )
            return logits

        # Same word has a different relevance according to its position
        lime_explainer = LimeTextExplainer(bow=False, **init_args)

        np.random.seed(42)
        expl = lime_explainer.explain_instance(
            " ".join([str(i) for i in token_ids]),
            fn_prediction_token_ids,
            labels=[target],
            num_features=len(token_ids),
            num_samples=10,
            **call_args
        )

        token_scores = list(dict(sorted(expl.local_exp[target])).values())
        output = Explanation(
            text, self.get_tokens(text), token_scores, self.NAME, target
        )
        return output
