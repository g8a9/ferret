import torch


class Model:
    """
    Model wrapper
    """

    def __init__(self, model, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer

    def _get_input_embeds_from_ids(self, ids):
        embeddings = self.model.bert.embeddings.word_embeddings(ids)
        return embeddings

    def _get_class_predicted_probability(self, text, tokenizer, target):
        outputs = self._forward(text, tokenizer)
        logits = outputs.logits[0]
        class_prob = logits.softmax(-1)[target].item()
        return class_prob

    # TODO - Uniformate
    def _get_class_predicted_probabilities_texts(self, texts, tokenizer, target):
        # TODO
        tokenizer = tokenizer if tokenizer else self.tokenizer
        if tokenizer is None:
            raise ValueError("Tokenizer is not specified")
        inputs = tokenizer(texts, return_tensors="pt", padding="longest")

        with torch.no_grad():
            outputs = self.model(**inputs)

        return outputs.logits.softmax(-1)[:, target]

    def _forward(self, idx, tokenizer=None, no_grad=True, use_inputs=False):
        self.model.eval()
        tokenizer = tokenizer if tokenizer else self.tokenizer
        if tokenizer is None:
            raise ValueError("Tokenizer is not specified")

        item = tokenizer(idx, return_tensors="pt")

        def _foward_pass(use_inputs=False):

            if use_inputs:
                embeddings = self._get_input_embeds(idx)
                outputs = self.model(
                    inputs_embeds=embeddings,
                    attention_mask=item["attention_mask"],
                    token_type_ids=item["token_type_ids"],
                    output_hidden_states=True,
                )

                return outputs, embeddings

            else:
                outputs = self.model(
                    **item, output_attentions=True, output_hidden_states=True
                )
                return outputs

        if no_grad:
            with torch.no_grad():
                outputs = _foward_pass(use_inputs)
        else:
            outputs = _foward_pass(use_inputs)

        return outputs
