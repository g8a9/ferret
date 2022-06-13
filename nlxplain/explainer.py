"""Main module."""

from builtins import breakpoint
from multiprocessing.sharedctypes import Value
from unicodedata import normalize
import matplotlib.pyplot as plt
import torch
from einops import rearrange
import numpy as np
import pandas as pd
from tqdm import tqdm
from captum.attr import KernelShap, Saliency, IntegratedGradients, InputXGradient
from shap import Explainer as ShapExplainer
from transformers import pipeline
import copy
from lime.lime_text import LimeTextExplainer

# # SOC
# from hiex.soc_api import SamplingAndOcclusionExplain
# from utils.config import configs
# from soc import Processor


class Explainer:
    def __init__(self, model, tokenizer, raw_data=None, proc_data=None):
        self.model = model
        self.tokenizer = tokenizer

        self.raw_data = raw_data
        self.proc_data = proc_data

    def _get_item(self, idx):
        if isinstance(idx, int):
            return self.proc_data[[idx]]
        elif isinstance(idx, str):
            return self.tokenizer(idx, return_tensors="pt")
        else:
            raise ValueError(f"{idx} is of unknown type")

    def _get_input_embeds_from_ids(self, ids):
        embeddings = self.model.bert.embeddings.word_embeddings(ids)
        return embeddings

    def _get_input_embeds(self, idx):
        item = self._get_item(idx)
        embeddings = self._get_input_embeds_from_ids(item["input_ids"][0])
        embeddings = rearrange(embeddings, "s h -> () s h")
        return embeddings

    def _forward(self, idx, no_grad=True, model=None, use_inputs=False):
        model = model if model else self.model
        model.eval()
        item = self._get_item(idx)

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
                outputs = model(
                    **item, output_attentions=True, output_hidden_states=True
                )
                return outputs

        if no_grad:
            with torch.no_grad():
                outputs = _foward_pass(use_inputs)
        else:
            outputs = _foward_pass(use_inputs)

        return outputs

    def _normalize_input_attributions(self, attr):
        attr = attr.sum(-1)  # sum over hidden size
        attr /= attr.norm(dim=-1, p=1)  # L1 vector normalization
        return attr

    def _get_attentions(self, idx, head, layer):
        item = self._get_item(idx)
        input_len = item["attention_mask"][0].sum().item()

        outputs = self._forward(idx)
        attentions = torch.cat(outputs.attentions)
        attentions = rearrange(attentions, "l h s1 s2 -> h l s1 s2")
        attentions = attentions[head, layer, :input_len, :input_len]
        return attentions

    def get_tokens(self, idx):
        item = self._get_item(idx)
        input_len = item["attention_mask"].sum()
        return self.tokenizer.convert_ids_to_tokens(item["input_ids"][0][:input_len])

    def get_hta(self, idx, layer=10):
        item = self._get_item(idx)
        input_len = item["attention_mask"].sum()

        embedding_matrix = self.model.bert.embeddings.word_embeddings.weight
        vocab_size = embedding_matrix.shape[0]
        onehot = torch.nn.functional.one_hot(item["input_ids"][0], vocab_size).float()
        embeddings = torch.matmul(onehot, embedding_matrix)
        embeddings = rearrange(embeddings, "s h -> () s h")

        outputs = self.model(
            inputs_embeds=embeddings,
            attention_mask=item["attention_mask"],
            token_type_ids=item["token_type_ids"],
            output_hidden_states=True,
        )

        # get hidden states of a specific layer
        hidden_states = outputs.hidden_states[layer + 1][0]

        grads = list()
        pbar = tqdm(total=input_len.item())
        for hs in hidden_states[:input_len]:

            grad = torch.autograd.grad(
                hs.unsqueeze(0),
                embeddings,
                grad_outputs=torch.ones_like(hs.unsqueeze(0)),
                retain_graph=True,
            )[0]

            grads.append(grad)
            pbar.update()

        pbar.close()

        grads = torch.cat(grads)  # (input_len, max_len, hidden_size)
        grads = grads[:, :input_len, :]

        # compute per-token HTAs
        htas = list()
        for g in grads:
            g = g.norm(dim=-1)
            g /= g.sum()
            htas.append(g)

        htas = torch.stack(htas)
        return htas

    def get_kernel_shap(self, idx, target=1):
        item = self._get_item(idx)
        input_len = item["attention_mask"].sum().item()

        def func(input_embeds):
            outputs = self.model(
                inputs_embeds=input_embeds,
                attention_mask=item["attention_mask"],
                token_type_ids=item["token_type_ids"],
            )
            scores = outputs.logits.softmax(-1)[0]
            return scores[target]

        ks = KernelShap(func)
        inputs = self._get_input_embeds(idx)
        fmask = list()
        for i in range(inputs.shape[1]):
            fmask.append(torch.full((inputs.shape[-1],), i))
        fmask = torch.stack(fmask).unsqueeze(0)

        attr = ks.attribute(
            inputs, n_samples=200, feature_mask=fmask, show_progress=True
        )
        attr = attr[0, :input_len, 0]  # attributions are equal on the last dim

        return attr

    def get_shap(self, idx, target=1):
        if isinstance(idx, int):
            # no tokenization - raw data
            text = self.raw_data[[idx]]["text"]
        elif isinstance(idx, str):
            # no tokenization
            text = [idx]
        else:
            raise ValueError(f"{idx} is of unknown type")

        pred = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            return_all_scores=True,
        )

        explainer_partition = ShapExplainer(pred)
        shap_values = explainer_partition(text)
        return shap_values.values[0][:, target]

    def _generate_baselines(self, input_len):
        ids = (
            [self.tokenizer.cls_token_id]
            + [self.tokenizer.pad_token_id] * (input_len - 2)
            + [self.tokenizer.sep_token_id]
        )
        embeddings = self._get_input_embeds_from_ids(torch.tensor(ids))
        return embeddings.unsqueeze(0)

    def get_integrated_gradients(self, idx, target=1):
        item = self._get_item(idx)
        input_len = item["attention_mask"].sum().item()

        def func(input_embeds):
            outputs = self.model(
                inputs_embeds=input_embeds,
                attention_mask=item["attention_mask"],
                token_type_ids=item["token_type_ids"],
            )
            scores = outputs.logits[0]
            return scores[target].unsqueeze(0)

        dl = IntegratedGradients(func, multiply_by_inputs=True)
        inputs = self._get_input_embeds(idx)
        baselines = self._generate_baselines(input_len)

        attr = dl.attribute(inputs, baselines=baselines)
        attr = attr[0, :input_len, :]

        norm_attr = self._normalize_input_attributions(attr.detach())
        return norm_attr

    def get_gradients(self, idx, target=1, multiply_by_inputs=False):
        item = self._get_item(idx)
        input_len = item["attention_mask"].sum().item()

        def func(input_embeds):
            outputs = self.model(
                inputs_embeds=input_embeds,
                attention_mask=item["attention_mask"],
                token_type_ids=item["token_type_ids"],
            )
            scores = outputs.logits[0]
            return scores[target].unsqueeze(0)

        dl = InputXGradient(func) if multiply_by_inputs else Saliency(func)

        inputs = self._get_input_embeds(idx)
        attr = dl.attribute(inputs)
        attr = attr[0, :input_len, :]

        norm_attr = self._normalize_input_attributions(attr.detach())
        return norm_attr

    # def get_soc(self, idx, lm_dir, data_dir=None, train_file=None, valid_file=None):
    #     # update SOC configs
    #     configs.hiex = False
    #     configs.lm_dir = lm_dir
    #     configs.data_dir = data_dir
    #     configs.hiex_tree_height = 5
    #     configs.hiex_add_itself = False
    #     configs.hiex_abs = False

    #     processor = Processor(
    #         configs,
    #         tokenizer=self.tokenizer,
    #         train_file=train_file,
    #         valid_file=valid_file,
    #     )

    #     explainer = SamplingAndOcclusionExplain(
    #         model=self.model,
    #         configs=configs,
    #         tokenizer=self.tokenizer,
    #         output_path="hiex_output",  # shouldn't be used
    #         device="cuda:0",
    #         lm_dir=lm_dir,
    #         train_dataloader=processor.get_dataloader("train"),
    #         dev_dataloader=processor.get_dataloader("dev"),
    #         vocab=self.tokenizer.vocab,
    #     )

    #     item = self._get_item(idx)

    #     self.model.to("cuda")
    #     scores = explainer.word_level_explanation_bert(
    #         item["input_ids"].to("cuda"),
    #         item["attention_mask"].to("cuda"),
    #         item["token_type_ids"].to("cuda"),
    #     )

    #     self.model.to("cpu")

    #     scores = torch.tensor(scores)
    #     scores /= scores.norm(dim=-1, p=1)
    #     return scores

    def show_attention(self, idx, head, **kwargs):
        layer = kwargs.get("layer", 10)
        fontsize = kwargs.get("fontsize", 14)
        figsize = kwargs.get("figsize", (8, 8))

        attentions = self._get_attentions(idx, head, layer)

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(attentions)

        item = self.proc_data[idx]
        input_len = item["attention_mask"].sum()
        ticks = self.tokenizer.batch_decode(item["input_ids"][:input_len])

        ax.set_xticks(np.arange(input_len))
        ax.set_yticks(np.arange(input_len))
        ax.set_xticklabels(ticks, rotation=90, fontsize=fontsize)
        ax.set_yticklabels(ticks, fontsize=14)

        fig.tight_layout()

    def _get_effective_attention(self, idx, head, layer, effective_model):
        item = self._get_item(idx)
        input_len = item["attention_mask"].sum()

        outputs = self._forward(idx, model=effective_model)
        values = [v.detach() for v in outputs.value]
        attentions = [a.detach()[0] for a in outputs.attentions]

        # effective_attention_map = []
        # for current_layer in range(12):
        U, S, V = torch.Tensor.svd(values[layer], some=False, compute_uv=True)
        bound = torch.finfo(S.dtype).eps * max(U.shape[1], V.shape[1])

        greater_than_bound = S > bound

        basis_start_index = torch.max(torch.sum(greater_than_bound, dtype=int, axis=2))
        null_space = U[:, :, :, basis_start_index:]

        B = torch.matmul(attentions[layer], null_space)
        transpose_B = torch.transpose(B, -1, -2)
        projection_attention = torch.matmul(null_space, transpose_B)
        projection_attention = torch.transpose(projection_attention, -1, -2)
        effective_attention = torch.sub(attentions[layer], projection_attention)

        # select head in effective attention
        effective_attention = effective_attention[0][head, :input_len, :input_len]
        return effective_attention

    # def show_effective_attention(self, idx, head, **kwargs):
    #     layer = kwargs.get("layer", 10)
    #     fontsize = kwargs.get("fontsize", 14)
    #     effective_model = kwargs["effective_model"]

    #     item = self.proc_data[idx]
    #     input_len = item["attention_mask"].sum()

    #     effective_attention = self._get_effective_attention(idx, head, layer, effective_model=effective_model)

    #     fig, ax = plt.subplots(figsize=(11,11))
    #     ax.imshow(effective_attention)
    #     ticks = self.tokenizer.batch_decode(item["input_ids"][:input_len])

    #     ax.set_xticks(np.arange(input_len))
    #     ax.set_yticks(np.arange(input_len))
    #     ax.set_xticklabels(ticks, rotation=90, fontsize=fontsize)
    #     ax.set_yticklabels(ticks, fontsize=14)

    def compare_attentions(self, idx, head, layer, **kwargs):
        fontsize = kwargs.get("fontsize", 14)
        effective_model = kwargs["effective_model"]
        remove_special_tokens = kwargs.get("remove_special_tokens", True)

        effective_attentions = self._get_effective_attention(
            idx, head, layer, effective_model
        )
        attentions = self._get_attentions(idx, head, layer)
        hta = self.get_hta(idx, layer=layer)

        if remove_special_tokens:
            effective_attentions = effective_attentions[1:-1, 1:-1]
            attentions = attentions[1:-1, 1:-1]
            hta = hta[1:-1, 1:-1]

        fig, ax = plt.subplots(ncols=3, figsize=(18, 8), sharey=True)
        ax1, ax2, ax3 = ax
        ax1.imshow(attentions)
        # ax1.set_title("Attention")
        ax2.imshow(effective_attentions)
        # ax2.set_title("Effective attention")
        ax3.imshow(hta)
        # ax3.set_title("HTA")

        item = self._get_item(idx)
        input_len = item["attention_mask"].sum().item()

        ticks = self.tokenizer.batch_decode(item["input_ids"][0][:input_len])

        if remove_special_tokens:
            ticks = ticks[1:-1]

        list(map(lambda x: x.set_xticks(np.arange(len(ticks))), ax))
        list(
            map(lambda x: x.set_xticklabels(ticks, rotation=90, fontsize=fontsize), ax)
        )

        ax1.set_yticks(np.arange(len(ticks)))
        ax1.set_yticklabels(ticks, fontsize=fontsize)

        # ax1.set_xticks()
        # ax1.set_xticklabels(ticks, rotation=90, fontsize=fontsize)

        fig.tight_layout()

        return fig

    def get_gradient(self, idx, target=1):
        outputs, embeddings = self._forward(idx, use_inputs=True, no_grad=False)
        out = outputs.logits[0][target]

        item = self._get_item(idx)
        input_len = item["attention_mask"].sum().item()

        # compute loss
        scores = outputs.logits.softmax(-1)[0]
        loss = -torch.log(scores[target] / scores.exp().sum())  # cross entropy

        # compute gradients of loss wrt input embeddings
        grad = torch.autograd.grad(loss, embeddings)[0]
        grad = grad[:, :input_len, :]

        embeddings = embeddings[:, :input_len, :]

        prods = list()
        for g, e in zip(grad[0], embeddings[0]):
            r = torch.dot(-g, e)
            prods.append(r)

        grad_input = torch.tensor(prods)
        grad_input /= grad_input.norm(dim=-1, p=1)  # l1 normalization

        normalized_grad = self._normalize_input_attributions(grad[0])

        # normalized_grad = grad[0].sum(-1) # avg over hidden size
        # normalized_grad /= normalized_grad.norm(dim=-1, p=1) # normalize over tokens

        return grad_input, normalized_grad

    
    def get_lime_explanation(self, idx, target=1):
        
        if isinstance(idx, int):
            # no tokenization - raw data (a single str)
            text = self.raw_data[[idx]]["text"][0]
        elif isinstance(idx, str):
            # no tokenization - the input sentence (str)
            text = idx
        else:
            raise ValueError(f"{idx} is of unknown type")

        #https://github.com/copenlu/xai-benchmark/blob/1cb264c21fb2c0b036127cf3bb8e035c5c5e95da/saliency_gen/interpret_lime.py
        def fn_prediction_token_ids(token_ids_sentences):
            token_ids = [[int(i) for i in instance_ids.split(' ') if i != '' and i !="UNKWORDZ"] for
                                instance_ids in token_ids_sentences]
            max_batch_id = max([len(_l) for _l in token_ids])
            padded_batch_ids = [
                _l + [self.tokenizer.pad_token_id] * (max_batch_id - len(_l))
                for _l in token_ids]
            tokens_tensor = torch.tensor(padded_batch_ids)
            logits = self.model(tokens_tensor, attention_mask=tokens_tensor.long() > 0).logits.softmax(-1).detach().cpu().numpy()
            return logits



        from lime.lime_text import LimeTextExplainer
        # Same word has a different relevance according to its position 
        lime_explainer = LimeTextExplainer(bow = False)
        token_ids = self.tokenizer.encode(text)

        np.random.seed(42)        
        expl = lime_explainer.explain_instance(
                " ".join([str(i) for i in token_ids]), fn_prediction_token_ids,
                labels = [target],
                num_features=len(token_ids), num_samples=10)
            
            
        token_scores = list(dict(sorted(expl.local_exp[target])).values())
        
        return token_scores

    """ Explanation at the word level - unused """


    def get_lime_explanation_word(self, idx, target=1):
        
        if isinstance(idx, int):
            # no tokenization - raw data (a single str)
            text = self.raw_data[[idx]]["text"][0]
        elif isinstance(idx, str):
            # no tokenization - the input sentence (str)
            text = idx
        else:
            raise ValueError(f"{idx} is of unknown type")

        def fn_prediction(sentences):
            inputs = self.tokenizer(sentences, return_tensors="pt", padding="longest")
            with torch.no_grad():
                outputs = self.model(**inputs)

            logits = outputs.logits.softmax(-1).detach().cpu().numpy()
            return logits

        from lime.lime_text import LimeTextExplainer
        lime_explainer = LimeTextExplainer(bow = False)
        token_ids = self.tokenizer.encode(text)

        np.random.seed(42)
        expl = lime_explainer.explain_instance(
                            text, fn_prediction,
                            labels = [target],
                            num_features=len(token_ids))
        expl_scores = list(dict(sorted(expl.local_exp[target])).values())
        return expl_scores
    
    def classify(self, idx):
        text = idx if isinstance(idx, str) else self.raw_data[idx]["text"]

        print("IDX:", idx)
        print("Text:", text)

        outputs = self._forward(idx)
        logits = outputs.logits

        if not isinstance(idx, str):
            print("True label:", self.raw_data[idx]["label"])

        scores = torch.nn.functional.softmax(logits, -1)
        print("Probabilities:", scores)
        print("Prediction:", logits.argmax(-1).item())

        return scores, logits.argmax(-1).item()

    def get_predicted_label(self, idx):
        outputs = self._forward(idx)
        logits = outputs.logits

        prediction = logits.argmax(-1).item()
        return prediction

    def compute_table(self, idx, target=1, **kwargs):
        """Compute a comparison table.

        `idx` can either be an index of the dataset or a string
        """
        d = dict()

        item = self._get_item(idx)
        input_len = item["attention_mask"].sum().item()
        tokens = self.tokenizer.batch_decode(item["input_ids"][0])[:input_len]

        # saliency methods
        # grad_inputs, normalized_grad = self.get_gradient(idx, target=target)
        grads = self.get_gradients(idx, target)
        grads_by_inputs = self.get_gradients(idx, target, multiply_by_inputs=True)
        ig = self.get_integrated_gradients(idx, target=target)

        # shap
        # k_shap = self.get_kernel_shap(idx, target=target)

        # SHAP library - SHAP Partition with transformer
        p_shap = self.get_shap(idx, target=target)
        normalized_p_shap = torch.tensor(p_shap)
        normalized_p_shap /= normalized_p_shap.norm(dim=-1, p=1)

        # LIME
        lime_expl = self.get_lime_explanation(idx, target=target)
        normalized_lime = torch.tensor(lime_expl)
        normalized_lime /= normalized_lime.norm(dim=-1, p=1)

        # SOC
        # soc_kwargs = kwargs.get("soc_kwargs", dict())
        # soc = self.get_soc(
        #     idx, **soc_kwargs
        # )  # target is always for class = 1 (? see implementation)

        d = {
            "tokens": tokens,
            "G": grads,
            "GxI": grads_by_inputs,
            "IG": ig,
            "SHAP": normalized_p_shap,
            # "SOC": soc,
            "LIME": normalized_lime
        }

        table = pd.DataFrame(d).set_index("tokens").T
        table = table.iloc[:, 1:-1]

        return table

    def compute_occlusion_importance(self, idx, target=1, remove_first_last=True):
        item = self._get_item(idx)
        input_len = item["attention_mask"].sum().item()
        input_ids = item["input_ids"][0][:input_len].tolist()

        if remove_first_last == True:
            input_ids = input_ids[1:-1]

        outputs = self._forward(idx)
        logits = outputs.logits[0]
        baseline = logits.softmax(-1)[target].item()

        samples = list()
        for occ_idx in range(len(input_ids)):
            sample = copy.copy(input_ids)
            sample.pop(occ_idx)
            sample = self.tokenizer.decode(sample)
            samples.append(sample)

        inputs = self.tokenizer(samples, return_tensors="pt", padding="longest")

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits.softmax(-1)[:, target]
        occlusion_importance = logits - baseline

        return occlusion_importance
