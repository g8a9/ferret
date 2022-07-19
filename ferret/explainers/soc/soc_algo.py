import torch
from .agglomeration import *
import numpy as np
import copy
import pickle
from skimage import measure
from torch.nn import functional as F


class DotDict:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


Batch = DotDict


def convert_examples_to_features_sst(examples, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = example.text
        mapping = example.mapping

        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        # if tokens_b:
        #     tokens += tokens_b + ["[SEP]"]
        #     segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        mapping += [-1] * (max_seq_length - len(mapping))

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = example.label

        features.append(
            DotDict(input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    label_id=label_id,
                    offset=example.offset,
                    mapping=mapping))
    return features


class _SamplingAndOcclusionBaseAlgo:
    def __init__(self, model, tokenizer, output_path, configs):
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_length = configs.max_seq_length
        self.batch_start = 0
        self.batch_stop = 50000
        self.batch_size = configs.batch_size
        self.nb_range = configs.nb_range
        self.output_path = output_path
        self.configs = configs

        self.gpu = 0

    def occlude_input_with_masks(self, inp, inp_mask, x_regions, nb_regions):
        region_indicator = np.zeros(len(inp), dtype=np.int32)
        for region in nb_regions:
            region_indicator[region[0]:region[1] + 1] = 2
        for region in x_regions:
            region_indicator[region[0]:region[1] + 1] = 1
        # input expectation over neighbourhood
        inp_enb = copy.copy(inp)
        inp_mask_enb = copy.copy(inp_mask)
        # input expectation over neighbourhood and selected span
        inp_ex = copy.copy(inp)
        inp_mask_ex = copy.copy(inp_mask)

        inp_enb, inp_mask_enb = self.mask_region_masked(inp_enb, inp_mask_enb, region_indicator, [2])
        inp_ex, inp_mask_ex = self.mask_region_masked(inp_ex, inp_mask_ex, region_indicator, [1, 2])

        return inp_enb, inp_mask_enb, inp_ex, inp_mask_ex

    def mask_region_masked(self, inp, inp_mask, region_indicator, mask_value):
        new_seq = []
        new_mask_seq = []
        flg = False
        for i in range(len(region_indicator)):
            if region_indicator[i] not in mask_value:
                new_seq.append(inp[i])
                new_mask_seq.append(inp_mask[i])
            elif region_indicator[i] == 1 and not flg:
                new_seq.append(self.tokenizer.vocab['[PAD]'])
                # TODO: test performance of this
                new_mask_seq.append(0)
            else:
                new_seq.append(self.tokenizer.vocab['[PAD]'])
                new_mask_seq.append(0)
                # flg = True
        if not new_seq:
            new_seq.append(self.tokenizer.vocab['[PAD]'])
        new_seq = np.array(new_seq)
        new_mask_seq = np.array(new_mask_seq)
        return new_seq, new_mask_seq

    def get_ngram_mask_region(self, region, inp):
        # find the [PAD] token
        idx = 0
        while idx < len(inp) and inp[idx] != 0:
            idx += 1
        return [(max(region[0] - self.nb_range, 1), min(region[1] + self.nb_range, idx - 2))]

    def do_attribution(self, input_ids, input_mask, segment_ids, region, label=None):
        raise NotImplementedError

    def agglomerate(self, inputs, percentile_include, method, sweep_dim,
                    num_iters=5, subtract=True, absolute=True, label=None):
        """
        Code from ACD paper
        :param inputs:
        :param percentile_include:
        :param method:
        :param sweep_dim:
        :param num_iters:
        :param subtract:
        :param absolute:
        :param label:
        :return:
        """
        text_orig = inputs[0].cpu().clone().numpy().transpose((1, 0))
        for t in range(text_orig.shape[0]):
            if text_orig[t, 0] == 0:
                text_orig = text_orig[:t]
                break
        text_len = text_orig.shape[0]
        score_orig = self.do_attribution(*inputs, region=[1, text_len - 2], label=label)
        # get scores
        texts = gen_tiles(text_orig, method=method, sweep_dim=sweep_dim)
        texts = texts.transpose()

        starts, stops = tiles_to_cd(texts)

        scores = []
        for (start, stop) in zip(starts, stops):
            score = self.do_attribution(*inputs, region=[start, stop], label=label)
            scores.append(score)

        # threshold scores
        mask = threshold_scores(scores, percentile_include, absolute=absolute)

        # initialize lists
        scores_list = [np.copy(scores)]
        mask_list = [mask]
        comps_list = []
        comp_scores_list = [{0: score_orig}]

        # iterate
        for step in range(num_iters):
            # find connected components for regions
            comps = np.copy(measure.label(mask_list[-1], background=0, connectivity=1))

            # loop over components
            comp_scores_dict = {}
            for comp_num in range(1, np.max(comps) + 1):

                # make component tile
                comp_tile_bool = (comps == comp_num)
                comp_tile = gen_tile_from_comp(text_orig, comp_tile_bool, method)

                # make tiles around component
                border_tiles = gen_tiles_around_baseline(text_orig, comp_tile_bool,
                                                         method=method,
                                                         sweep_dim=sweep_dim)

                # predict for all tiles
                # format tiles into batch
                tiles_concat = np.hstack((comp_tile, np.squeeze(border_tiles[0]).transpose()))
                # batch.text.data = torch.LongTensor(tiles_concat).to(device)

                starts, stops = tiles_to_cd(tiles_concat)
                scores_all = []
                for (start, stop) in zip(starts, stops):
                    score = self.do_attribution(*inputs, region=[start, stop], label=label)
                    scores_all.append(score)

                score_comp = np.copy(scores_all[0])
                scores_border_tiles = np.copy(scores_all[1:])

                # store the predicted class scores
                comp_scores_dict[comp_num] = np.copy(score_comp)

                # update pixel scores
                tiles_idxs = border_tiles[1]
                for i, idx in enumerate(tiles_idxs):
                    scores[idx] = scores_border_tiles[i] - score_comp

            # get class preds and thresholded image
            scores = np.array(scores)
            scores[mask_list[-1]] = np.nan
            mask = threshold_scores(scores, percentile_include, absolute=absolute)

            # add to lists
            scores_list.append(np.copy(scores))
            mask_list.append(mask_list[-1] + mask)
            comps_list.append(comps)
            comp_scores_list.append(comp_scores_dict)

            if np.sum(mask) == 0:
                break

        # pad first image
        comps_list = [np.zeros(text_orig.size, dtype=np.int)] + comps_list

        return {'scores_list': scores_list,  # arrs of scores (nan for selected)
                'mask_list': mask_list,  # boolean arrs of selected
                'comps_list': comps_list,  # arrs of comps with diff number for each comp
                'comp_scores_list': comp_scores_list,  # dicts with score for each comp
                'score_orig': score_orig}  # original score

    def repr_result_region(self, inp, spans, contribs, label=None):
        tokens = self.tokenizer.convert_ids_to_tokens(inp)
        outputs = []
        assert (len(spans) == len(contribs))
        for span, contrib in zip(spans, contribs):
            outputs.append((' '.join(tokens[span[0]:span[1] + 1]), contrib))
        output_str = ' '.join(['%s %.6f\t' % (x, y) for x, y in outputs])
        if label is not None and hasattr(self, 'label_vocab') and self.label_vocab is not None:
            output_str = self.label_vocab.itos[label] + '\t' + output_str
        return output_str


class _SamplingAndOcclusionAlgo(_SamplingAndOcclusionBaseAlgo):
    def __init__(self, model, tokenizer, lm_model, output_path, configs):
        super().__init__(model, tokenizer, output_path, configs)
        self.lm_model = lm_model
        self.batch_size = configs.batch_size
        self.sample_num = configs.sample_n
        self.mask_outside_nb = configs.mask_outside_nb
        self.use_padding_variant = configs.use_padding_variant
        self.hiex_tree_height = configs.hiex_tree_height
        self.hiex_add_itself = configs.hiex_add_itself
        self.hiex_abs = configs.hiex_abs
        self.device = configs.device if hasattr(configs, "device") else "cpu"

    def occlude_input_with_masks_and_run(self, inp_bak, inp_mask, segment_ids, x_regions, nb_regions, label,
                                         return_variable=False, additional_mask=[]):
        x_region = x_regions[0]
        nb_region = nb_regions[0]
        inp = copy.copy(inp_bak)

        inp_length = 0
        for i in range(len(inp_mask)):
            if inp_mask[i] == 1:
                inp_length += 1
            else:
                break
        inp_lm = copy.copy(inp)
        for i in range(len(inp_lm)):
            if nb_region[0] <= i <= nb_region[1] and not x_region[0] <= i <= x_region[1]:
                inp_lm[i] = self.tokenizer.vocab['[PAD]']

        inp_th = torch.from_numpy(inp_lm).long().view(-1, 1)
        inp_length = torch.LongTensor([inp_length])
        fw_pos = torch.LongTensor([min(x_region[1] + 1, len(inp))])
        bw_pos = torch.LongTensor([max(x_region[0] - 1, -1)])

        if self.gpu >= 0:
            inp_th = inp_th.to(self.gpu)
            inp_length = inp_length.to(self.gpu)
            fw_pos = fw_pos.to(self.gpu)
            bw_pos = bw_pos.to(self.gpu)

        batch = Batch(text=inp_th, length=inp_length, fw_pos=fw_pos, bw_pos=bw_pos)

        inp_enb, inp_ex = [], []
        inp_enb_mask, inp_ex_mask = [], []

        max_sample_length = self.nb_range + 1
        fw_sample_outputs, bw_sample_outputs = self.lm_model.sample_n('random', batch,
                                                                      max_sample_length=max_sample_length,
                                                                      sample_num=self.sample_num)

        extra = 0
        if self.hiex_add_itself:
            extra = 1

        for sample_i in range(self.sample_num + extra):
            if sample_i == self.sample_num:
                # add itself
                filled_inp = copy.copy(inp)
            else:
                fw_sample_seq, bw_sample_seq = fw_sample_outputs[:, sample_i].cpu().numpy(), \
                                               bw_sample_outputs[:, sample_i].cpu().numpy()
                filled_inp = copy.copy(inp)

                if self.mask_outside_nb:
                    for i in range(len(filled_inp)):
                        if 1 <= i < nb_region[0] or nb_region[1] < i <= inp_length - 2:
                            filled_inp[i] = self.tokenizer.vocab['[PAD]']

                len_bw = x_region[0] - nb_region[0]
                len_fw = nb_region[1] - x_region[1]
                if len_bw > 0:
                    filled_inp[nb_region[0]:x_region[0]] = bw_sample_seq[-len_bw:]
                if len_fw > 0:
                    filled_inp[x_region[1] + 1:nb_region[1] + 1] = fw_sample_seq[:len_fw]

            filled_inp_, mask_inp_ = [], []
            for i in range(len(filled_inp)):
                if self.configs.keep_other_nw or not (i in additional_mask and not x_region[0] <= i <= x_region[1]):
                    filled_inp_.append(filled_inp[i])
                    mask_inp_.append(inp_mask[i])
                else:
                    filled_inp_.append(self.tokenizer.vocab['[PAD]'])
                    mask_inp_.append(0)

            filled_ex, mask_ex = [], []

            for i in range(len(filled_inp)):
                if not x_region[0] <= i <= x_region[1] and (i not in additional_mask or self.configs.keep_other_nw):
                    filled_ex.append(filled_inp[i])
                    mask_ex.append(inp_mask[i])
                else:
                    if self.configs.keep_other_nw:
                        filled_ex.append(self.tokenizer.vocab['[PAD]'])
                        mask_ex.append(0)

            filled_inp_ = np.array(filled_inp_, dtype=np.int32)
            mask_inp_ = np.array(mask_inp_, dtype=np.int32)
            inp_enb.append(filled_inp_)
            inp_enb_mask.append(mask_inp_)

            filled_ex = np.array(filled_ex, dtype=np.int32)
            mask_ex = np.array(mask_ex, dtype=np.int32)
            inp_ex.append(filled_ex)
            inp_ex_mask.append(mask_ex)

        inp_enb, inp_ex = np.stack(inp_enb), np.stack(inp_ex)
        inp_enb_mask, inp_ex_mask = np.stack(inp_enb_mask), np.stack(inp_ex_mask)
        inp_enb, inp_ex = torch.from_numpy(inp_enb).long(), torch.from_numpy(inp_ex).long()
        inp_enb_mask, inp_ex_mask = torch.from_numpy(inp_enb_mask).long(), torch.from_numpy(inp_ex_mask).long()

        if self.gpu >= 0:
            inp_enb, inp_ex = inp_enb.to(self.gpu), inp_ex.to(self.gpu)
            inp_enb_mask, inp_ex_mask = inp_enb_mask.to(self.gpu), inp_ex_mask.to(self.gpu)
            segment_ids = segment_ids.to(self.gpu)

        inp_enb_mask = inp_enb_mask.expand(inp_enb.size(0), -1)
        segment_ids = segment_ids.expand(inp_enb.size(0), -1)

        logits_enb = self.model(
            input_ids=inp_enb,
            token_type_ids=segment_ids[:, :inp_enb.size(1)],
            attention_mask=inp_enb_mask
        )
        logits_ex = self.model(
            input_ids=inp_ex,
            token_type_ids=segment_ids[:, :inp_ex.size(1)],
            attention_mask=inp_ex_mask
        )

        # if type(logits_enb) is tuple:
        #     logits_enb = logits_enb[0]
        #     logits_ex = logits_ex[0]
        logits_enb = logits_enb.logits
        logits_ex = logits_ex.logits

        contrib_logits = logits_enb - logits_ex
        contrib_score = contrib_logits[:, 1] - contrib_logits[:, 0]  # [B]

        contrib_score = contrib_score.mean()
        if not return_variable:
            return contrib_score.item()
        else:
            return contrib_score

    def do_attribution(self, input_ids, input_mask, segment_ids, region, label=None, return_variable=False,
                       additional_mask=[]):
        inp_flatten = input_ids.view(-1).cpu().numpy()
        inp_mask_flatten = input_mask.view(-1).cpu().numpy()
        mask_regions = self.get_ngram_mask_region(region, inp_flatten)

        score = self.occlude_input_with_masks_and_run(inp_flatten, inp_mask_flatten, segment_ids, [region],
                                                      mask_regions, label, return_variable, additional_mask)
        return score

    def do_attribution_pad_variant(self, input_ids, input_mask, segment_ids, region, label=None, return_variable=False,
                                   additional_mask=[]):
        """
        A variant of SOC algorithm that pads the context instead of sampling. Will be faster but it will degenerate into
        direct feed, which is trivial and performs not as good as original SOC. Used for debugging.
        :param input_ids:
        :param input_mask:
        :param segment_ids:
        :param region:
        :param label:
        :return:
        """
        inp_flatten = input_ids.view(-1).cpu().numpy()
        inp_mask_flatten = input_mask.view(-1).cpu().numpy()

        mask_regions = self.get_ngram_mask_region(region, inp_flatten)

        inp_enb, inp_mask_enb, inp_ex, inp_mask_ex = self.occlude_input_with_masks(inp_flatten, inp_mask_flatten,
                                                                                   [region], mask_regions,
                                                                                   additional_mask)

        inp_enb, inp_mask_enb, inp_ex, inp_mask_ex = torch.from_numpy(inp_enb).long().view(1, -1), torch.from_numpy(
            inp_mask_enb).long().view(1, -1), torch.from_numpy(inp_ex).long().view(1, -1), torch.from_numpy(
            inp_mask_ex).long().view(1, -1)

        if self.gpu >= 0:
            inp_enb, inp_mask_enb, inp_ex, inp_mask_ex = inp_enb.to(self.gpu), inp_mask_enb.to(self.gpu), \
                                                         inp_ex.to(self.gpu), inp_mask_ex.to(self.gpu)
            segment_ids = segment_ids.to(self.gpu)
        logits_enb = self.model(inp_enb, segment_ids[:, :inp_enb.size(1)], inp_mask_enb)
        logits_ex = self.model(inp_ex, segment_ids[:, :inp_ex.size(1)], inp_mask_ex)
        contrib_logits = logits_enb - logits_ex  # [1 * C]

        contrib_score = contrib_logits[0, 1] - contrib_logits[0, 0]
        if not return_variable:
            return contrib_score.item()
        else:
            return contrib_score
    
    def do_hierarchical_explanation(self, input_ids, input_mask, segment_ids, label_ids):
        logits_pred = self.model(input_ids, segment_ids, input_mask)
        logits_pred = logits_pred[:,1] - logits_pred[:,0]

        inp = input_ids.view(-1).cpu().numpy()
        lists = self.agglomerate((input_ids, input_mask, segment_ids), percentile_include=90, method='cd',
                                 sweep_dim=1, num_iters=self.hiex_tree_height, label=label_ids.item(),
                                 absolute=self.hiex_abs)
        lists = collapse_tree(lists)
        seq_len = lists['scores_list'][0].shape[0]
        data = lists_to_tabs(lists, seq_len)
        text = ' '.join(self.tokenizer.convert_ids_to_tokens(inp)[:seq_len])
        tab = {
            'tab': data,
            'text': text,
            'label': label_ids.item(),
            'pred': logits_pred.item()
        }
        return tab


