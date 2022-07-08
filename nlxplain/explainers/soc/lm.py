from builtins import breakpoint
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.distributions import Categorical
from .layers import *


class BiGRULanguageModel(nn.Module):
    def __init__(self, config, vocab, pad_token=0, device=0):
        super().__init__()
        self.hidden_size = config.lm_d_hidden
        self.embed_size = config.lm_d_embed
        self.n_vocab = len(vocab)
        self.device = device

        self.gpu = 0

        # +2 because of <GO> tokens
        self.encoder = DynamicEncoder(self.n_vocab + 2, self.embed_size, self.hidden_size, self.gpu)
        self.fw_proj = nn.Linear(self.hidden_size, self.n_vocab + 2)
        self.bw_proj = nn.Linear(self.hidden_size, self.n_vocab + 2)

        self.loss = nn.CrossEntropyLoss(ignore_index=pad_token)
        self.vocab = vocab

        self.warning_flag = False

        # <GO> tokens
        self.fw_start_token = self.n_vocab
        self.bw_start_token = self.n_vocab + 1
        self.pad_token = pad_token

    def append_start_end_tokens(self, inp, inp_length):
        batch_size = inp.size(1)
        start_tokens = torch.LongTensor([self.fw_start_token] * batch_size).view(1, -1).to(inp.device) # [1,B]
        end_tokens_pad = torch.LongTensor([self.pad_token] * batch_size).view(1, -1).to(inp.device) # [1,B]

        new_inp = torch.cat([start_tokens, inp, end_tokens_pad], 0)
        for b in range(batch_size):
            new_inp[inp_length[b] + 1, b] = self.bw_start_token

        new_inp_length = inp_length + 2
        return new_inp, new_inp_length

    def forward(self, batch):
        inp = batch.text
        inp = inp.to(self.device)

        # append <GO> token
        inp, inp_len = self.append_start_end_tokens(inp, batch.length)

        inp_len_np = inp_len.cpu().numpy()
        output = self.encoder(inp, inp_len_np)
        fw_output, bw_output = output[:,:,:self.hidden_size], output[:,:,self.hidden_size:]
        fw_proj, bw_proj = self.fw_proj(fw_output), self.bw_proj(bw_output)

        inp_trunc = inp[:output.size(0)]
        fw_loss = self.loss(fw_proj[:-1].view(-1,fw_proj.size(2)).contiguous(), inp_trunc[1:].view(-1).contiguous())
        bw_loss = self.loss(bw_proj[1:].view(-1,bw_proj.size(2)).contiguous(), inp_trunc[:-1].view(-1).contiguous())
        return fw_loss, bw_loss

    def sample_single_sequence(self, method, direction, token_inp, hidden, length):
        outputs = []
        for t in range(length):
            output, hidden = self.encoder.rollout(token_inp, hidden, direction=direction)
            if direction == 'fw':
                proj = self.fw_proj(output[:,:,:self.hidden_size])
            elif direction == 'bw':
                proj = self.bw_proj(output[:,:,self.hidden_size:])
            assert(proj.size(0) == 1)
            proj = proj.squeeze(0)
            # outputs.append(proj)
            if method == 'max':
                _, token_inp = torch.max(proj)
                outputs.append(token_inp)
            elif method == 'random':
                dist = Categorical(F.softmax(proj ,-1))
                token_inp = dist.sample()
                outputs.append(token_inp)
            token_inp = token_inp.view(1,-1)
        if direction == 'bw':
            outputs = list(reversed(outputs))
        outputs = torch.stack(outputs)
        return outputs

    def sample_n_sequences(self, method, direction, token_inp, hidden, length, sample_num):
        outputs = []
        token_inp = token_inp.repeat(1, sample_num) # [1, N]
        hidden = hidden.repeat(1, sample_num, 1) # [x, N, H]
        for t in range(length):
            output, hidden = self.encoder.rollout(token_inp, hidden, direction=direction)
            if direction == 'fw':
                proj = self.fw_proj(output[:, :, :self.hidden_size])
            elif direction == 'bw':
                proj = self.bw_proj(output[:, :, self.hidden_size:])
            proj = proj.squeeze(0)
            if method == 'max':
                _, token_inp = torch.max(proj,-1)
                outputs.append(token_inp.view(-1))
            elif method == 'random':
                dist = Categorical(F.softmax(proj, -1))
                token_inp = dist.sample()
                outputs.append(token_inp)
            token_inp = token_inp.view(1, -1)
        if direction == 'bw':
            outputs = list(reversed(outputs))
        outputs = torch.stack(outputs)
        return outputs

    def set_device(self, device):
        if device == "cuda" or device == "cuda:0":
            self.gpu = 0
            self.encoder.to(self.gpu)
        else:
            self.gpu = -1
            self.encoder.to("cpu")

    def sample_n(self, method, batch, max_sample_length, sample_num):
        """
        this function to not assume input have <GO> tokens.

        :param method:
        :param batch:
        :param max_sample_length:
        :param sample_num:
        :return:
        """
        inp = batch.text
        inp_len_np = batch.length.cpu().numpy()

        pad_inp1 = torch.LongTensor([self.fw_start_token] * inp.size(1)).view(1,-1)
        pad_inp2 = torch.LongTensor([self.pad_token] * inp.size(1)).view(1,-1)

        if self.gpu >= 0:
            inp = inp.to(self.gpu)
            pad_inp1 = pad_inp1.to(self.gpu)
            pad_inp2 = pad_inp2.to(self.gpu)

        padded_inp = torch.cat([pad_inp1, inp, pad_inp2], 0)
        padded_inp[inp_len_np + 1] = self.bw_start_token

        assert padded_inp.max().item() < self.n_vocab + 2
        assert inp_len_np[0] + 2 <= padded_inp.size(0)
        padded_enc_out = self.encoder(padded_inp, inp_len_np + 2)  # [T+2,B,H]

        # extract forward hidden state
        assert 0 <= batch.fw_pos.item() - 1 <= padded_enc_out.size(0) - 1
        assert 0 <= batch.fw_pos.item() <= padded_enc_out.size(0) - 1
        fw_hidden = padded_enc_out.index_select(0,batch.fw_pos - 1)
        fw_hidden = torch.cat([fw_hidden[:,:,:self.hidden_size],fw_hidden[:,:,self.hidden_size:]], 0)
        fw_next_token = padded_inp.index_select(0,batch.fw_pos).view(1,-1)

        # extract backward hidden state
        assert 0 <= batch.bw_pos.item() + 3 <= padded_enc_out.size(0) - 1
        assert 0 <= batch.bw_pos.item() + 2 <= padded_enc_out.size(0) - 1
        bw_hidden = padded_enc_out.index_select(0,batch.bw_pos + 3)
        bw_hidden = torch.cat([bw_hidden[:,:,:self.hidden_size], bw_hidden[:,:,self.hidden_size:]], 0)
        bw_next_token = padded_inp.index_select(0,batch.bw_pos + 2).view(1,-1)

        fw_sample_outputs = self.sample_n_sequences(method, 'fw', fw_next_token, fw_hidden, max_sample_length, sample_num)
        bw_sample_outputs = self.sample_n_sequences(method, 'bw', bw_next_token, bw_hidden, max_sample_length, sample_num)

        self.filter_special_tokens(fw_sample_outputs)
        self.filter_special_tokens(bw_sample_outputs)

        return fw_sample_outputs, bw_sample_outputs

    def filter_special_tokens(self, m):
        for i in range(m.size(0)):
            for j in range(m.size(1)):
                if m[i,j] >= self.n_vocab - 2 or m[i,j] == self.vocab.get('[CLS]',0) \
                    or m[i,j] == self.vocab.get('[SEP]',0):
                    m[i,j] = 0
