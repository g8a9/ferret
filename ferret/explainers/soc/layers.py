import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class DynamicEncoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, gpu, n_layers=1, dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.embedding = nn.Embedding(input_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers, bidirectional=True)

        self.gpu = gpu

    def forward(self, input_seqs, input_lens, hidden=None, return_hidden=False):
        """
        forward procedure. **No need for inputs to be sorted**
        :param return_hidden:
        :param input_seqs: Variable of [T,B]
        :param hidden:
        :param input_lens: *numpy array* of len for each input sequence
        :return:
        """
        batch_size = input_seqs.size(1)
        embedded = self.embedding(input_seqs)
        embedded = embedded.transpose(0, 1)  # [B,T,E]
        sort_idx = np.argsort(-input_lens)
        unsort_idx = torch.LongTensor(np.argsort(sort_idx))
        if self.gpu >= 0:
            unsort_idx = unsort_idx.to(self.gpu)
        input_lens = input_lens[sort_idx]
        sort_idx = torch.LongTensor(sort_idx)
        if self.gpu >= 0:
            sort_idx = sort_idx.to(self.gpu)
        embedded = embedded[sort_idx].transpose(0, 1)  # [T,B,E]
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lens)
        outputs, hidden = self.gru(packed, hidden)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        # outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        outputs = outputs.transpose(0, 1)[unsort_idx].transpose(0, 1).contiguous()
        return outputs

    def rollout(self, input_word, prev_hidden, direction):
        embed = self.embedding(input_word)
        output, hidden = self.gru(embed, prev_hidden)
        return output, hidden
