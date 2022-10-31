# -*- coding: utf-8 -*-
# @Author : zheng
# @Date : 2022/7/20 17:03

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

rnn_class = {
    'RNN': nn.RNN,
    'GRU': nn.GRU,
    'LSTM': nn.LSTM
}

class RNNModel(nn.Module):
    def __init__(self, input_size=128, hidden_size=128, num_layers=1, rnn_type='LSTM', bidirectional=True,
                 bias=True, batch_first=True, dropout=0.0, packing=True, sorted=True):
        """
        A rnn model as sequence encoder.
        Args:
            input_size：Embedding dim of input x
            hidden_size：Embedding dim of hidden state h
            num_layers：Number of recurrent layers. Default: 1
            bias: If False, then the layer does not use bias weights b_ih and b_hh. Default: True
            nonlinearity: The non-linearity to use. Default: 'tanh'
            batch_first: If True, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature). Default: False
            dropout: If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer, with dropout probability equal to dropout. Default: 0
            bidirectional：If True, becomes a bidirectional LSTM. Default: False
        """
        super(RNNModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size  #  // 2 if bidirectional else hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional

        self.dropout = dropout
        self.bias =bias
        self.batch_first = batch_first

        self.packing = packing
        self.sorted = sorted

        rnn = rnn_class[rnn_type]
        self.rnn = rnn(input_size=self.input_size,
                       hidden_size=self.hidden_size,
                       num_layers=self.num_layers,
                       dropout=self.dropout,
                       bidirectional=self.bidirectional,
                       bias=self.bias,
                       batch_first=self.batch_first
                       )

    def forward(self, x, x_mask=None):
        """
        Args:
            torch.Tensor [batch_size, seq_max_length, input_size], [B, L, H_in] 一般是经过embedding后的值
            x_len: torch.Tensor [L] 已经排好序的句长值
        Returns:
            output: torch.Tensor [B, L, H_out] 序列标注的使用结果
            hn:     torch.Tensor [B, N, H_out] / [B, H_out] 分类的结果，当 last_layer_hn 时只有最后一层结果
        """
        batch_size, max_seq_length, input_size = x.shape
        # print(self.rnn)
        # print(x)
        # print(x_mask)

        if self.packing:
            # [batch_size * max_seq_length]
            seq_lengths = x_mask.eq(1).long().sum(1)  # .squeeze()
            # if self.sorted:
            #     _, index_sorted = torch.sort(seq_lengths, dim=0, descending=True)
            #     # print(logit_sorted)
            #     # print(index_sorted)
            #     _, index_unsorted = torch.sort(index_sorted, dim=0)
            #     # seq_lengths = list(seq_lengths[index_sorted])
            #     # print(logit_unsorted, index_unsorted)
            #     seq_lengths = seq_lengths[index_sorted]
            #     x = x.index_select(0, index_sorted)
            # seq_lengths = torch.as_tensor(seq_lengths, dtype=torch.int64)
            # input = pack_padded_sequence(x, seq_lengths.cpu(), batch_first=True, enforce_sorted=True)
            input = pack_padded_sequence(x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
        else:
            input = x

        # if bidirectional：output ---> (batch_size, max_seq_length, num_direction * hidden_size)
        # else：output ---> (batch_size, max_seq_length, hidden_size)
        # hn ---> (num_direction, batch_size, hidden_size)
        output, hn = self.rnn(input)

        if self.rnn_type == 'LSTM':
            hn = hn[0]

        output, _ = pad_packed_sequence(output, batch_first=True, total_length=max_seq_length)

        if self.bidirectional:
            num_direction = 2
            #  [batch_size * max_seq_length * hidden_size]
            output = torch.sum(output.view(batch_size, max_seq_length, num_direction, self.hidden_size), dim=2) / 2
            # [batch_size * hidden_size]
            hn = torch.sum(hn, dim=0) / 2

        # if self.sorted:
        #     output = output.index_select(0, index_unsorted)

        if x_mask is not None:
            output = x_mask.float().unsqueeze(-1).mul(output)

        # print(output)
        # print(hn)

        return output, hn
