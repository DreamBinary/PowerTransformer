# -*- coding:utf-8 -*-
# @FileName : model.py
# @Time : 2024/3/20 17:48
# @Author : fiv
import math
from typing import Callable
from typing import Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F, TransformerEncoderLayer, TransformerEncoder
from torch.nn.modules.normalization import LayerNorm


class Transformer(nn.Module):
    def __init__(self, n_out=4, wav_length=512, d_model=40, nhead: int = 8, num_encoder_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = True, norm_first: bool = False,
                 bias: bool = True, device=None, dtype=None):
        super(Transformer, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.d_model = d_model
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                activation, layer_norm_eps, batch_first, norm_first,
                                                bias, **factory_kwargs)
        encoder_norm = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # self.embedding = nn.Embedding(n_out, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.linear = nn.Linear(d_model * wav_length, n_out)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        # self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        x = self.pos_encoder(x)
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
