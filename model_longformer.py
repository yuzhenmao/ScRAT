import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
from math import sqrt
from transformers import LongformerModel, LongformerConfig
import pdb


class Longformer(nn.Module):
    def __init__(self, enc_in, c_out, out_len,
                 factor=5, d_model=128, n_heads=8, e_layers=6, d_ff=128,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False):
        super(Longformer, self).__init__()

        # Encoding & Attention
        configuration = LongformerConfig()
        configuration.num_attention_heads = n_heads
        configuration.num_hidden_layers = e_layers
        configuration.hidden_size = d_model
        configuration.max_position_embeddings = 500
        configuration.attention_window = 500
        self.attn = LongformerModel(configuration)

        self.projection = nn.Linear(out_len, c_out, bias=True)

    def forward(self, x_enc):
        enc_out = self.attn(x_enc).last_hidden_state
        # print(enc_out.shape)
        enc_out = enc_out.reshape(enc_out.shape[0], -1)
        # print(enc_out.shape)

        enc_out = self.projection(enc_out)

        return enc_out  # [B, L, D]


if __name__ == '__main__':
    model = Longformer(4 ** 4, 1, 128000, d_model=128, n_heads=8, e_layers=6, d_ff=128, dropout=0.6).cuda()
    x = torch.randint(4 ** 4, (1, 1000)).cuda()
    y = model(x)
    print(y.shape)