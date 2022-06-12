import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
from math import sqrt
from reformer_pytorch import ReformerLM, Autopadder


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = nn.Embedding(c_in, d_model)

    def forward(self, x):
        x = self.value_embedding(x)
        return x


class Reformer(nn.Module):
    def __init__(self, enc_in, c_out, out_len,
                 factor=5, d_model=128, n_heads=8, e_layers=6, d_ff=128,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False):
        super(Reformer, self).__init__()
        self.output_attention = output_attention

        # Encoding & Attention
        self.attn = ReformerLM(
                    num_tokens=enc_in,
                    dim=d_model,
                    depth=e_layers,
                    max_seq_len=500,
                    heads=n_heads,
                    # lsh_dropout=dropout,
                    # ff_dropout=dropout,
                    # post_attn_dropout=dropout,
                    layer_dropout=dropout,
                    # emb_dim=64,
                    # ff_chunks=200,
                    # attn_chunks=8,
                    causal=True,
                    reverse_thres=1024,
                    full_attn_thres=1024,
                    use_full_attn=False,
                    axial_position_emb=True,
                    axial_position_shape=(25, 25),
                    return_embeddings=True,
                )
        self.attn = Autopadder(self.attn)
        self.projection = nn.Linear(out_len, c_out, bias=True)

    def forward(self, x_enc):
        enc_out = self.attn(x_enc)
        # print(enc_out.shape)
        enc_out = enc_out.reshape(enc_out.shape[0], -1)
        # print(enc_out.shape)
        enc_out = self.projection(enc_out)

        return enc_out  # [B, L, D]


if __name__ == '__main__':
    model = Reformer(4**4, 1, 128000, d_model=128, n_heads=8, e_layers=6, d_ff=128, dropout=0.6).cuda()
    x = torch.randint(4**4, (1, 1000)).cuda()
    y = model(x)
    print(y.shape)