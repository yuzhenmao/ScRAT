import torch
from torch import nn
import math
from torch.autograd import Variable
import copy


###### Transformer ######
class Embedder(nn.Module):
    def __init__(self, vocab_size, h_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, h_dim)

    def forward(self, x):
        return self.embed(x)


class PositionalEncoder(nn.Module):
    def __init__(self, h_dim, max_seq_len=500):
        super().__init__()
        self.h_dim = h_dim

        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, h_dim)
        for pos in range(max_seq_len):
            for i in range(0, h_dim, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / h_dim)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / h_dim)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.h_dim)
        # add constant to embedding
        seq_len = x.size(1)
        x = x + Variable(self.pe[:, :seq_len], requires_grad=False).cuda()
        return x


def Attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = nn.Softmax(dim=-1)(scores)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output, scores


class MultiHeadAttention(nn.Module):
    def __init__(self, attention, heads, h_dim, dropout=0.1):
        super().__init__()

        self.h_dim = h_dim
        self.d_k = h_dim // heads
        self.h = heads

        self.attention = attention

        self.q_linear = nn.Linear(h_dim, h_dim)
        self.v_linear = nn.Linear(h_dim, h_dim)
        self.k_linear = nn.Linear(h_dim, h_dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(h_dim, h_dim)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * h_dim

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        scores, scores_ = self.attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.h_dim)

        output = self.out(concat)

        return output, scores_


class Norm(nn.Module):
    def __init__(self, h_dim, eps=1e-6):
        super().__init__()

        self.size = h_dim
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class FeedForward_(nn.Module):
    def __init__(self, h_dim1, h_dim2, d_ff=2048, dropout=0.1):
        super().__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(h_dim1, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, h_dim2)

    def forward(self, x):
        x = self.dropout(nn.ReLU(True)(self.linear_1(x)))
        x = self.linear_2(x)
        return x


# build an encoder layer with one multi-head attention layer and one # feed-forward layer
class EncoderLayer(nn.Module):
    def __init__(self, attention, h_dim, heads, d_ff=2048, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(h_dim)
        self.norm_2 = Norm(h_dim)
        self.attn = MultiHeadAttention(attention, heads, h_dim, dropout)
        self.ff = FeedForward_(h_dim, h_dim, d_ff, dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        out, weights = self.attn(x2, x2, x2, mask)
        x = x + self.dropout_1(out)
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x, weights


# We can then build a convenient cloning function that can generate multiple layers:
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Encoder(nn.Module):
    def __init__(self, emb_dim, h_dim, N, heads, attention, d_ff, dropout):
        super().__init__()
        self.N = N
        # self.embed = Embedder(vocab_size, h_dim)
        # self.pe = PositionalEncoder(h_dim)
        self.linear_1 = nn.Linear(emb_dim, h_dim)
        self.layers = get_clones(EncoderLayer(attention, h_dim, heads, d_ff, dropout), N)
        self.norm = Norm(h_dim)

    def forward(self, src, mask):
        x = self.linear_1(src)
        # x = self.embed(src)
        # x = self.pe(x)
        for i in range(self.N):
            x, w = self.layers[i](x, mask)
        return self.norm(x), w

###### Transformer ######

class Transformer(nn.Module):
    def __init__(self, seq_len=100, input_dim=100, emb_dim=128, h_dim=128, N=6, heads=8, attention=Attention, d_ff=2048,
                 dropout=0.1, cl=6, pca=False):
        super().__init__()
        if pca:
            emb_dim = input_dim
        self.dimRedu = torch.nn.Sequential(nn.Linear(input_dim, emb_dim*2), nn.ReLU(), nn.Dropout(p=dropout), nn.Linear(emb_dim*2, emb_dim))
        # self.dimRedu = torch.nn.Sequential(nn.Linear(input_dim, emb_dim), nn.ReLU(), nn.Linear(emb_dim, emb_dim))
        self.encoder = Encoder(emb_dim, h_dim, N, heads, attention, d_ff, dropout)
        self.out = nn.Linear(h_dim, cl)
        self.attens = None
        self.dropout = nn.Dropout(dropout)
        self.pca = pca

    def forward(self, src, src_mask=None):
        if not self.pca:
            src = self.dimRedu(src)
        e_outputs, self.attens = self.encoder(src, src_mask)
        # e_outputs = self.encoder(src, src_mask)
        e_outputs = e_outputs.mean(1)
        output = self.out(e_outputs)
        return output

###### Linear model ######

class FeedForward(nn.Module):
    def __init__(self, input_dim, h_dim=50, cl=2, dropout=0.3):
        super().__init__()
        self.ff = FeedForward_(h_dim1=input_dim, h_dim2=cl, d_ff=h_dim, dropout=dropout)

    def forward(self, src, mask=None):
        e_outputs = torch.mean(src, dim=1)
        output = self.ff(e_outputs)
        return output


class Linear_Classfier(nn.Module):
    def __init__(self, input_dim=50, cl=2):
        super().__init__()
        self.out1 = nn.Linear(input_dim, cl)

    def forward(self, src, mask=None):
        e_outputs = torch.mean(src, dim=1)
        output = self.out1(e_outputs)
        return output


class scFeedForward(nn.Module):
    def __init__(self, input_dim=50, cl=2, model_dim=128, dropout=0, pca=False):
        super().__init__()
        self.input_net = FeedForward_(h_dim1=input_dim, h_dim2=model_dim, d_ff=model_dim, dropout=dropout)
        self.dimRedu_net = torch.nn.Sequential(
            nn.Linear(input_dim, model_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 2, model_dim)
        )
        # Output classifier per sequence lement
        self.output_net = nn.Sequential(
            # nn.Linear(model_dim, model_dim),
            # nn.LayerNorm(model_dim),
            # nn.ReLU(inplace=True),
            # nn.Dropout(dropout),
            nn.Linear(model_dim, cl)
        )
        self.pca = pca

    def forward(self, src, mask=None):
        if not self.pca:
            x = self.dimRedu_net(src)
        else:
            x = self.input_net(src)
        x = x.mean(1)
        x = self.output_net(x)
        return x

