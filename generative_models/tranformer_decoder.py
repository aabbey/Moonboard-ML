import sys

import torch
from torch import nn
import prediction_models
import torch.nn.init as init
import math

device = "cuda" if torch.cuda.is_available() else "cpu"
max_len = 28 + 2  # + 2 for start and end tokens
hold_channels = 198  # one hot encoding for each hold

# hyper parameters
batch_size = 64
n_heads = 8
att_dropout = 0.2
ff_dropout = 0.2


class MyCrLoss(nn.Module):
    def __init__(self):
        super(MyCrLoss, self).__init__()
        self.lsm = nn.LogSoftmax(dim=0)

    def forward(self, logits, targets):
        lsm = self.lsm(logits)
        return - torch.sum(targets @ lsm.transpose(-2, -1))


class Transformer(nn.Module):
    def __init__(self, feature_size, head_size, block_size=14):
        super(Transformer, self).__init__()
        self.sa = TransformerHead(feature_size, head_size)
        self.out_head = nn.Linear(head_size, feature_size)

    def forward(self, x):
        sa_out = self.sa(x)
        logits = self.out_head(sa_out)
        return logits


class TransformerHead(nn.Module):
    def __init__(self, feature_size, head_size, block_size=14):
        super(TransformerHead, self).__init__()
        self.key = nn.Linear(feature_size, head_size, bias=False)
        self.query = nn.Linear(feature_size, head_size, bias=False)
        self.value = nn.Linear(feature_size, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, C)
        q = self.query(x)  # (B, T, C)
        v = self.value(x)  # (B, T, C)
        #calc attention weights
        wei = k @ q.transpose(-2, -1)  # (B, T, T)
        #scale
        wei = wei * 1/(math.sqrt(C))
        # mask to weight only tokens that came before
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = wei.softmax(dim=1)
        out = wei @ v
        return out


class TransformerDecoder(nn.Module):
    """
    stacks blocks together to make the full model
    """
    def __init__(self, embed_size, n_blocks, hold_embed_size):
        super(TransformerDecoder, self).__init__()

        self.block_stack = nn.Sequential(*[Block(embed_size) for _ in range(n_blocks)])
        self.model_head = nn.Linear(embed_size, hold_embed_size)

    def forward(self, x):
        x = self.block_stack(x)
        return self.model_head(x)


class Block(nn.Module):
    """
    performs communication between nodes (attention scores between each hold) via "multi_att", then
    performs computation on that output via "feed_forward"
    """
    def __init__(self, embed_size):
        super(Block, self).__init__()

        self.mult_att = MultiheadedAttention(embed_size)
        self.feed_forward = FeedForward(embed_size)
        self.layer_norm = nn.LayerNorm(embed_size)

    def forward(self, x):
        x = x + self.mult_att(self.layer_norm(x))  # residual connection (x +), norm calculated before
        x = x + self.feed_forward(self.layer_norm(x))  # residual connection (x +), norm calculated before
        return x


class MultiheadedAttention(nn.Module):
    """
    calculates attention scores between nodes
    """
    def __init__(self, embed_size):
        super(MultiheadedAttention, self).__init__()
        head_size = embed_size // n_heads

        B, T, C = (batch_size, max_len, head_size)

        self.key = nn.Linear(hold_channels, head_size, bias=False)
        self.query = nn.Linear(hold_channels, head_size, bias=False)
        self.value = nn.Linear(hold_channels, head_size, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(size=(B, T, C))))
        self.dropout = nn.Dropout(att_dropout)
        self.projection = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        B, T, C = x.size()

        k = self.key(x)
        q = self.query(x)
        wei = k @ q.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))
        wei = wei.masked_fill(self.mask[:,:T,:T] == 0, float('-inf'))
        wei = wei.softmax()
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v
        return self.projection(out)


class FeedForward(nn.Module):
    def __init__(self, embed_size):
        super(FeedForward, self).__init__()
        self.ff = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.LeakyReLU()
        )
        self.projection = nn.Linear(embed_size * 4, embed_size)
        self.dropout = nn.Dropout(ff_dropout)

    def forward(self, x):
        x = self.projection(self.ff(x))
        return self.dropout(x)
