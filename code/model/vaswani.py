import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time

from utils import *
from ipdb import set_trace


class Vaswani(nn.Module):
    """
    Use Decoder w/o mask == Encoder with diff query
    """
    def __init__(self, encoder=None, decoder=None, src_embed=None, tgt_embed=None, generator=None, args=None):
        super(Vaswani, self).__init__()
        #self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.generator = generator
        if args.aggregation == 'cat':
            self.linear = nn.Linear(args.n_dim+args.image_dim, args.n_dim)
        elif args.aggregation in ['sum', 'elemwise']:
            self.linear = nn.Linear(args.n_dim, args.n_dim)
        self.args = args

    @classmethod
    def resolve_args(cls, args):
        c = copy.deepcopy

        emb = Embeddings(args.image_dim, args.image_dim)
        W = emb.lut.weight

        attn = MultiHeadedAttention(args.h, args.image_dim, dropout=args.dropout_tr, max_rel_pos=args.rel_positional_emb)
        ff = PositionwiseFeedForward(args.image_dim, args.d_ff, args.dropout_tr)

        model = Vaswani(
            decoder = Decoder(DecoderLayer(args.image_dim, c(attn), c(attn),
                                 c(ff), args.dropout_tr), args.N),
            src_embed = emb,
            generator = Generator(args.image_dim, args.image_dim, tied_embedding=W),
            args = args
            )
        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


        return model # check model.generator / encoder / decoder / tgt_embed / src_embed

    def forward(self, q, images, s, src_mask=None, tgt_mask=None):
        "Take in and process masked src and target sequences."
        encoded = self.generator(self.decode(q, images, s, src_mask, tgt_mask) )
        return encoded  # bsz, 1, hid

    def decode(self, q, images, s, src_mask, tgt_mask):
        #set_trace()
        #print("decode starts")
        s = s.unsqueeze(-2).repeat(1, self.args.nsample, 1) # bsz, nsample, ndim
        q = q.unsqueeze(-2) # bsz, 1, ndim
        if self.args.aggregation == 'cat':
            fused = torch.cat( (self.src_embed(images), s), dim =-1) # bsz, nsample, ndim+image_dim
        elif self.args.aggregation == 'elemwise':
            assert s.size(2) == images.size(2)
            fused = self.src_embed(images) * s # bsz, nsample, ndim
        elif self.args.aggregation == 'sum':
            assert s.size(2) == images.size(2)
            fused = self.src_embed(images) + s # bsz, nsample, ndim

        fused = self.linear(fused) # bsz, nsample, ndim
        return self.decoder(q, fused, src_mask, tgt_mask) # bsz, 1, ndim

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, in_, out_, tied_embedding=None):
        super(Generator, self).__init__()
        self.proj = nn.Linear(in_, out_)
        if tied_embedding is not None:
            self.proj.weight = tied_embedding
    def forward(self, x):
        return self.proj(x) #F.log_softmax(self.proj(x), dim=-1)



class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        #positional embedding only applies for the foremost layer embeddign step.
        for i in range(1, len(self.layers)):
            self.layers[i].self_attn.max_rel_pos=0
            self.layers[i].src_attn.max_rel_pos=0
        self.layers[0].src_attn.max_rel_pos=0
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            ##set_trace()
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        #print(f"2 DecoderLayer")
        #print(f"x: {x.shape}")
        #print(f"memory: {memory.shape}")
        #print(f"tgt_mask: {tgt_mask.shape}")
        #set_trace()
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask)) # 12 8 512
        
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask)) ## this is original
        return self.sublayer[2](x, self.feed_forward) # 12 1 512 + 12 8 512 = 12 8 512...



class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))



class MultiHeadedAttention(nn.Module):
    def __init__(self, h, image_dim, dropout=0.1, max_rel_pos=0):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert image_dim % h == 0
        # We assume d_v always equals d_k
        self.d_k = image_dim // h
        self.h = h
        self.linears = clones(nn.Linear(image_dim, image_dim), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

        self.rel_pos_emb = nn.Embedding(2*max_rel_pos+1, self.d_k)
        self.max_rel_pos = max_rel_pos

    def forward(self, query, key, value, mask=None):
        '''
        mostly from openNMT MultiHeadedAttention,
        cached property not implemented
        '''
        #set_trace()
        #print("x, m, m: src attention")
        #print("x, x, x: tgt attention (positional)")

        bsz = query.size(0)

        def shape(x):
            """Projection."""
            return x.view(bsz, -1, self.h, self.d_k) \
                .transpose(1, 2)

        def unshape(x):
            """Compute context."""
            return x.transpose(1, 2).contiguous() \
                    .view(bsz, -1, self.h * self.d_k)

        q_linear, k_linear, v_linear, fin_linear = self.linears

        query = q_linear(query)
        key = k_linear(key)
        value = v_linear(value)

        key = shape(key)
        value = shape(value)

        if self.max_rel_pos>0:
            keylen = key.shape[2]
            rel_pos_mat = generate_relative_positions_matrix(keylen, self.max_rel_pos).to(query.device)#, cache=True)
            relpos_keys = self.rel_pos_emb(rel_pos_mat).float()
            relpos_vals = self.rel_pos_emb(rel_pos_mat).clone().float()

        query = shape(query)
        querylen = query.shape[2]
        query = query/math.sqrt(self.d_k)
        q_k = torch.matmul(query, key.transpose(-2,-1))

        scores = q_k
        if self.max_rel_pos>0:
            ##print(f"query: {query.shape}") #b h len d_k
            ##print(f"relpos_keys: {relpos_keys.shape}") # len len d_k
            ##print(f"scores: {scores.shape}, res: {relative_matmul(query, relpos_keys, True).shape}") # batch h len len | b h len len
            ##set_trace()
            scores = scores + relative_matmul(query, relpos_keys, True)

        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e18)

        attn = F.softmax(scores, dim=-1)
        dp_attn = self.dropout(attn)

        context = torch.matmul(dp_attn, value)
        if self.max_rel_pos>0:
            ##print(f"dp_attn: {dp_attn.shape}")
            ##print(f"relpos_vals: {relpos_vals.shape}")
            ##print(f"context: {context.shape}, res: {relative_matmul(dp_attn, relpos_vals, False).shape}")
            ##set_trace()
            context = context + relative_matmul(dp_attn, relpos_vals, False)

        x,  self.attn = context, dp_attn # added to preserve self.attn access

        x = unshape(x)
        return fin_linear(x)



class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, image_dim, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(image_dim, d_ff)
        self.w_2 = nn.Linear(d_ff, image_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Embeddings(nn.Module):
    def __init__(self, in_, out_):
        super(Embeddings, self).__init__()
        self.lut = nn.Linear(in_, out_)
        self.image_dim = in_

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.image_dim)



def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


### OpenNMT-py implementation of relative position encodings
def generate_relative_positions_matrix(length, max_relative_positions,
                                       cache=False):
    """Generate the clipped relative positions matrix
       for a given length and maximum relative positions"""
    if cache:
        distance_mat = torch.arange(-length+1, 1, 1).unsqueeze(0)
    else:
        range_vec = torch.arange(length)
        range_mat = range_vec.unsqueeze(-1).expand(-1, length).transpose(0, 1)
        distance_mat = range_mat - range_mat.transpose(0, 1)
    distance_mat_clipped = torch.clamp(distance_mat,
                                       min=-max_relative_positions,
                                       max=max_relative_positions)
    # Shift values to be >= 0
    final_mat = distance_mat_clipped + max_relative_positions
    return final_mat


def relative_matmul(x, z, transpose):
    """Helper function for relative positions attention."""
    batch_size = x.shape[0] # x: b h len d_k
    heads = x.shape[1]
    length = x.shape[2]
    x_t = x.permute(2, 0, 1, 3) # x_t: len b h d_k
    x_t_r = x_t.reshape(length, heads * batch_size, -1) # len, h*b, d_k
    if transpose:
        z_t = z.transpose(1, 2) # len len d_k => len d_k len
        x_tz_matmul = torch.matmul(x_t_r, z_t) # len h*b len
    else:
        x_tz_matmul = torch.matmul(x_t_r, z)
    x_tz_matmul_r = x_tz_matmul.reshape(length, batch_size, heads, -1) # len b h len
    x_tz_matmul_r_t = x_tz_matmul_r.permute(1, 2, 0, 3)
    return x_tz_matmul_r_t #b h len len
