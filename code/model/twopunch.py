import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel
from model.vaswani import Vaswani
from math import log2
from utils import *
#from einops import rearrange
from copy import deepcopy
from ipdb import set_trace


class Twopunch(nn.Module):
    def __init__(self, args, vocab, n_dim, image_dim, layers, dropout, num_choice=5):
        super().__init__()

        self.text_feature_names = args.text_feature_names
        self.feature_names = args.use_inputs

        self.vocab = vocab
        self.args = args
        V = len(vocab)
        D = n_dim

        Bert = BertModel.from_pretrained('distilbert-base-uncased')
        for param in Bert.parameters():
            param.requires_grad = False # see if this causes prob in optimizer passing
            # if does see
            # https://discuss.pytorch.org/t/how-the-pytorch-freeze-network-in-some-layers-only-the-rest-of-the-training/7088/23
        self.sent_embedder = Bert
        self.image_encoder = Vaswani.resolve_args(args)
        self.distil2imsize = nn.Linear(768, args.image_dim)


        if args.aggregation == 'cat':
            self.answerer = MLP(args, args.n_dim+args.image_dim)
            #self.attention_pull = AddAttn(args, MLP = deepcopy(self.answerer))
        elif args.aggregation in ['elemwise', 'sum']:
            self.answerer = MLP(args, args.n_dim)
            #self.attention_pull = AddAttn(args, MLP = deepcopy(self.answerer))


    @classmethod
    def resolve_args(cls, args, vocab):
        return cls(args, vocab, args.n_dim, args.image_dim, args.layers, args.dropout)

    def forward(self, que, answers, **features):
        bsz, choices, seqlen = list(answers.shape)
        answers = answers.view(-1, seqlen)

        subtitle = features['subtitle']
        images = features['images']

        sampled_images = sample_images(self.args, images) # bsz, args.nsample,

        q = self.distil2imsize(self.sent_embedder(que)[0][:, 0]) # (lasthidden: bsz seqlen hid)[:, 0] == CLS hidden (bsz, hid)
        s = self.distil2imsize(self.sent_embedder(subtitle)[0][:, 0])
        t = self.distil2imsize(self.sent_embedder(answers)[0][:, 0])# (lasthidden: bsz * choices, hid)[:, 0] == CLS hidden (bsz * choices , hid)

        t = t.view(bsz, choices, -1) # bsz, choices, ndim

        im_qs = self.image_encoder(q, sampled_images, s) # bsz, 1, hid
        #q_imqs, _ = self.attention_pull(q, im_qs)
        q_imqs = im_qs

        # bsz, 1, ndim
        q_imqs = q_imqs.repeat(1, choices, 1)

        if self.args.aggregation == 'cat':
            im_qs_t = torch.cat((q_imqs, t), dim=-1) #bsz, choices, ndim+image_dim)
        elif self.args.aggregation == 'elemwise':
            assert im_qs.size(2) == t.size(2)
            # broadcasted
            im_qs_t = q_imqs * t #bsz, choices, ndim
        elif self.args.aggregation == 'sum':
            assert im_qs.size(2) == t.size(2)
            # broadcasted
            im_qs_t = q_imqs + t #bsz, choices, ndim

        o = self.answerer(im_qs_t).squeeze() # bsz, choices, 1 -> bsz, choices

        return o



class MLP(nn.Module):
    def __init__(self, args, n_dim):
        super().__init__()
        self.linears = nn.Sequential(*[
                                        nn.Linear( n_dim//(2**i), n_dim//(2**(i+1)) )
                                        for i in range(int(log2(n_dim)))
                                        ]
                                    )
        self.args = args

    def forward(self, x):
        for layer in self.linears:
            x = F.leaky_relu(
                    F.dropout(layer(x), p=self.args.dropout)
                )
        return x


'''
class AddAttn(nn.Module):
    def __init__(self, args, MLP=None):
        super().__init__()
        self.args = args
        self.attn = None # attention coeffs
        self.mlp = MLP

    def forward(self, q, kv):
        bsz, nsample, image_dim = list(kv.shape)
        q = q.unsqueeze(-2).repeat(1, nsample, 1) # bsz, nsample, ndim
        if self.args.aggregation == 'cat':
            qkv = torch.cat((q,kv), dim=-1)
        elif self.args.aggregation == 'sum':
            qkv = q+kv
        elif self.args.aggregation == 'elemwise':
            qkv = q*kv
        qkv_reduced = self.mlp(qkv) # bsz, nsample, ndim+image_dim(or n_dim) -> bsz, nsample,1
        self.attn = F.softmax(qkv_reduced.squeeze(), dim=-1).unsqueeze(-1) # bsz, nsample, 1
        pulled = (self.attn * kv ).sum(dim=-2) # bsz, 1, ndim
        return pulled, self.attn
'''
