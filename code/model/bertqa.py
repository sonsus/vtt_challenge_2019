import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertConfig, BertForMultipleChoice
from utils import *
#from einops import rearrange
from copy import deepcopy
from ipdb import set_trace


class bertqa(nn.Module):
    def __init__(self, args, vocab, n_dim, image_dim, layers, dropout, num_choice=5):
        super().__init__()

        self.text_feature_names = args.text_feature_names
        self.feature_names = args.use_inputs

        self.vocab = vocab
        self.args = args

        Bert = BertForMultipleChoice.from_pretrained('bert-base-uncased')
        #DistilBert = DistilBertForSequenceClassification(config)
        #DistilBert = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

        self.embedder = Bert


    @classmethod
    def resolve_args(cls, args, vocab):
        return cls(args, vocab, args.n_dim, args.image_dim, args.layers, args.dropout)

    def forward(self, que, answers, **features):

        bsz, choices, n_dim = list(answers.shape)

        que = que.unsqueeze(-2).repeat(1, choices, 1) # bsz choices seqlen

        paired = torch.cat((que, answers), dim=-1)
        scores = self.embedder(paired)[0] # bsz*choices, 1

        return scores



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
