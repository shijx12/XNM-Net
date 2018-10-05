import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic_modules import *
from IPython import embed

class ExistOrCountModule(nn.Module):
    def __init__(self, dim_v):
        super().__init__()
        self.projection = nn.Sequential(
                nn.Linear(1, 128),
                nn.ReLU(),
                nn.Linear(128, dim_v)
                )
        for layer in self.projection:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.constant_(layer.bias, val=0)

    def forward(self, attn):
        out = self.projection(torch.sum(attn, dim=0, keepdim=True)) 
        return out


class FindModule(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.attendNode = AttendNodeModule()
        self.attnAnd = AndModule()

    def forward(self, attn, feat, query):
        new_object_attn = self.attendNode(feat, query)
        out = self.attnAnd(attn, new_object_attn)
        return out


class DescribeModule(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        dim_v = kwargs['dim_v']
        self.count = ExistOrCountModule(dim_v)
        self.gate = nn.Sequential(
                nn.Linear(dim_v, 128),
                nn.ReLU(),
                nn.Linear(128, 2),
                nn.Softmax(dim=0)
                )
        K = kwargs['k_desc']
        self.query_to_weight = nn.Sequential(
            nn.Linear(dim_v, K),
            nn.Softmax(dim=0),
                )
        self.mappings = nn.Parameter(torch.zeros((K, dim_v, dim_v)))
        nn.init.normal_(self.mappings.data, mean=0, std=0.01)

    def forward(self, attn, feat, query):
        # -------- describe ----------
        out_1 = torch.matmul(attn, feat)
        desc_weight = self.query_to_weight(query)
        mapping = torch.sum(self.mappings * desc_weight.view(-1,1,1), dim=0) # (dim_v, dim_v)
        out_1 = torch.matmul(mapping, out_1)
        # -------- count ----------
        out_2 = self.count(attn)
        # ------------------------
        weight = self.gate(query)
        out = torch.matmul(weight, torch.stack([out_1, out_2]))
        return out # (dim_v, )


class RelateModule(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        dim_v = kwargs['dim_v']
        self.attendEdge = AttendEdgeModule()
        self.transWeit = TransWeightModule()
        self.gate = nn.Sequential(
                nn.Linear(dim_v, 128),
                nn.ReLU(),
                nn.Linear(128, 2),
                nn.Softmax(dim=0)
                )

    def forward(self, attn, feat_edge, query):
        weit_matrix = self.attendEdge(feat_edge, query)
        out = self.transWeit(attn, weit_matrix)
        weight = self.gate(query)
        out = torch.matmul(weight, torch.stack([out, attn]))
        return out
