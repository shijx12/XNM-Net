import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed

class FindModule(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.map_q = nn.Sequential(
                nn.Linear(kwargs['dim_hidden'], kwargs['dim_v'])
                )
        self.map_weight = nn.Sequential(
            nn.Linear(kwargs['dim_v'], 1),
            nn.Sigmoid(),
            )

    def forward(self, attn, feat, feat_edge, query, relation_mask):
        query = self.map_q(query)
        elt_prod = F.normalize(query.unsqueeze(1)*feat, p=2, dim=2) # (batch_size, num_feat, dim_v)
        new_attn = self.map_weight(elt_prod).squeeze(2) # (bsz, num_feat) 
        out = torch.min(attn, new_attn)
        return out


class RelateModule(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.map_q = nn.Sequential(
                nn.Linear(kwargs['dim_hidden'], kwargs['dim_edge'])
                )
        self.map_weight = nn.Sequential(
            nn.Linear(kwargs['dim_edge'], 1),
            nn.Sigmoid(),
            )
        self.gate = nn.Sequential(
                nn.Linear(kwargs['dim_edge'], 128),
                nn.ReLU(),
                nn.Linear(128, 2),
                nn.Softmax(dim=1)
                )

    def forward(self, attn, feat, feat_edge, query, relation_mask):
        batch_size = query.size(0)
        query = self.map_q(query)
        weit_matrix = self.map_weight(feat_edge * query.view(batch_size, 1, 1, -1)).squeeze(3)
        weit_matrix = weit_matrix * relation_mask.float()
        new_attn = torch.matmul(attn.unsqueeze(1), weit_matrix).squeeze(1)
        # ---------
        norm = torch.max(new_attn, dim=1, keepdim=True)[0].detach()
        norm[norm<=1] = 1
        new_attn /= norm
        # ---------
        weight = self.gate(query)
        # print(weight)
        out = torch.matmul(weight.unsqueeze(1), torch.stack([attn, new_attn], dim=1)).squeeze(1) #(bsz, num_node)
        return out


class DescribeModule(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.count = nn.Sequential(
                nn.Linear(1, 128),
                nn.ReLU(),
                nn.Linear(128, kwargs['dim_v'])
                )
        self.gate = nn.Sequential(
                nn.Linear(kwargs['dim_hidden'], 256),
                nn.ReLU(),
                nn.Linear(256, 2),
                nn.Softmax(dim=1)
                )

    def forward(self, attn, feat, feat_edge, query, relation_mask):
        # -------- count ----------
        count = self.count(torch.sum(attn, dim=1, keepdim=True))
        # -------- describe ----------
        attn = F.softmax(attn, dim=1)
        desc = torch.matmul(attn.unsqueeze(1), feat).squeeze(1)
        # ------------------------
        weight = self.gate(query)
        out = torch.matmul(weight.unsqueeze(1), torch.stack([count, desc], dim=1)).squeeze(1) #(bsz, dim_v)
        return out # (bsz, dim_v)

