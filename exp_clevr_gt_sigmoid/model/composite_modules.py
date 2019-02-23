import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic_modules import *


class AttentionModule(nn.Module):
    """ 
    Corresponding CLEVR programs: 'filter_<att>'
    Output: attention
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.attendNode = AttendNodeModule()
        self.attnAnd = AndModule()

    def forward(self, attn, feat, query):
        """
        Args:
            attn [Tensor] (num_node, ) : attention weight produced by previous module
            feat [Tensor] (num_node, dim_v) : node features
            query [Tensor] (dim_v, ) : embedding of <att>, such as 'red', 'large', and etc
        """
        new_attn = self.attendNode(feat, query)
        out = self.attnAnd(attn, new_attn)
        return out


class SameModule(nn.Module):
    """
    Corresponding CLEVR programs: 'same_<cat>' 
    Output: attention
    """    
    def __init__(self, **kwargs):
        super().__init__()
        self.query = QueryModule(**kwargs)
        self.attend = AttentionModule(**kwargs)
        self.attnNot = NotModule()
        
    def forward(self, attn, feat, query):
        """
        Args:
            attn [Tensor] (num_node, ) : attention weight produced by previous module
            feat [Tensor] (num_node, dim_v) : node features
            query [Tensor] (dim_v, ) : embedding of <cat>, such as 'color'
        """
        # map category query to corresponding value query, such as, 'color' -> 'red'
        value_query = self.query(attn, feat, query)
        # find other red objects
        out = self.attend(self.attnNot(attn), feat, value_query)
        return out



class ExistOrCountModule(nn.Module):
    """
    Corresponding CLEVR programs: 'exist', 'count'
    Output: encoding
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.projection = nn.Sequential(
                nn.Linear(1, 128),
                nn.ReLU(),
                nn.Linear(128, kwargs['dim_v'])
                )

    def forward(self, attn):
        # sum up the node attention
        out = self.projection(torch.sum(attn, dim=0, keepdim=True)) 
        return out



class RelateModule(nn.Module):
    """
    Corresponding CLEVR programs: 'relate_<cat>'
    Output: attention
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.attendEdge = AttendEdgeModule()
        self.transfer = TransferModule()

    def forward(self, attn, feat, query, edge_feat):
        """
        Args:
            attn [Tensor] (num_node, ) : attention weight produced by previous module
            feat [Tensor] (num_node, dim_v) : node features
            query [Tensor] (dim_v, ), embedding of <cat>, specifying a relationship category, such as 'left'
            edge_feat [Tensor] (num_node, num_node, dim_v) : edge features
        """
        weit_matrix = self.attendEdge(edge_feat, query)
        out = self.transfer(attn, weit_matrix)
        return out


class QueryModule(nn.Module):
    """
    Corresponding CLEVR programs: 'query_<cat>'
    Output: encoding
    """
    def __init__(self, **kwargs):
        super().__init__()
        dim_v = kwargs['dim_v']
        K = 4  # given attribute category number
        self.query_to_weight = nn.Sequential(
            nn.Linear(dim_v, K),
            nn.Softmax(dim=0),
                )
        self.mappings = nn.Parameter(torch.zeros((K, dim_v, dim_v)))
        nn.init.normal_(self.mappings.data, mean=0, std=0.01)

    def forward(self, attn, feat, query):
        """
        Args:
            attn [Tensor] (num_node, ) : attention weight produced by previous module
            feat [Tensor] (num_node, dim_v) : node features
            query [Tensor] (dim_v, ) : embedding of <cat>, specifying an attribute category, such as 'color'
        """
        out = torch.matmul(attn, feat)
        # compute a probability distribution over K aspects
        weight = self.query_to_weight(query)
        mapping = torch.sum(self.mappings * weight.view(-1,1,1), dim=0) # (dim_v, dim_v)
        out = torch.matmul(mapping, out)
        return out # (dim_v, )


class ComparisonModule(nn.Module):
    """
    Corresponding CLEVR programs: 'equal_<cat>', 'equal_integer', 'greater_than' and 'less_than'
    Output: encoding
    """
    def __init__(self, **kwargs):
        super().__init__()
        dim_v = kwargs['dim_v']
        self.projection = nn.Sequential(
                nn.Linear(dim_v, 128),
                nn.ReLU(),
                nn.Linear(128, dim_v)
            )

    def forward(self, enc1, enc2):
        """
        Args: two encodings from two QueryModule or two CountModule
        """
        input = enc1 - enc2
        out = self.projection(input)
        return out

        
