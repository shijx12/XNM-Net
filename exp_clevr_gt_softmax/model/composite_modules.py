import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic_modules import *


class AttentionModule(nn.Module):
    """ 
    Corresponding CLEVR programs: 'filter_<att>'
    Output: attention
    """
    def __init__(self):
        super().__init__()
        self.attendNode = AttendNodeModule()
        self.transConn = TransferModule()
        self.attnAnd = AndModule()

    def forward(self, attn, conn_matrix, feat, query):
        """
        Args:
            attn [Tensor] (num_node, ) : attention weight produced by previous module
            conn_matrix [Tensor] (num_node, num_node)
            feat [Tensor] (num_node, dim_v)
            query [Tensor] (dim_v, )
        """
        attribute_attn = self.attendNode(feat, query)
        new_object_attn = self.transConn(attribute_attn, conn_matrix)
        out = self.attnAnd(attn, new_object_attn)
        return out


class SameModule(nn.Module):
    """
    Corresponding CLEVR programs: 'same_<cat>' 
    Output: attention
    """
    def __init__(self):
        super().__init__()
        self.relate = RelateModule()
        self.transConn = TransferModule()
        self.attnAnd = AndModule()
        self.attnNot = NotModule()
        
    def forward(self, attn, cat_matrix, edge_cat_vectors, query, conn_matrix):
        attribute_attn = self.relate(attn, cat_matrix, edge_cat_vectors, query, None)
        new_object_attn = self.transConn(attribute_attn, conn_matrix)
        # exclude current object
        out = self.attnAnd(self.attnNot(attn), new_object_attn)
        return out



class ExistOrCountModule(nn.Module):
    """
    Corresponding CLEVR programs: 'exist', 'count'
    Output: encoding
    """
    def __init__(self, dim_v):
        super().__init__()
        self.projection = nn.Sequential(
                nn.Linear(1, 128),
                nn.ReLU(),
                nn.Linear(128, dim_v)
                )

    def forward(self, attn):
        out = self.projection(torch.sum(attn, dim=0, keepdim=True)) 
        return out



class RelateModule(nn.Module):
    """
    Corresponding CLEVR programs: 'relate_<cat>'. Besides, it is also used inside QueryModule and SameModule
    Output: attention
    """
    def __init__(self):
        super().__init__()
        self.attendEdge = AttendEdgeModule()
        self.transfer = TransferModule()

    def forward(self, attn, cat_matrix, edge_cat_vectors, query, feat):
        """
        Args:
            attn [Tensor] (num_node, ) : attention weight produced by previous module
            cat_matrix [Tensor] (num_node, num_node)
            edge_cat_vectors [Tensor] (num_edge_cat, dim_v)
            query [Tensor] (dim_v, )
        """
        weit_matrix = self.attendEdge(edge_cat_vectors, query, cat_matrix)
        out = self.transfer(attn, weit_matrix)
        return out


class QueryModule(nn.Module):
    """
    Corresponding CLEVR programs: 'query_<cat>'
    Output: encoding
    """
    def __init__(self):
        super().__init__()
        self.relate = RelateModule()

    def forward(self, attn, cat_matrix, edge_cat_vectors, query, feat):
        attribute_attn = self.relate(attn, cat_matrix, edge_cat_vectors, query, None)
        out = torch.matmul(attribute_attn, feat)
        return out # (dim_v, )


class ComparisonModule(nn.Module):
    """
    Corresponding CLEVR programs: 'equal_<cat>', 'equal_integer', 'greater_than' and 'less_than'
    Output: encoding
    """
    def __init__(self, dim_v):
        super().__init__()
        self.projection = nn.Sequential(
                nn.Linear(dim_v, 128),
                nn.ReLU(),
                nn.Linear(128, dim_v)
            )

    def forward(self, enc1, enc2):
        input = enc1 - enc2
        out = self.projection(input)
        return out

        
