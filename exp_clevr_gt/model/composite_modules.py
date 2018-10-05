import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic_modules import *


class AttentionModule(nn.Module):
    """ 
    1 unified instance: 'filter_<cat>'
    value_inputs: attribute values such as 'green', 'large', etc. 
    output: attention
    """
    def __init__(self):
        super().__init__()
        self.attendNode = AttendNodeModule()
        self.transConn = TransConnectModule()
        self.attnAnd = AndModule()

    def forward(self, attn, conn_matrix, feat, query):
        """
        Args:
            attn [Tensor] (num_node, ) : attention weight produced by previous module
            conn_matrix [Tensor] (num_node, num_node)
            feat [Tensor] (num_node, dim_v)
            query [Tensor] (dim_v, )
        Returns:
            attention
        """
        attribute_attn = self.attendNode(feat, query)
        new_object_attn = self.transConn(attribute_attn, conn_matrix)
        out = self.attnAnd(attn, new_object_attn)
        return out


class SameModule(nn.Module):
    """
    1 unified instance: 'same_<cat>'. 
    value_inputs: <cat> including 'color','shape','material' and 'size'
    output: attention
    """

    def __init__(self):
        super().__init__()
        self.relate = RelateModule()
        self.transConn = TransConnectModule()
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
    2 instance: 'exist', 'count'
    input: attention
    output: encoding
    """
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
        # TODO: other strategies? maps variable-length attn to fixed-length encoding
        return out



class RelateModule(nn.Module):
    """
    3 instances: 'relate_<cat>', 'query_<cat>' contained in QueryModule, 
            and 'same_<cat>' contained in SameModule
    value_inputs: <cat>
    output: attention
    """
    def __init__(self):
        super().__init__()
        self.attendEdge = AttendEdgeModule()
        self.transWeit = TransWeightModule()

    def forward(self, attn, cat_matrix, edge_cat_vectors, query, feat):
        """
        Args:
            attn [Tensor] (num_node, ) : attention weight produced by previous module
            cat_matrix [Tensor] (num_node, num_node)
            edge_cat_vectors [Tensor] (num_edge_cat, dim_v)
            query [Tensor] (dim_v, )
        Returns:
            new attention
        """
        weit_matrix = self.attendEdge(edge_cat_vectors, query, cat_matrix)
        out = self.transWeit(attn, weit_matrix)
        return out


class QueryModule(nn.Module):
    """
    1 unified instance: 'query_<cat>'
    output: encoding
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
    4 different instances: 'equal_<cat>', 'equal_integer', 'greater_than' and 'less_than'
    inputs: two encodings from two QueryModule or two CountModule
    output: encoding
    """
    def __init__(self, dim_v):
        super().__init__()
        self.projection = nn.Sequential(
                nn.Linear(dim_v, 256),
                nn.ReLU(),
                nn.Linear(256, dim_v)
            )
        for layer in self.projection:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.constant_(layer.bias, val=0)

    def forward(self, enc1, enc2):
        input = enc1 - enc2
        out = self.projection(input)
        return out

        
