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
    def __init__(self, **kwargs):
        super().__init__()
        self.dim_v = kwargs['dim_v']
        self.attendNode = AttendNodeModule()
        self.attnAnd = AndModule()

    def forward(self, attn, feat, query): # value query
        """
        Args:
            attn [Tensor] (num_node, ) : attention weight produced by previous module
            conn_matrix [Tensor] (num_node, num_node)
            feat [Tensor] (num_node, dim_v)
            query [Tensor] (dim_v, )
        Returns:
            attention
        """
        new_object_attn = self.attendNode(feat, query)
        out = self.attnAnd(attn, new_object_attn)
        return out


class SameModule(nn.Module):
    """
    1 unified instance: 'same_<cat>'. 
    value_inputs: <cat> including 'color','shape','material' and 'size'
    output: attention
    """

    def __init__(self, **kwargs):
        super().__init__()
        dim_v = kwargs['dim_v']
        self.query = QueryModule(**kwargs)
        self.attend = AttentionModule(**kwargs)
        self.attnNot = NotModule()
        
    def forward(self, attn, feat, query):
        value_query = self.query(attn, feat, query) # map cat query to value query
        # exclude current object
        out = self.attend(self.attnNot(attn), feat, value_query)
        return out



class ExistOrCountModule(nn.Module):
    """
    2 instance: 'exist', 'count'
    input: attention
    output: encoding
    """
    def __init__(self, **kwargs):
        super().__init__()
        dim_v = kwargs['dim_v']
        self.projection = nn.Sequential(
                nn.Linear(1, 128),
                nn.ReLU(),
                nn.Linear(128, dim_v)
                )
        for layer in self.projection:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.constant_(layer.bias, val=0)

    def forward(self, attn, feat, query):
        #attn = F.sigmoid((attn-0.5)/0.01) # squash function
        #mask = attn.ge(0.5).float()
        #attn = mask * attn
        # ------------------

        out = self.projection(torch.sum(attn, dim=0, keepdim=True)) 
        return out



class RelateModule(nn.Module):
    """
    3 instances: 'relate_<cat>'
    value_inputs: <cat>
    output: attention
    """
    def __init__(self, **kwargs):
        super().__init__()
        if kwargs['edge_class'] == 'learncat':
            self.attendEdge = AttendLearnedEdgeModule()
        elif kwargs['edge_class'] == 'dense':
            self.attendEdge = AttendDenseEdgeModule()
        else:
            self.attendEdge = AttendEdgeModule()
        self.transWeit = TransWeightModule()

    def forward(self, attn, feat, query, edge_cat_vectors, cat_matrix):
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
    def __init__(self, **kwargs):
        super().__init__()
        dim_v = kwargs['dim_v']
        K = kwargs['k_attr']  # given attribute category number
        self.query_to_weight = nn.Sequential(
            nn.Linear(dim_v, K),
            nn.Softmax(dim=0),
                )
        self.mappings = nn.Parameter(torch.zeros((K, dim_v, dim_v)))
        nn.init.normal_(self.mappings.data, mean=0, std=0.01)
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, attn, feat, query):
        out = torch.matmul(attn, feat)
        weight = self.query_to_weight(query)
        mapping = torch.sum(self.mappings * weight.view(-1,1,1), dim=0) # (dim_v, dim_v)
        out = torch.matmul(mapping, out)
        return out # (dim_v, )


class ComparisonModule(nn.Module):
    """
    4 different instances: 'equal_<cat>', 'equal_integer', 'greater_than' and 'less_than'
    inputs: two encodings from two QueryModule or two CountModule
    output: encoding
    """
    def __init__(self, **kwargs):
        super().__init__()
        dim_v = kwargs['dim_v']
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

        
