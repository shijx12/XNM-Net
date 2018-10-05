import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AndModule(nn.Module):
    """
    1 unified instance: 'intersect'
    inputs: two node attentions
    output: attention
    """
    def forward(self, attn1, attn2):
        out = torch.min(attn1, attn2)
        return out


class OrModule(nn.Module):
    """ A neural module that (basically) performs a logical or on two node attention weights.

    Extended Summary
    ----------------
    An :class:`OrModule` is a neural module that takes two input attention masks and (basically)
    performs a set union. This would be used in a question like "How many cubes are left of the
    brown sphere or right of the cylinder?" After localizing the regions left of the brown sphere
    and right of the cylinder, an :class:`OrModule` would be used to find the union of the two. Its
    output would then go into an :class:`AttentionModule` that finds cubes.
    """
    def forward(self, attn1, attn2):
        out = torch.max(attn1, attn2)
        return out


class NotModule(nn.Module):
    """ A neural module that (basically) performs a logical not on a node attention weight
    """
    def forward(self, attn):
        out = 1.0 - attn
        return out


class AttendNodeModule(nn.Module):
    """ A neural module that (basically) attends graph nodes based on a query.

    """
    def forward(self, node_vectors, query):
        """
        Args:
            node_vectors [Tensor] (num_node, dim_v)
            query [Tensor] (dim_v, )
        Returns:
            attn [Tensor] (num_node, ): node attentions by a softmax layer
        """
        assert node_vectors.size(1) == query.size(0)
        logit = torch.matmul(node_vectors, query)
        #attn = F.sigmoid(logit)
        attn = F.softmax(logit, dim=0)

#        idx = torch.argmax(attn, dim=0)
#        num_node = node_vectors.size(0)
#        gate = np.asarray(np.arange(num_node)==idx, dtype=np.int32)
#        gate = torch.Tensor(gate).to(query.device).float()
#        attn = (gate - attn).detach() + attn

        return attn


class AttendEdgeModule(nn.Module):
    def forward(self, edge_vectors, query):
        # edge_vectors: (n, n, dim)
        # query: (dim, )
        query = query.view(1, 1, -1).expand_as(edge_vectors)
        logit = torch.sum(edge_vectors * query, dim=2) # (n, n)
        attn = F.sigmoid(logit)
        return attn



class TransWeightModule(nn.Module):
    """ A neural module that transforms the node attention vector according to a weighted adjacent matrix, which is obtained by edge attention

    """
    def forward(self, node_attn, weit_matrix):
        """
        Args:
            node_attn [Tensor] (num_node, )
            weit_matrix [Tensor] (num_node, num_node) : Weighted adjacent matrix produced by edge attention.
        Returns:
            new_attn [Tensor] (num_node, )
        """
        new_attn = torch.matmul(node_attn, weit_matrix)
        return new_attn

