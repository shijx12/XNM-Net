import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AndModule(nn.Module):
    def forward(self, attn1, attn2):
        out = torch.min(attn1, attn2)
        return out


class OrModule(nn.Module):
    def forward(self, attn1, attn2):
        out = torch.max(attn1, attn2)
        return out


class NotModule(nn.Module):
    def forward(self, attn):
        out = 1.0 - attn
        return out


class AttendNodeModule(nn.Module):
    def forward(self, node_vectors, query):
        """
        Args:
            node_vectors [Tensor] (num_node, dim_v) : node feature vectors
            query [Tensor] (dim_v, ) : query vector
        Returns:
            attn [Tensor] (num_node, ): node attention by *SOFTMAX*. Actually it is attribute value attention, because we regard attribute values as nodes
        """
        logit = torch.matmul(node_vectors, query)
        attn = F.softmax(logit, dim=0)
        return attn


class AttendEdgeModule(nn.Module):
    def forward(self, edge_cat_vectors, query, cat_matrix):
        """
        Args:
            edge_cat_vectors [Tensor] (num_edge_cat, dim_v): Vectors of edge categories, such as 'left', 'right', 'color', 'shape' and etc.
            query [Tensor] (dim_v, )
            cat_matrix [LongTensor] (num_node, num_node, num_rel) : cat_matrix[i][j] is the indexes of relationship category between node i and j. num_rel is the maximum number of relationships between each pair.
        Returns:
            attn [Tensor] (num_node, ): Edge attention weights
        """
        logit = torch.matmul(edge_cat_vectors, query)
        cat_attn = F.softmax(logit, dim=0) # (num_edge_cat, )
        # cat 0 means no edge, so assign zero to its weight and renormalize cat_attn
        mask = torch.ones(logit.size(0)).to(query.device)
        mask[0] = 0
        cat_attn = cat_attn * mask
        cat_attn /= cat_attn.sum()

        num_node, num_rel = cat_matrix.size(0), cat_matrix.size(2)
        cat_attn = cat_attn.view(-1, 1, 1).expand(-1, num_node, num_rel) # (num_edge_cat, num_node, num_rel)
        attn = torch.gather(cat_attn, dim=0, index=cat_matrix) # (num_node, num_node, num_rel)
        attn = torch.sum(attn, dim=2) # (num_node, num_node)
        return attn


class TransferModule(nn.Module):
    def forward(self, node_attn, edge_attn):
        """
        Transfer node attention along with activated edges (transfer between different objects) or adjacent matrix (transfer from attribute value nodes to object nodes)
        Args:
            node_attn [Tensor] (num_node, )
            edge_attn [Tensor] (num_node, num_node) : Weighted adjacent matrix produced by edge attention
        Returns:
            new_attn [Tensor] (num_node, )
        """
        new_attn = torch.matmul(node_attn, edge_attn.float())
        return new_attn

