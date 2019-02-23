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
    def forward(self, node_feat, query):
        """
        Args:
            node_feat [Tensor] (num_node, dim_v)
            query [Tensor] (dim_v, )
        Returns:
            attn [Tensor] (num_node, ): node attention by a *SIGMOID* layer
        """
        logit = torch.matmul(node_feat, query)
        attn = F.sigmoid(logit)
        return attn


class AttendEdgeModule(nn.Module):
    def forward(self, edge_feat, query):
        """
        Args:
            edge_feat [Tensor] (num_node, num_node, dim_v)
            query [Tensor] (dim_v, )
        Returns:
            attn [Tensor] (num_node, num_node): edge attention by a *SIGMOID* layer
        """
        query = query.view(1, 1, -1).expand_as(edge_feat)
        logit = torch.sum(edge_feat * query, dim=2) # (n, n)
        attn = F.sigmoid(logit)
        return attn


class TransferModule(nn.Module):
    def forward(self, node_attn, edge_attn):
        """
        Args:
            node_attn [Tensor] (num_node, )
            edge_attn [Tensor] (num_node, num_node) : Weighted adjacent matrix produced by edge attention.
        Returns:
            new_attn [Tensor] (num_node, )
        """
        new_attn = torch.matmul(node_attn, edge_attn)
        return new_attn

