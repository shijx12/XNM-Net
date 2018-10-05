import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
        attn = F.softmax(logit, dim=0)

#        idx = torch.argmax(attn, dim=0)
#        num_node = node_vectors.size(0)
#        gate = np.asarray(np.arange(num_node)==idx, dtype=np.int32)
#        gate = torch.Tensor(gate).to(query.device).float()
#        attn = (gate - attn).detach() + attn

        return attn


class AttendEdgeModule(nn.Module):
    """ A neural module that (basically) attends graph edge based on a query.

    """
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
        #cat_attn = F.softmax(logit, dim=0) # (num_edge_cat, )
        cat_attn = F.softmax(logit/0.01, dim=0) # (num_edge_cat, )

        # cat 0 means no edge, so assign zero to its weight and renormalize cat_attn
        mask = torch.ones(logit.size(0)).to(query.device)
        mask[0] = 0
        cat_attn = cat_attn * mask
        cat_attn /= cat_attn.sum()

        # -----------
        num_node, num_rel = cat_matrix.size(0), cat_matrix.size(2)
        cat_attn = cat_attn.view(-1, 1, 1).expand(-1, num_node, num_rel) # (num_edge_cat, num_node, num_rel)
        attn = torch.gather(cat_attn, dim=0, index=cat_matrix) # (num_node, num_node, num_rel)
        attn = torch.max(attn, dim=2)[0] # (num_node, num_node)
        return attn


class TransConnectModule(nn.Module):
    """ A neural module that transforms the node attention vector according to a connectivity matrix

    """
    def forward(self, node_attn, conn_matrix):
        """
        Args:
            node_attn [Tensor] (num_node, )
            conn_matrix [Tensor] (num_node, num_node) : Connectivity adjacent matrix, whose [i][j]==1 iff. there is an edge between node i and j.
        Returns:
            new_attn [Tensor] (num_node, )
        """
        new_attn = torch.matmul(node_attn, conn_matrix.float())
        return new_attn


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

