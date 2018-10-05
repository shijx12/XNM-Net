import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
from .basic_modules import *
from utils.misc import convert_to_one_hot
from IPython import embed

MODULE_INPUT_NUM = {
    '_NoOp': 0,
    '_Find': 0,
    '_Transform': 1,
    '_Filter': 1,
    '_And': 2,
    '_Describe': 1,
}

MODULE_OUTPUT_NUM = {
    '_NoOp': 0,
    '_Find': 1,
    '_Transform': 1,
    '_Filter': 1,
    '_And': 1,
    '_Describe': 1,
}

"""
Note that batch_size=1 in all of my modules.
att_stack: (batch_size, dim_att(i.e., num_vertex), stack_len)
att: (batch_size, dim_att, 1)
stack_ptr: (batch_size, stack_len)
mem: (batch_size, dim_v)
"""

class NoOpModule(nn.Module):
    def __init__(self, dim_v):
        super().__init__()

    def forward(self, cat_matrix, conn_matrix, edge_cat_vectors, feat, c_i, att_stack, stack_ptr, mem_in):
        return att_stack, stack_ptr, mem_in


class AndModule(nn.Module):
    def __init__(self, dim_v):
        super().__init__()
        self.mem_zero = torch.zeros(1, dim_v)

    def forward(self, cat_matrix, conn_matrix, edge_cat_vectors, feat, c_i, att_stack, stack_ptr, mem_in):
        # feat (num_vertex, dim_v): vertex embedding
        # c_i (dim_v): query
        att2 = _read_from_stack(att_stack, stack_ptr)
        stack_ptr = _move_ptr_bw(stack_ptr)
        att1 = _read_from_stack(att_stack, stack_ptr)
        att_out = torch.min(att1, att2)
        att_stack = _write_to_stack(att_stack, stack_ptr, att_out)
        return att_stack, stack_ptr, self.mem_zero.to(feat.device)


class FindModule(nn.Module):

    def __init__(self, dim_v):
        super().__init__()
        self.attendNode = AttendNodeModule()
        self.transConn = TransConnectModule()
        self.mem_zero = torch.zeros(1, dim_v)

    def forward(self, cat_matrix, conn_matrix, edge_cat_vectors, feat, c_i, att_stack, stack_ptr, mem_in):
        stack_ptr = _move_ptr_fw(stack_ptr)
        attribute_att = self.attendNode(feat, c_i)
        att_out = self.transConn(attribute_att, conn_matrix) # (att_dim, )
        att_out = att_out.view(1, -1, 1) #(batch_size, att_dim, 1)
        att_stack = _write_to_stack(att_stack, stack_ptr, att_out)
        return att_stack, stack_ptr, self.mem_zero.to(feat.device)


class FilterModule(nn.Module):
    def __init__(self, dim_v):
        super().__init__()
        self.Find = FindModule(dim_v)
        self.And = AndModule(dim_v)
        self.mem_zero = torch.zeros(1, dim_v)

    def forward(self, cat_matrix, conn_matrix, edge_cat_vectors, feat, c_i, att_stack, stack_ptr, mem_in):
        att_stack, stack_ptr, _ = self.Find(cat_matrix, conn_matrix, edge_cat_vectors, feat, c_i, att_stack, stack_ptr, mem_in)
        att_stack, stack_ptr, _ = self.And(cat_matrix, conn_matrix, edge_cat_vectors, feat, c_i, att_stack, stack_ptr, mem_in)
        return att_stack, stack_ptr, self.mem_zero.to(feat.device)



class TransformModule(nn.Module):

    def __init__(self, dim_v):
        super().__init__()
        self.attendEdge = AttendEdgeModule()
        self.transWeit = TransWeightModule()
        self.mem_zero = torch.zeros(1, dim_v)

    def forward(self, cat_matrix, conn_matrix, edge_cat_vectors, feat, c_i, att_stack, stack_ptr, mem_in):
        att_in = _read_from_stack(att_stack, stack_ptr)
        weit_matrix = self.attendEdge(edge_cat_vectors, c_i, cat_matrix)
        att_out = self.transWeit(att_in.view(1,-1), weit_matrix)
        att_out = att_out.view(1, -1, 1) #(batch_size, att_dim, 1)
        att_stack = _write_to_stack(att_stack, stack_ptr, att_out)
        return att_stack, stack_ptr, self.mem_zero.to(feat.device)

class _MutedTransformModule(nn.Module):

    def __init__(self, dim_v):
        super().__init__()
        self.attendEdge = AttendEdgeModule()
        self.transWeit = TransWeightModule()

    def forward(self, cat_matrix, conn_matrix, edge_cat_vectors, feat, c_i, att_stack, stack_ptr, mem_in):
        att_in = _read_from_stack(att_stack, stack_ptr)
        weit_matrix = self.attendEdge(edge_cat_vectors, c_i, cat_matrix)
        att_out = self.transWeit(att_in.view(1,-1), weit_matrix)
        return att_out

class DescribeModule(nn.Module):

    def __init__(self, dim_v):
        super().__init__()
        self.projection = nn.Sequential(
                nn.Linear(1, 128),
                nn.ReLU(),
                nn.Linear(128, dim_v)
                )
        self.fc_gate = nn.Sequential(
                nn.Linear(dim_v, 3),
                nn.Softmax(dim=0) # c_i is (dim_v,)
            )
        for layer in chain(self.projection, self.fc_gate):
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.constant_(layer.bias, val=0)
        self.transform = _MutedTransformModule(dim_v)

    def forward(self, cat_matrix, conn_matrix, edge_cat_vectors, feat, c_i, att_stack, stack_ptr, mem_in):
        att_in = _read_from_stack(att_stack, stack_ptr).view(1,-1)
        mem_out_1 = self.projection(torch.sum(att_in, dim=1, keepdim=True))
        attribute_att = self.transform(cat_matrix, conn_matrix, edge_cat_vectors, feat, c_i, att_stack, stack_ptr, mem_in)
        mem_out_2 = torch.matmul(attribute_att, feat)
        att_stack = _write_to_stack(att_stack, stack_ptr, torch.zeros(1,feat.size(0),1).to(feat.device))

        out_weight = self.fc_gate(c_i) # (3,)
        out_cat = torch.cat([mem_in, mem_out_1, mem_out_2], dim=0) # (3, dim_v)
        mem_out = torch.matmul(out_weight, out_cat).view(1,-1) # (1, dim_v)
        return att_stack, stack_ptr, mem_out










def _move_ptr_fw(stack_ptr):
    """
    Move the stack pointer forward (i.e. to push to stack).
    stack_ptr: (batch_size, stack_len)
    Return: (batch_size, stack_len)
    """
    filter_fw = torch.FloatTensor([1, 0, 0]).view(1, 1, 3).to(stack_ptr.device)
    batch_size, stack_len = stack_ptr.size()
    new_stack_ptr = F.conv1d(stack_ptr.view(batch_size, 1, stack_len), filter_fw, padding=1).view(batch_size, stack_len)
    # when the stack pointer is already at the stack top, keep
    # the pointer in the same location (otherwise the pointer will be all zero)
    stack_top_mask = torch.zeros(stack_len).to(stack_ptr.device)
    stack_top_mask[stack_len - 1] = 1 # [stack_len, ]
    new_stack_ptr += stack_top_mask * stack_ptr
    return new_stack_ptr


def _move_ptr_bw(stack_ptr):
    """
    Move the stack pointer backward (i.e. to pop from stack).
    """
    filter_fw = torch.FloatTensor([0, 0, 1]).view(1, 1, 3).to(stack_ptr.device)
    batch_size, stack_len = stack_ptr.size()
    new_stack_ptr = F.conv1d(stack_ptr.view(batch_size, 1, stack_len), filter_fw, padding=1).view(batch_size, stack_len)
    # when the stack pointer is already at the stack bottom, keep
    # the pointer in the same location (otherwise the pointer will be all zero)
    stack_bottom_mask = torch.zeros(stack_len).to(stack_ptr.device)
    stack_bottom_mask[0] = 1
    new_stack_ptr += stack_bottom_mask * stack_ptr
    return new_stack_ptr


def _read_from_stack(att_stack, stack_ptr):
    """
    Read the value at the given stack pointer.
    """
    batch_size, stack_len = stack_ptr.size()
    stack_ptr_expand = stack_ptr.view(batch_size, 1, stack_len)
    # The stack pointer is a one-hot vector, so just do dot product
    att = torch.sum(att_stack * stack_ptr_expand, dim=-1, keepdim=True)
    return att # (batch_size, att_dim, 1)


def _write_to_stack(att_stack, stack_ptr, att):
    """
    Write value 'att' into the stack at the given stack pointer. Note that the
    result needs to be assigned back to att_stack
    """
    batch_size, stack_len = stack_ptr.size()
    stack_ptr_expand = stack_ptr.view(batch_size, 1, stack_len)
    att_stack = att * stack_ptr_expand + att_stack * (1 - stack_ptr_expand)
    return att_stack # (batch_size, att_dim, stack_len)


def _sharpen_ptr(stack_ptr, hard):
    """
    Sharpen the stack pointers into (nearly) one-hot vectors, using argmax
    or softmax. The stack values should always sum up to one for each instance.
    """
    if hard:
        # hard (non-differentiable) sharpening with argmax
        stack_len = stack_ptr.size(1)
        new_stack_ptr_indices = torch.argmax(stack_ptr, dim=1)[1]
        new_stack_ptr = convert_to_one_hot(new_stack_ptr_indices, stack_len)
    else:
        # soft (differentiable) sharpening with softmax
        temperature = 10
        new_stack_ptr = F.softmax(stack_ptr / temperature)
    return new_stack_ptr


def _build_module_validity_mat(stack_len, module_names):
    """
    Build a module validity matrix, ensuring that only valid modules will have
    non-zero probabilities. A module is only valid to run if there are enough
    attentions to be popped from the stack, and have space to push into
    (e.g. _Find), so that stack will not underflow or overflow by design.

    module_validity_mat is a stack_len x num_module matrix, and is used to
    multiply with stack_ptr to get validity boolean vector for the modules.
    """

    module_validity_mat = np.zeros((stack_len, len(module_names)), np.float32)
    for n_m, m in enumerate(module_names):
        # a module can be run only when stack ptr position satisfies
        # (min_ptr_pos <= ptr <= max_ptr_pos), where max_ptr_pos is inclusive
        # 1) minimum position:
        #    stack need to have MODULE_INPUT_NUM[m] things to pop from
        min_ptr_pos = MODULE_INPUT_NUM[m]
        # the stack ptr diff=(MODULE_OUTPUT_NUM[m] - MODULE_INPUT_NUM[m])
        # ensure that ptr + diff <= stack_len - 1 (stack top)
        max_ptr_pos = (
            stack_len - 1 + MODULE_INPUT_NUM[m] - MODULE_OUTPUT_NUM[m])
        module_validity_mat[min_ptr_pos:max_ptr_pos+1, n_m] = 1.

    return module_validity_mat
