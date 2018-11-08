import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import chain
from . import composite_modules as modules
from .questionEncoder import BiGRUEncoder
from .controller import Controller
from IPython import embed


class XNMNet(nn.Module):
    def __init__(self, **kwargs):
        """
        kwargs:
             vocab,
             dim_v, # vertex and edge embedding of scene graph
             dim_word, # word embedding
             dim_hidden, # hidden of seq2seq
             dim_vision,
             dim_edge,
             glimpses,
             cls_fc_dim,
             dropout_prob,
             T_ctrl,
             stack_len,
             device,
             use_gumbel,
             use_validity,
        """
        super().__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)
        
        self.num_classes = len(self.vocab['answer_token_to_idx'])
        self.classifier = Classifier(
            in_features=(self.glimpses * self.dim_vision, self.dim_hidden),
            mid_features=self.cls_fc_dim,
            out_features=self.num_classes,
            count_features=self.glimpses * self.dim_vision, # count -> module
            mode=self.class_mode,
            drop=self.dropout_prob
            )

        self.visionAttention = VisionAttention(
            v_features=self.dim_vision,
            q_features=self.dim_hidden,
            mid_features=512,
            glimpses=self.glimpses,
            drop=self.dropout_prob,
            )
        self.map_vision_to_v = nn.Sequential(
                nn.Dropout(self.dropout_prob),
                nn.Linear(self.dim_vision, self.dim_v, bias=False),
                )
        self.map_two_v_to_edge = nn.Sequential(
                nn.Dropout(self.dropout_prob),
                nn.Linear(self.dim_v * 2, self.dim_edge, bias=False),
                )
        self.num_token = len(self.vocab['question_token_to_idx'])
        self.token_embedding = nn.Embedding(self.num_token, self.dim_word)
        self.dropout = nn.Dropout(self.dropout_prob)

        # modules
        self.module_names = modules.MODULE_INPUT_NUM.keys()
        self.num_module = len(self.module_names)
        self.module_funcs = [getattr(modules, m[1:]+'Module')(**kwargs) for m in self.module_names]
        self.module_validity_mat = modules._build_module_validity_mat(self.stack_len, self.module_names)
        self.module_validity_mat = torch.Tensor(self.module_validity_mat).to(self.device)
        for name, func in zip(self.module_names, self.module_funcs):
            self.add_module(name, func)
        # question encoder
        self.question_encoder = BiGRUEncoder(self.dim_word, self.dim_hidden)
        # controller
        controller_kwargs = {
            'num_module': len(self.module_names),
            'dim_lstm': self.dim_hidden,
            'T_ctrl': self.T_ctrl,
            'use_gumbel': self.use_gumbel,
        }
        self.controller = Controller(**controller_kwargs)
        
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
        nn.init.normal_(self.token_embedding.weight, mean=0, std=1/np.sqrt(self.dim_word))


    def forward(self, questions, questions_len, vision_feat, relation_mask, debug=False):
        """
        Args:
            questions [Tensor] (batch_size, seq_len)
            questions_len [Tensor] (batch_size)
            vision_feat (batch_size, dim_vision, num_feat)
            relation_mask (batch_size, num_feat, num_feat)
        """
        batch_size = len(questions)
        questions = questions.permute(1, 0) # (seq_len, batch_size)
        questions_embedding = self.token_embedding(questions) # (seq_len, batch_size, dim_word)
        questions_embedding = torch.tanh(self.dropout(questions_embedding))
        questions_outputs, questions_hidden = self.question_encoder(questions, questions_embedding, questions_len)
        module_logits, module_probs, c_list, cv_list = self.controller(
            questions_outputs, questions_hidden, questions_embedding, questions_len)
        ## feature processing
        vision_feat = vision_feat / (vision_feat.norm(p=2, dim=1, keepdim=True) + 1e-12)
        feat_inputs = vision_feat.permute(0,2,1)
        if self.dim_v != self.dim_vision:
            feat_inputs = self.map_vision_to_v(feat_inputs) # (batch_size, num_feat, dim_v)
        num_feat = feat_inputs.size(1)
        feat_inputs_expand_0 = feat_inputs.unsqueeze(1).expand(batch_size, num_feat, num_feat, self.dim_v)
        feat_inputs_expand_1 = feat_inputs.unsqueeze(2).expand(batch_size, num_feat, num_feat, self.dim_v)
        feat_edge = torch.cat([feat_inputs_expand_0, feat_inputs_expand_1], dim=3) # (bs, num_feat, num_feat, 2*dim_v)
        feat_edge = self.map_two_v_to_edge(feat_edge)

        ## stack initialization
        att_stack = torch.zeros(batch_size, num_feat, self.glimpses, self.stack_len).to(self.device)
        stack_ptr = torch.zeros(batch_size, self.stack_len).to(self.device)
        stack_ptr[:, 0] = 1
        mem = torch.zeros(batch_size, self.glimpses * self.dim_vision).to(self.device)

        for t in range(self.T_ctrl):
            c_i = c_list[t] #(batch_size, dim_hidden)
            module_logit = module_logits[t] # (batch_size, num_module)
            if self.use_validity:
                if t < self.T_ctrl-1:
                    module_validity = torch.matmul(stack_ptr, self.module_validity_mat)
                    module_validity[:, 5] = 0
                else: # last step must describe
                    module_validity = torch.zeros(batch_size, self.num_module).to(self.device)
                    module_validity[:, 5] = 1
                module_invalidity = (1 - torch.round(module_validity)).byte() # hard validate
                module_logit.masked_fill_(module_invalidity, -float('inf'))
                module_prob = F.gumbel_softmax(module_logit, hard=self.use_gumbel)
            else:
                module_prob = module_probs[t]
            module_prob = module_prob.permute(1,0) # (num_module, batch_size)

            # run all modules
            res = [f(vision_feat.permute(0,2,1), feat_inputs, feat_edge, c_i, relation_mask, att_stack, stack_ptr, mem) for f in self.module_funcs]
            att_stack_avg = torch.sum(module_prob.view(self.num_module,batch_size,1,1,1) * torch.stack([r[0] for r in res]), dim=0)
            stack_ptr_avg = torch.sum(module_prob.view(self.num_module,batch_size,1) * torch.stack([r[1] for r in res]), dim=0)
            stack_ptr_avg = modules._sharpen_ptr(stack_ptr_avg, hard=False)
            mem_avg = torch.sum(module_prob.view(self.num_module,batch_size,1) * torch.stack([r[2] for r in res]), dim=0)
            att_stack, stack_ptr, mem = att_stack_avg, stack_ptr_avg, mem_avg
            if debug:
                print('%d / %d' % (t, self.T_ctrl))
                print(module_prob)
                embed()
            
        ## Part 1. features from scene graph module network. (batch, dim_v)
        module_outputs = mem
        ## Part 2. question prior. (batch, dim_hidden)
        questions_hidden = questions_hidden
        ## Part 3. bottom-up vision features. (batch, glimpses * dim_vision)
        if vision_feat.dim() == 3:
            vision_feat = vision_feat.unsqueeze(2) # (batch, dim_vision, 1, num_feat)
        if 'v' in self.class_mode:
            a = self.visionAttention(vision_feat, questions_hidden)
            vision_feat = apply_attention(vision_feat, a)
        else:
            vision_feat = None

        ##### final prediction
        predicted_logits = self.classifier(vision_feat, questions_hidden, module_outputs, debug)
        others = {
        }
        return predicted_logits, others



"""
The Classifier and Attention is according to the codes of Learning to Count 
[https://github.com/Cyanogenoid/vqa-counting/blob/master/vqa-v2/model.py]
"""
class Fusion(nn.Module):
    """ Crazy multi-modal fusion: negative squared difference minus relu'd sum
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return - (x - y)**2 + F.relu(x + y)

class Classifier(nn.Sequential):
    def __init__(self, in_features, mid_features, count_features, out_features, mode, drop=0.0):
        super().__init__()
        self.mode = mode
        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU()
        self.fusion = Fusion()
        self.lin11 = nn.Linear(in_features[0], mid_features)
        self.lin12 = nn.Linear(in_features[1], mid_features)
        self.lin2 = nn.Linear(mid_features, out_features)
        self.lin3 = nn.Linear(2*mid_features, mid_features)
        self.lin_c = nn.Linear(count_features, mid_features)
        self.bn = nn.BatchNorm1d(mid_features)
        self.bn2 = nn.BatchNorm1d(mid_features)
        self.elt_gate = nn.Sequential(
                nn.Linear(in_features[1], mid_features),
                nn.ReLU(),
                nn.Linear(mid_features, mid_features),
                nn.Sigmoid()
                )

    def forward(self, x, y, c, debug=False):
        if self.mode == 'c':
            x = self.bn2(self.relu(self.lin_c(self.drop(c))))
        elif self.mode == 'qv':
            x = self.fusion(self.lin11(self.drop(x)), self.lin12(self.drop(y)))
        elif self.mode == 'qc':
            x = self.fusion(self.lin12(self.drop(y)), self.lin_c(self.drop(c)))
        elif self.mode == 'qvc':
            x = self.fusion(self.lin11(self.drop(x)), self.lin12(self.drop(y)))
            c = self.bn2(self.relu(self.lin_c(self.drop(c))))
            g = self.elt_gate(y)
            x = g * x + (1-g) * c
        x = self.lin2(self.drop(self.bn(x)))
        return x



class VisionAttention(nn.Module):
    def __init__(self, v_features, q_features, mid_features, glimpses, drop=0.0):
        super().__init__()
        self.v_conv = nn.Conv2d(v_features, mid_features, 1, bias=False)  # let self.lin take care of bias
        self.q_lin = nn.Linear(q_features, mid_features)
        self.x_conv = nn.Conv2d(mid_features, glimpses, 1) # kernel_size=1

        self.drop = nn.Dropout(drop)
        self.fusion = Fusion()

    def forward(self, v, q):
        # param v: (batch_size, dim_vision, num_vision=36)
        q_in = q
        v = self.v_conv(self.drop(v))
        q = self.q_lin(self.drop(q))
        q = tile_2d_over_nd(q, v)
        x = self.fusion(v, q)
        x = self.x_conv(self.drop(x))
        return x


def apply_attention(input, attention):
    """ Apply any number of attention maps over the input.
        The attention map has to have the same size in all dimensions except dim=1.
    """
    n, c = input.size()[:2]
    glimpses = attention.size(1)

    # flatten the spatial dims into the third dim, since we don't need to care about how they are arranged
    input = input.view(n, c, -1)
    attention = attention.view(n, glimpses, -1)
    s = input.size(2)

    # apply a softmax to each attention map separately
    # since softmax only takes 2d inputs, we have to collapse the first two dimensions together
    # so that each glimpse is normalized separately
    attention = attention.view(n * glimpses, -1)
    attention = F.softmax(attention, dim=1)

    # apply the weighting by creating a new dim to tile both tensors over
    target_size = [n, glimpses, c, s]
    input = input.view(n, 1, c, s).expand(*target_size)
    attention = attention.view(n, glimpses, 1, s).expand(*target_size)
    weighted = input * attention
    # sum over only the spatial dimension
    weighted_mean = weighted.sum(dim=3, keepdim=True)
    # the shape at this point is (n, glimpses, c, 1)
    return weighted_mean.view(n, -1)


def tile_2d_over_nd(feature_vector, feature_map):
    """ Repeat the same feature vector over all spatial positions of a given feature map.
        The feature vector should have the same batch size and number of features as the feature map.
    """
    n, c = feature_vector.size()
    spatial_sizes = feature_map.size()[2:]
    tiled = feature_vector.view(n, c, *([1] * len(spatial_sizes))).expand(n, c, *spatial_sizes)
    return tiled
