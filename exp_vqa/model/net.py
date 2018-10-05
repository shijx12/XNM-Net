import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from itertools import chain

from . import basic_modules
from . import composite_modules as modules
from .AttentionSeq2Seq import EncoderRNN
from IPython import embed


class XNMNet(nn.Module):
    def __init__(self, **kwargs):
        """         
            vocab,
            dim_v, # word, vertex and edge embedding of scene graph
            dim_hidden, # hidden of seq2seq
            dim_vision,
            dropout_prob,
            device,
            program_scheme,
            cls_fc_dim
        Initializes a XNMNet object.
        """
        super().__init__()
        for k,v in kwargs.items():
            setattr(self, k, v)

        print('program scheme: %s' % ('->'.join(self.program_scheme)))

        
        self.num_classes = len(self.vocab['answer_token_to_idx'])
        glimpses = 2
        self.classifier = Classifier(
            in_features=(glimpses * self.dim_vision, self.dim_hidden),
            mid_features=self.cls_fc_dim,
            out_features=self.num_classes,
            count_features=self.dim_v, # count -> module
            mode=self.class_mode,
            drop=self.dropout_prob
            )

        self.visionAttention = VisionAttention(
            v_features=self.dim_vision,
            q_features=self.dim_hidden,
            mid_features=512,
            glimpses=glimpses,
            drop=self.dropout_prob,
            )
        self.map_word_to_v = nn.Sequential(
                nn.Linear(self.dim_word, self.dim_v),
                nn.ReLU(),
                )
        self.map_vision_to_v = nn.Sequential(
                nn.Linear(self.dim_vision, self.dim_v),
                nn.Tanh(),
                )
        self.map_two_edge_to_v = nn.Sequential(
                nn.Linear(self.dim_v * 2, 1024),
                nn.ReLU(),
                nn.Linear(1024, self.dim_v),
                nn.Tanh(),
                )
        # question+attribute+relation tokens. 0 for <NULL>
        self.num_token = len(self.vocab['question_token_to_idx'])
        self.token_embedding = nn.Embedding(self.num_token, self.dim_word)
        self.dropout = nn.Dropout(self.dropout_prob)

        # modules
        self.function_modules = {}
        for module_name in {'find', 'describe', 'relate'}:
            if module_name == 'find':
                module = modules.FindModule(**kwargs)
            elif module_name == 'relate':
                module = modules.RelateModule(**kwargs)
            elif module_name == 'describe':
                module = modules.DescribeModule(**kwargs)

            self.function_modules[module_name] = module
            self.add_module(module_name, module)

        self.question_encoder = EncoderRNN(self.num_token, self.dim_hidden, self.dim_word)
        self.att_func_list = []
        for i in range(3):
            self.att_func_list.append(
                    nn.Sequential(
                        nn.Linear(self.dim_word, 256),
                        nn.ReLU(),
                        nn.Linear(256, 1)
                        )
                    )
            self.add_module('att_func_%d'%i, self.att_func_list[i])
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
        nn.init.normal_(self.token_embedding.weight, mean=0, std=1/np.sqrt(self.dim_word))


    def forward(self, questions, questions_len, vision_feat):
        """
        Args:
            questions [Tensor] (batch_size, seq_len)
            questions_len [Tensor] (batch_size)
        """
        batch_size = len(questions)
        questions = questions.permute(1, 0) # (seq_len, batch_size)
        gt_programs = torch.zeros(batch_size, len(self.program_scheme)+1).to(self.device)
        for i, m in enumerate(self.program_scheme):
            gt_programs[:,i] = self.vocab['program_token_to_idx'][m]
        self.token_embedding.weight[0].data.zero_()
        questions_embedding = self.token_embedding(questions) # (seq_len, batch_size, dim_word)
        questions_embedding = torch.tanh(self.dropout(questions_embedding))
        feat_inputs = self.map_vision_to_v(vision_feat.permute(0,2,1))

        module_outputs = []
        for n in range(batch_size):
            feat_input = feat_inputs[n]
            num_node = len(feat_input)
            feat_input_expand_0 = feat_input.unsqueeze(0).expand(num_node, num_node, self.dim_v)
            feat_input_expand_1 = feat_input.unsqueeze(1).expand(num_node, num_node, self.dim_v)
            feat_edge = torch.cat([feat_input_expand_0, feat_input_expand_1], dim=2) # (num_node, num_node, 2*dim_v)
            feat_edge = self.map_two_edge_to_v(feat_edge)

            output = torch.ones(num_node).to(self.device)
            for i in range(len(gt_programs[n])): # NOTE: programs are reverted pre-order
                module_type = self.vocab['program_idx_to_token'][gt_programs[n, i].item()]
                if module_type in {'<eos>'}: # quit 
                    break
                #### module input
                question_logit = self.att_func_list[i](questions_embedding[:,n,:]).squeeze()
                mask = np.ones(question_logit.size(0))
                mask[0:questions_len[n]] = 0
                mask_tensor = torch.ByteTensor(mask).to(self.device)
                question_logit.data.masked_fill_(mask_tensor, -float('inf'))
                question_att = nn.functional.softmax(question_logit, dim=0) #(seq_len, )
                query = torch.matmul(question_att, questions_embedding[:,n,:]) #(dim_word, )
                query = self.map_word_to_v(query) #(dim_v, )
                ####

                module = self.function_modules[module_type]
                if module_type == 'relate':
                    output = module(output, feat_edge, query)
                else:
                    output = module(output, feat_input, query)
            module_outputs.append(output)
        
        ## Part 1. features from scene graph module network. (batch, dim_v)
        module_outputs = torch.stack(module_outputs)
        ## Part 2. question prior. (batch, dim_hidden)
        questions_hidden = self.question_encoder(questions, questions_embedding, questions_len)
        ## Part 3. bottom-up vision features. (batch, glimpses * dim_vision)
        if vision_feat.dim() == 3:
            vision_feat = vision_feat.unsqueeze(2) # (batch, dim_vision, 1, num_feat)
        vision_feat = vision_feat / (vision_feat.norm(p=2, dim=1, keepdim=True) + 1e-12)
        a = self.visionAttention(vision_feat, questions_hidden)
        vision_feat = apply_attention(vision_feat, a)

        ##### final prediction
        predicted_logits = self.classifier(vision_feat, questions_hidden, module_outputs)
        others = {
        }
        return predicted_logits, others


    def forward_and_return_intermediates(self, questions, questions_len, conn_matrixes, cat_matrixes, v_indexes, e_indexes, vision_feat):

        assert len(v_indexes) == 1, 'only support intermediates of batch size 1'
        intermediates = []

        batch_size = len(v_indexes)
        questions = questions.permute(1, 0) # (seq_len, batch_size)
        gt_programs = torch.zeros(batch_size, len(self.program_scheme)+1).to(self.device)
        for i, m in enumerate(self.program_scheme):
            gt_programs[:,i] = self.vocab['program_token_to_idx'][m]
        questions_embedding = self.token_embedding(questions)
        questions_embedding = torch.tanh(self.dropout(questions_embedding))

        n = 0
        edge_cat_vectors = self.map_word_to_v(self.token_embedding(e_indexes[n])) # (num_current_edge, dim_v)
        output = None
        feat_input = self.map_word_to_v(self.token_embedding(v_indexes[n]))
        for i in range(len(gt_programs[n])): # NOTE: programs are reverted pre-order
            module_type = self.vocab['program_idx_to_token'][gt_programs[n, i].item()]
            if module_type in {'<eos>'}:
                break  # quit 
            question_logit = self.att_func_list[i](questions_embedding[:,n,:]).squeeze()
            mask = np.ones(question_logit.size(0))
            mask[0:questions_len[n]] = 0
            mask_tensor = torch.ByteTensor(mask).to(self.device)
            question_logit.data.masked_fill_(mask_tensor, -float('inf'))
            question_att = nn.functional.softmax(question_logit, dim=0) #(seq_len, )
            query = torch.matmul(question_att, questions_embedding[:,n,:]) #(dim_word, )
            query = self.map_word_to_v(query) #(dim_v, )

            module = self.function_modules[module_type]
            if module_type == 'describe':
                output = module(output, query, feat_input)
            elif module_type == 'find':
                output = module(conn_matrixes[n], feat_input, query)
            elif module_type == 'relate':
                output = module(output, cat_matrixes[n], edge_cat_vectors, query)

            question_attention_str = ''
            for j in range(questions_len[n].item()):
                word = self.vocab['question_idx_to_token'][questions[j,n].item()]
                weight = question_att[j].item()
                question_attention_str += '%s-%.2f ' % (word, weight)
            vertex_weight =None
            if module_type in {'find', 'relate'}:
                vertex_weight = output.data.cpu().numpy()
            intermediates.append((
                    module_type + ':::' + question_attention_str, # question attention heatmap
                    vertex_weight, # vertex attention vector
                ))

        ## Part 1. features from scene graph module network. (batch, dim_v)
        module_outputs = output.view(1, -1)
        ## Part 2. question prior. (batch, dim_hidden)
        questions_hidden = self.question_encoder(questions, questions_embedding, questions_len)
        ## Part 3. bottom-up vision features. (batch, glimpses * dim_vision)
        if vision_feat.dim() == 3:
            vision_feat = vision_feat.unsqueeze(2) # (batch, dim_vision, 1, num_feat)
        vision_feat = vision_feat / (vision_feat.norm(p=2, dim=1, keepdim=True) + 1e-12)
        a = self.visionAttention(vision_feat, questions_hidden)
        vision_feat = apply_attention(vision_feat, a)

        ##### final prediction
        predicted_logits = self.classifier(vision_feat, questions_hidden, module_outputs)
        pred = predicted_logits.squeeze().max(0)[1]
        return (self.vocab['answer_idx_to_token'][pred.item()], intermediates)




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
        # found through grad student descent ;)
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
        self.lin_c = nn.Linear(count_features, mid_features)
        self.bn = nn.BatchNorm1d(mid_features)
        self.bn2 = nn.BatchNorm1d(mid_features)

    def forward(self, x, y, c):
        if self.mode == 'c':
            x = self.bn2(self.relu(self.lin_c(c)))
        elif self.mode == 'qv':
            x = self.fusion(self.lin11(self.drop(x)), self.lin12(self.drop(y)))
        elif self.mode == 'qvc':
            x = self.fusion(self.lin11(self.drop(x)), self.lin12(self.drop(y)))
            x = x + self.bn2(self.relu(self.lin_c(c)))
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
