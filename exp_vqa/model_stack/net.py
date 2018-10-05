import torch
import torch.nn as nn
import numpy as np
from itertools import chain
from . import modules
from .questionEncoder import Encoder
from .controller import Controller
from IPython import embed


class TbDNet(nn.Module):
    def __init__(self, **kwargs):
        """
        kwargs:
             vocab,
             dim_v, # vertex and edge embedding of scene graph
             dim_word, # word embedding
             dim_hidden, # hidden of seq2seq
             cls_fc_dim,
             dropout,
             T_ctrl,
             stack_len,
             device,
             use_gumbel,
             use_validity,
        Initializes a TbDNet object.

        Parameters
        ----------
        vocab : Dict[str, Dict[Any, Any]]
            The vocabulary holds dictionaries that provide handles to various objects. Valid keys 
            into vocab are
            - 'answer_idx_to_token' whose keys are ints and values strings
            - 'answer_token_to_idx' whose keys are strings and values ints
            - 'program_idx_to_token' whose keys are ints and values strings
            - 'program_token_to_idx' whose keys are strings and values ints
            - 'edge_token_to_idx' whose keys are strings and values ints
            - 'edge_idx_to_token' whose keys are ints and values strings
            These value dictionaries provide retrieval of an answer word or program token from an
            index, or an index from a word or program token.
        """
        super().__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)
        
        self.num_token = len(self.vocab['question_token_to_idx'])
        
        # The classifier takes the output of the last module
        # and produces a distribution over answers
        self.num_classes = len(self.vocab['answer_token_to_idx'])
        self.classifier = nn.Sequential(
            nn.Linear(self.dim_v+self.dim_hidden, self.cls_fc_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.cls_fc_dim, self.num_classes)  # no softmax here
            )
        # edge embedding
        num_edge_cat = len(self.vocab['edge_token_to_idx'])
        self.edge_cat_vectors = nn.Parameter(torch.Tensor(num_edge_cat, self.dim_v))
        
        # map node index to corresponding embedding of dim_v
        # 0 for object nodes, others for attribute nodes TODO fix 0?
        self.token_embedding = nn.Embedding(self.num_token, self.dim_v)
        # map word embedding to a query for attention
        self.map_ci_to_query = nn.Sequential(
            nn.Linear(self.dim_hidden, self.dim_v),
            nn.ReLU()
            )

        # modules
        self.module_names = modules.MODULE_INPUT_NUM.keys()
        self.num_module = len(self.module_names)
        self.module_funcs = [getattr(modules, m[1:]+'Module')(self.dim_v) for m in self.module_names]
        self.module_validity_mat = modules._build_module_validity_mat(self.stack_len, self.module_names)
        self.module_validity_mat = torch.Tensor(self.module_validity_mat).to(self.device)
        for name, func in zip(self.module_names, self.module_funcs):
            self.add_module(name, func)
        # question encoder
        encoder_kwargs = {
            'num_vocab': len(self.vocab['question_token_to_idx']), 
            'word_dim': self.dim_word,
            'lstm_dim': self.dim_hidden
            }
        self.question_encoder = Encoder(**encoder_kwargs)
        # controller
        controller_kwargs = {
            'num_module': len(self.module_names),
            'dim_lstm': self.dim_hidden,
            'T_ctrl': self.T_ctrl,
            'use_gumbel': self.use_gumbel,
        }
        self.controller = Controller(**controller_kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in chain(self.classifier, self.map_ci_to_query):
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.constant_(layer.bias, val=0)
        nn.init.normal_(self.edge_cat_vectors.data, mean=0, std=1/np.sqrt(self.dim_v))
        nn.init.normal_(self.attribute_embedding.weight, mean=0, std=1/np.sqrt(self.dim_v))

    def forward(self, questions, questions_len, conn_matrixes, cat_matrixes, v_indexes):
        """
        Args:
            conn_matrixes [list of Tensor]
            cat_matrixes [list of Tensor]
            v_indexes [list of Tensor], each Tensor is (num_node, ) : node index of each graph
            questions [Tensor] (batch_size, seq_len)
            questions_len [Tensor] (batch_size)
        """
        batch_size = len(v_indexes)
        questions = questions.permute(1, 0) # (seq_len, batch_size)
        question_outputs, question_hidden, question_wordemb = self.question_encoder(questions, questions_len)
        module_logits, module_probs, c_list, cv_list = self.controller(
            question_outputs, question_hidden, question_wordemb, questions_len)

        final_module_outputs = []
        for n in range(batch_size):
            feat_input = self.attribute_embedding(v_indexes[n])
            num_node = feat_input.size(0)
            att_stack = torch.zeros(1, num_node, self.stack_len).to(self.device) # batch_size=1, dim of attention=num_node
            stack_ptr = torch.zeros(1, self.stack_len).to(self.device)
            stack_ptr[0, 0] = 1
            mem = torch.zeros(1, self.dim_v).to(self.device)

            for t in range(self.T_ctrl):
                c_i = self.map_ci_to_query(c_list[t, n]) # (dim_v,)
                module_prob = module_probs[t, n] # (num_module,)
                if self.use_validity:
                    module_validity = torch.matmul(stack_ptr, self.module_validity_mat).view(self.num_module)
                    module_validity = torch.round(module_validity)
                    module_prob = module_prob * module_validity
                    _ = torch.sum(module_prob)
                    module_prob /= (_ if _.item() > 0 else 1) # (num_module,)

                # run all modules
                res = [f(cat_matrixes[n], conn_matrixes[n], self.edge_cat_vectors, feat_input, c_i, att_stack, stack_ptr, mem)
                        for f in self.module_funcs]
                att_stack_avg = torch.sum(module_prob.view(-1,1,1,1) * torch.stack([r[0] for r in res]), dim=0)
                stack_ptr_avg = torch.sum(module_prob.view(-1,1,1) * torch.stack([r[1] for r in res]), dim=0)
                mem_avg = torch.sum(module_prob.view(-1,1,1) * torch.stack([r[2] for r in res]), dim=0)
                att_stack, stack_ptr, mem = att_stack_avg, stack_ptr_avg, mem_avg

            final_module_outputs.append(mem)
            
        final_module_outputs = torch.cat(final_module_outputs, dim=0) # (batch_size, dim_v)
        final_module_outputs = torch.cat([final_module_outputs, question_hidden], dim=1) # (batch_size, dim_v+dim_hidden)
        predicted_logits = self.classifier(final_module_outputs)
        others = {
        }
        return predicted_logits, others


    def forward_and_return_intermediates(self, questions, questions_len, gt_programs, conn_matrixes, cat_matrixes, v_indexes):

        assert len(v_indexes) == 1, 'only support intermediates of batch size 1'
        intermediates = []

        batch_size = len(v_indexes)
        questions = questions.permute(1, 0) # (seq_len, batch_size)
        question_outputs, question_hidden, question_wordemb = self.question_encoder(questions, questions_len)
        module_logits, module_probs, c_list, cv_list = self.controller(
            question_outputs, question_hidden, question_wordemb, questions_len)

        final_module_outputs = []
        n = 0
        feat_input = self.attribute_embedding(v_indexes[n])
        num_node = feat_input.size(0)
        att_stack = torch.zeros(1, num_node, self.stack_len).to(self.device) # batch_size=1, dim of attention=num_node
        stack_ptr = torch.zeros(1, self.stack_len).to(self.device)
        stack_ptr[0, 0] = 1
        mem = torch.zeros(1, self.dim_v).to(self.device)

        for t in range(self.T_ctrl):
            c_i = self.map_ci_to_query(c_list[t, n]) # (dim_v,)
            module_prob = module_probs[t, n] # (num_module,)
            if self.use_validity:
                module_validity = torch.matmul(stack_ptr, self.module_validity_mat).view(self.num_module)
                module_validity = torch.round(module_validity)
                module_prob = module_prob * module_validity
                _ = torch.sum(module_prob)
                module_prob /= (_ if _.item() > 0 else 1) # (num_module,)

            # run all modules
            res = [f(cat_matrixes[n], conn_matrixes[n], self.edge_cat_vectors, feat_input, c_i, att_stack, stack_ptr, mem)
                    for f in self.module_funcs]
            att_stack_avg = torch.sum(module_prob.view(-1,1,1,1) * torch.stack([r[0] for r in res]), dim=0)
            stack_ptr_avg = torch.sum(module_prob.view(-1,1,1) * torch.stack([r[1] for r in res]), dim=0)
            mem_avg = torch.sum(module_prob.view(-1,1,1) * torch.stack([r[2] for r in res]), dim=0)
            att_stack, stack_ptr, mem = att_stack_avg, stack_ptr_avg, mem_avg

            module_type = vocab['program_idx_to_token'][module_prob.max()[1].item()]
            attention = cv_list[t, n]
            question_attention_str = ''
            for j in range(questions_len[n].item()):
                word = self.vocab['question_idx_to_token'][questions[j,n].item()]
                weight = attention[j].item()
                question_attention_str += '%s-%.2f ' % (word, weight)
            vertex_weight = None
            if module_type in {'_Find', '_Transform', '_Filter', '_Describe'}:
                vertex_weight = output.data.cpu().numpy()
            intermediates.append((
                'time %d, '%t + module_type + ':::' + question_attention_str, # question attention heatmap
                vertex_weight, # vertex attention vector
                    ))

        final_module_outputs.append(mem)
            
        final_module_outputs = torch.cat(final_module_outputs, dim=0) # (batch_size, dim_v)
        final_module_outputs = torch.cat([final_module_outputs, question_hidden], dim=1) # (batch_size, dim_v+dim_hidden)
        pred = self.classifier(final_module_outputs.squeeze()).max(0)[1]

        return (self.vocab['answer_idx_to_token'][pred.item()], intermediates)

