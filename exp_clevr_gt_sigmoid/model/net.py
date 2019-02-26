import torch
import torch.nn as nn
import math

from . import composite_modules as modules
from . import basic_modules


class XNMNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        for k,v in kwargs.items():
            setattr(self, k, v)
        
        # The classifier takes the output of the last module
        # and produces a distribution over answers
        self.classifier = nn.Sequential(
            nn.Linear(self.dim_v, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_class)
            )

        # embedding of all words, including question tokens, attribute values and relationship categories 
        self.word_embedding = nn.Embedding(len(self.vocab['question_token_to_idx']), self.dim_v)
        nn.init.normal_(self.word_embedding.weight, mean=0, std=1/math.sqrt(self.dim_v))

        # map concatenation of attribute value embeddings to feature vectors of dim_v
        self.reduce_4v_to_v = nn.Sequential(
                nn.Linear(4 * self.dim_v, 2 * self.dim_v),
                nn.ReLU(),
                nn.Linear(2 * self.dim_v, self.dim_v),
                nn.ReLU()
                )
        self.reduce_2v_to_v = nn.Sequential(
                nn.Linear(2 * self.dim_v, self.dim_v),
                nn.ReLU(),
                )

        self.function_modules = {}  # holds our modules
        # go through the vocab and add all the modules to our model
        for module_name in self.vocab['program_token_to_idx']:
            if module_name in ['<NULL>', '<START>', '<END>', '<UNK>', 'unique']:
                continue  # we don't need modules for the placeholders
            
            # figure out which module we want we use
            if module_name == 'scene':
                # scene is just a flag that indicates the start of a new line of reasoning
                # we set `module` to `None` because we still need the flag 'scene' in forward()
                module = None
            elif module_name == 'intersect':
                module = basic_modules.AndModule()
            elif module_name == 'union':
                module = basic_modules.OrModule()
            elif module_name in {'equal', 'equal_integer', 'less_than', 'greater_than'}:
                # 'equal_<cat>' are merged into 'equal', <cat> is ignored
                module = modules.ComparisonModule(**kwargs)
            elif module_name in {'exist', 'count'}:
                module = modules.ExistOrCountModule(**kwargs)
            elif module_name == 'query':
                # 'query_<cat>' are merged into 'query', <cat> is viewed as input
                module = modules.QueryModule(**kwargs)
            elif module_name == 'relate':
                module = modules.RelateModule(**kwargs)
            elif module_name == 'same':
                # 'same_<cat>' are merged into 'same', <cat> is viewed as input
                module = modules.SameModule(**kwargs)
            elif module_name == 'filter':
                # 'filter_<cat>' are merged into 'filter', <cat> is ignored.
                module = modules.AttentionModule(**kwargs)
            else:
                raise Exception("Invalid module_name")

            # add the module to our dictionary and register its parameters so it can learn
            self.function_modules[module_name] = module
            self.add_module(module_name, module)
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, programs, program_inputs, edge_index, node_index):
        """
        Args:
            edge_index [list of Tensor]
            node_index [list of Tensor] (batch_size, (num_node, 4)) : represent vertex via 4 attribute value index
            programs [list of Tensor]
            program_inputs [list of Tensor]
        """
        batch_size = len(programs)
        device = programs[0].device

        final_module_outputs = []
        count_outputs = []
        for n in range(batch_size):
            num_node = len(node_index[n])
            node_feat = self.word_embedding(node_index[n]).view(num_node, -1) # concat 4 attribute value embeddings
            node_feat = self.reduce_4v_to_v(node_feat) # (num_node, dim_v)
            edge_feat = self.word_embedding(edge_index[n]).view(num_node, num_node, -1)
            edge_feat = self.reduce_2v_to_v(edge_feat) # (num_node, num_node, dim_v)
            saved_output, output = None, None
            try:
                for i in range(-1, -len(programs[n])-1, -1):
                    module_type_ = self.vocab['program_idx_to_token'][programs[n][i].item()]
                    if module_type_ in {'<NULL>', '<START>', '<END>', '<UNK>', 'unique'}:
                        continue  # the above are no-ops in our model
                    module_type = module_type_
                    module_input = program_inputs[n][i] # <NULL> if no input for module
                    query = self.word_embedding(module_input) # used in node or edge attention
                    
                    module = self.function_modules[module_type]
                    if module_type == 'scene':
                        # store the previous output; it will be needed later
                        # scene is just a flag, performing no computation
                        saved_output = output
                        output = torch.ones(num_node).to(device)
                        continue
                    
                    if module_type in {'intersect', 'union', 
                                'equal', 'equal_integer', 'less_than', 'greater_than'}:
                        output = module(output, saved_output)  # these modules take two feature maps
                    elif module_type in {'exist', 'count'}:
                        att_sum = output.sum()
                        output = module(output) # take as input one attention
                    elif module_type in {'query', 'same', 'filter'}:
                        output = module(output, node_feat, query)
                    elif module_type == 'relate':
                        output = module(output, node_feat, query, edge_feat)
                    else:
                        raise Exception("Invalid module type")
            except Exception as e:
                if self.training:
                    raise
                else:
                    print("Find a wrong program")
                    output = torch.zeros(self.dim_v).to(device)
            final_module_outputs.append(output)
            if module_type == 'count': # For questions whose type is count, return the sum of node attention
                count_outputs.append(att_sum)
            else:
                count_outputs.append(None)
            
        final_module_outputs = torch.stack(final_module_outputs)
        logits = self.classifier(final_module_outputs)
        others = {
                'count_outputs': count_outputs,
                }
        return logits, others


    def forward_and_return_intermediates(self, programs, program_inputs, edge_index, node_index):
        assert len(programs) == 1, 'only support intermediates of batch size 1'
        assert not self.training, 'only eval mode is supported'
        device = programs[0].device
        intermediates = []

        n = 0
        num_node = len(node_index[n])
        node_feat = self.word_embedding(node_index[n]).view(num_node, -1) # concat 4 attribute value embeddings
        node_feat = self.reduce_4v_to_v(node_feat) # (num_node, dim_v)
        edge_feat = self.word_embedding(edge_index[n]).view(num_node, num_node, -1)
        edge_feat = self.reduce_2v_to_v(edge_feat) # (num_node, num_node, dim_v)
        saved_output, output = None, None
        try:
            for i in range(-1, -len(programs[n])-1, -1):
                module_type = self.vocab['program_idx_to_token'][programs[n][i].item()]
                if module_type in {'<NULL>', '<START>', '<END>', '<UNK>', 'unique'}:
                    continue  # the above are no-ops in our model
                module_input = program_inputs[n][i] # <NULL> if no input for module
                query = self.word_embedding(module_input) # used in node or edge attention

                module = self.function_modules[module_type]
                if module_type == 'scene':
                    # store the previous output; it will be needed later
                    # scene is just a flag, performing no computation
                    saved_output = output
                    output = torch.ones(num_node).to(device)
                    intermediates.append(None)
                    continue
                
                if module_type in {'intersect', 'union', 
                            'equal', 'equal_integer', 'less_than', 'greater_than'}:
                    output = module(output, saved_output)  # these modules take two feature maps
                elif module_type in {'exist', 'count'}:
                    output = module(output) # take as input one attention
                elif module_type in {'query', 'same', 'filter'}:
                    output = module(output, node_feat, query)
                elif module_type == 'relate':
                    output = module(output, node_feat, query, edge_feat)
                else:
                    raise Exception("Invalid module type")
                
                if module_type in {'intersect', 'union'}:
                    intermediates.append(None)
                if module_type in {'intersect', 'union', 'relate', 'same', 'filter'}:
                    module_input = self.vocab['question_idx_to_token'][module_input.item()]
                    intermediates.append((
                        module_type + ('[%s]'%(module_input) if module_input!='<NULL>' else ''), 
                        output.data.cpu().numpy()
                        ))
        except Exception as e:
            print("Find a wrong program")
            output = torch.zeros(self.dim_v).to(device)
            intermediates = []
     
        _, pred = self.classifier(output.unsqueeze(0)).max(1)
        return (self.vocab['answer_idx_to_token'][pred.item()], intermediates)

