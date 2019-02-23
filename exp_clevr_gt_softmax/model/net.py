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
        #self.classifier = nn.Sequential(
        #    nn.Linear(self.dim_v, 256),
        #    nn.ReLU(),
        #    nn.Linear(256, self.num_class)
        #    )
        self.classifier = nn.Sequential(
                nn.Linear(self.dim_v, 256),
                nn.ReLU(),
                nn.Linear(256, self.num_class)
            )

        # embeddings of relationship categories (e.g., left, right) and attribute categories (e.g., color, size)
        self.edge_cat_vectors = nn.Parameter(torch.Tensor(self.num_edge_cat, self.dim_v))
        nn.init.normal_(self.edge_cat_vectors.data, mean=0, std=1/math.sqrt(self.dim_v))

        # program_input must be included in question tokens
        self.word_embedding = nn.Embedding(len(self.vocab['question_token_to_idx']), self.dim_v)
        nn.init.normal_(self.word_embedding.weight, mean=0, std=1/math.sqrt(self.dim_v))

        # map onehot node representation to feature vectors of dim_v
        self.map_pre_to_v = nn.Linear(self.dim_pre_v, self.dim_v)

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
                module = modules.ComparisonModule(self.dim_v)
            elif module_name in {'exist', 'count'}:
                module = modules.ExistOrCountModule(self.dim_v)
            elif module_name == 'query':
                # 'query_<cat>' are merged into 'query', <cat> is viewed as input
                module = modules.QueryModule()
            elif module_name == 'relate':
                module = modules.RelateModule()
            elif module_name == 'same':
                # 'same_<cat>' are merged into 'same', <cat> is viewed as input
                module = modules.SameModule()
            elif module_name == 'filter':
                # 'filter_<cat>' are merged into 'filter', <cat> is ignored.
                module = modules.AttentionModule()
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


    def forward(self, programs, program_inputs, conn_matrixes, cat_matrixes, pre_v):
        """
        Args:
            programs [list of Tensor]
            program_inputs [list of Tensor]
            conn_matrixes [list of Tensor] (batch_size, (num_node, num_node))
            cat_matrixes [list of Tensor] (batch_size, (num_node, num_node, max_rel))
            pre_v [list of Tensor] (batch_size, (num_node, dim_pre_v)) : one-hot representation of nodes
        """
        batch_size = len(pre_v)
        device = programs[0].device

        final_module_outputs = []
        for n in range(batch_size):
            feat_input = self.map_pre_to_v(pre_v[n])
            num_node = feat_input.size(0)
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
                        output[-self.num_attribute:] = 0 # assign 0 to attribute nodes
                        continue
                    
                    if module_type in {'intersect', 'union', 
                                'equal', 'equal_integer', 'less_than', 'greater_than'}:
                        output = module(output, saved_output)  # these modules take two feature maps
                    elif module_type in {'exist', 'count'}:
                        output = module(output) # take as input one attention
                    elif module_type in {'query', 'relate'}:
                        output = module(output, cat_matrixes[n], self.edge_cat_vectors, query, feat_input)
                    elif module_type == 'same':
                        output = module(output, cat_matrixes[n], self.edge_cat_vectors, query, conn_matrixes[n])
                    elif module_type == 'filter':
                        output = module(output, conn_matrixes[n], feat_input, query)
                    else:
                        raise Exception("Invalid module type")
            except Exception as e:
                if self.training:
                    raise
                else:
                    print("Find a wrong program")
                    output = torch.zeros(self.dim_v).to(device)
            final_module_outputs.append(output)
            
        final_module_outputs = torch.stack(final_module_outputs)
        logits = self.classifier(final_module_outputs)
        return logits


    def forward_and_return_intermediates(self, programs, program_inputs, conn_matrixes, cat_matrixes, pre_v):
        assert len(pre_v) == 1, 'only support intermediates of batch size 1'
        assert not self.training, 'only eval mode is supported'
        device = programs[0].device
        intermediates = []

        feat_input = self.map_pre_to_v(pre_v[0])
        num_node = feat_input.size(0)
        saved_output, output = None, None
        n = 0
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
                    output[-self.num_attribute:] = 0 # assign 0 to attribute nodes
                    intermediates.append(None)
                    continue
                
                if module_type in {'intersect', 'union', 
                            'equal', 'equal_integer', 'less_than', 'greater_than'}:
                    output = module(output, saved_output)  # these modules take two feature maps
                elif module_type in {'exist', 'count'}:
                    output = module(output) # take as input one attention
                elif module_type in {'query', 'relate'}:
                    output = module(output, cat_matrixes[n], self.edge_cat_vectors, query, feat_input)
                elif module_type == 'same':
                    output = module(output, cat_matrixes[n], self.edge_cat_vectors, query, conn_matrixes[n])
                elif module_type == 'filter':
                    output = module(output, conn_matrixes[n], feat_input, query)
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

