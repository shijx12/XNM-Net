import torch
import torch.nn as nn
import math
from . import basic_modules
from . import composite_modules as modules


class XNMNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        for k,v in kwargs.items():
            setattr(self, k, v)
       
        # The classifier takes the output of the last module and produces a distribution over answers
        self.classifier = nn.Sequential(
            nn.Linear(self.dim_v, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_class)
            )

        # map original features to dimension v
        self.map_node_feat_to_v = nn.Sequential(
                nn.Linear(self.dim_feature, 256),
                nn.ReLU(),
                nn.Linear(256, self.dim_v),
            )
        self.map_edge_feat_to_v = nn.Sequential(
                nn.Linear(self.dim_edge, 256),
                nn.ReLU(),
                nn.Linear(256, self.dim_v),
            )

        self.word_embedding = nn.Embedding(len(self.vocab['question_token_to_idx']), self.dim_v)
        nn.init.normal_(self.word_embedding.weight, mean=0, std=1/math.sqrt(self.dim_v))

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
                module = modules.ComparisonModule(**kwargs)
            elif module_name in {'exist', 'count'}:
                module = modules.ExistOrCountModule(**kwargs)
            elif module_name == 'query':
                module = modules.QueryModule(**kwargs)
            elif module_name == 'relate':
                module = modules.RelateModule(**kwargs)
            elif module_name == 'same':
                module = modules.SameModule(**kwargs)
            elif module_name == 'filter':
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


    def forward(self, programs, program_inputs, node_feats, edge_feats):
        """
        Args:
            programs [list of Tensor]
            program_inputs [list of Tensor]
            node_feats [list of Tensor] (batch_size, (num_node, self.dim_feature)) 
            edge_feats [list of Tensor] (batch_size, (num_node, num_node, self.dim_edge))
        """
        batch_size = len(programs)
        device = programs[0].device

        final_module_outputs = []
        for n in range(batch_size):
            node_feat = self.map_node_feat_to_v(node_feats[n])
            num_node = node_feat.size(0)
            edge_feat = self.map_edge_feat_to_v(edge_feats[n]) # (num_node, num_node, dim_v)

            saved_output, output = None, None
            try:
                for i in range(-1, -len(programs[n])-1, -1):
                    module_type = self.vocab['program_idx_to_token'][programs[n][i].item()]
                    if module_type in {'<NULL>', '<START>', '<END>', '<UNK>', 'unique'}:
                        continue  # the above are no-ops in our model
                    module_input = program_inputs[n][i] # <NULL> if no input for module
                    query = self.word_embedding(module_input) # (dim_v, ) 

                    module = self.function_modules[module_type]
                    if module_type == 'scene':
                        # store the previous output; it will be needed later
                        # scene is just a flag, performing no computation
                        saved_output = output
                        output = torch.ones(num_node).to(device)
                        continue
                    
                    if module_type in {'intersect', 'union', 
                                'equal', 'equal_integer', 'less_than', 'greater_than'}:
                        output = module(output, saved_output)
                    elif module_type in {'relate'}:
                        output = module(output, node_feat, query, edge_feat)
                    else:
                        output = module(output, node_feat, query)
            except Exception as e:
                if self.training:
                    raise
                else:
                    print("Find a wrong program")
                    output = torch.zeros(self.dim_v).to(device)
            final_module_outputs.append(output)
        final_module_outputs = torch.stack(final_module_outputs)
        return self.classifier(final_module_outputs)


    def forward_and_return_intermediates(self, programs, program_inputs, node_feats, edge_feats):

        assert len(programs) == 1, 'only support intermediates of batch size 1'
        device = programs[0].device
        intermediates = []

        n = 0
        node_feat = self.map_node_feat_to_v(node_feats[n]) # (num_node, dim_v)
        num_node = node_feat.size(0)
        edge_feat = self.map_edge_feat_to_v(edge_feats[n]) # (num_node, num_node, dim_v)

        saved_output, output = None, None
        for i in range(-1, -len(programs[n])-1, -1):
            module_type = self.vocab['program_idx_to_token'][programs[n][i].item()]
            if module_type in {'<NULL>', '<START>', '<END>', '<UNK>', 'unique'}:
                continue  # the above are no-ops in our model
            module_input = program_inputs[n][i] # <NULL> if no input for module
            query = self.word_embedding(module_input) # (dim_v, ) 

            module = self.function_modules[module_type]
            if module_type == 'scene':
                saved_output = output
                output = torch.ones(num_node).to(device)
                intermediates.append(None)
                continue
            
            if module_type in {'intersect', 'union', 
                        'equal', 'equal_integer', 'less_than', 'greater_than'}:
                output = module(output, saved_output)  # these modules take two feature maps
            elif module_type in {'relate'}:
                output = module(output, node_feat, query, edge_feat)
            else:
                output = module(output, node_feat, query)

            if module_type in {'intersect', 'union'}:
                intermediates.append(None)
            if module_type in {'intersect', 'union', 'relate', 'same', 'filter'}:
                if self.vocab['question_idx_to_token'][module_input.item()] == '<NULL>':
                    module_str = module_type
                else:
                    module_str = "%s[%s]" % (module_type, self.vocab['question_idx_to_token'][module_input.item()])
                intermediates.append((
                    module_str,
                    output.data.cpu().numpy()
                    ))
     
        _, pred = self.classifier(output.unsqueeze(0)).max(1)
        return (self.vocab['answer_idx_to_token'][pred.item()], intermediates)

