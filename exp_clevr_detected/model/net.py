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
       
        # The classifier takes the output of the last module
        # and produces a distribution over answers
        self.classifier = nn.Sequential(
            nn.Linear(self.dim_v, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_class)
            )

        self.map_feature_to_v = nn.Sequential(
                nn.Linear(self.dim_feature, 2*self.dim_v),
                nn.ReLU(),
                nn.Linear(2*self.dim_v, self.dim_v),
                nn.ReLU()
                )
        if self.edge_class == 'learncat':
            self.num_edge_cat = 9
            self.map_edge_to_class = nn.Sequential(
                nn.Linear(self.dim_edge, self.dim_v),
                nn.ReLU(),
                nn.Linear(self.dim_v, self.num_edge_cat),
                nn.Softmax(dim=2), # (num_node, num_node, num_edge_cat)
            )
        elif self.edge_class == 'dense':
            self.map_edge_to_v = nn.Sequential(
                nn.Linear(self.dim_edge, 300),
                nn.ReLU(),
                nn.Linear(300, self.dim_v),
            )

        self.edge_cat_vectors = nn.Parameter(torch.zeros(self.num_edge_cat, self.dim_v)) # null, left, right, front, behind
        nn.init.normal_(self.edge_cat_vectors.data, mean=0, std=1/math.sqrt(self.dim_v))

        # encode program_input, which must be included in question tokens
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



    def forward(self, programs, program_inputs, features, edge_vectors):
        """
        Args:
            features [list of Tensor] (batch_size, num_node, dim_v) 
            edge_vectors [list of Tensor] (batch_size, num_node, num_node, dim_edge)
            programs [list of Tensor]
            program_inputs [list of Tensor]
        """
        batch_size = len(programs)
        device = programs[0].device

        final_module_outputs = []
        for n in range(batch_size):
            feature = self.map_feature_to_v(features[n])
            num_node = feature.size(0)
            # edge_vector = self.map_edge_to_v(edge_vectors[n]) # (num_node, num_node, dim_v)
            edge_vector = edge_vectors[n] # (num_node, num_node, 2)
            if self.edge_class == 'learncat':
                edge_vector = self.map_edge_to_class(edge_vector) # (num_node, num_node, num_edge_cat)
            elif self.edge_class == 'dense':
                edge_vector = self.map_edge_to_v(edge_vector) # (num_node, num_node, dim_v)

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
                        output = module(output, saved_output)  # these modules take two feature maps
                    elif module_type in {'relate'}:
                        # output = module(output, feature, query, edge_vector)
                        output = module(output, feature, query, self.edge_cat_vectors, edge_vector)
                    else:
                        output = module(output, feature, query)
            except Exception as e:
                if self.training:
                    raise
                else:
                    print("Find a wrong program")
                    output = torch.zeros(self.dim_v).to(device)
            final_module_outputs.append(output)
            
        final_module_outputs = torch.stack(final_module_outputs)
        others = {
                }
        return self.classifier(final_module_outputs), others


    def forward_and_return_intermediates(self, programs, program_inputs, features, edge_vectors):

        assert len(programs) == 1, 'only support intermediates of batch size 1'
        device = programs[0].device
        intermediates = []

        n = 0
        feature = self.map_feature_to_v(features[n])  # (num_node, dim_v)
        num_node = feature.size(0)
        edge_vector = edge_vectors[n] # (num_node, num_node, 2)
        if self.edge_class == 'learncat':
            edge_vector = self.map_edge_to_class(edge_vector) # (num_node, num_node, num_edge_cat)
        elif self.edge_class == 'dense':
            edge_vector = self.map_edge_to_v(edge_vector) # (num_node, num_node, dim_v)

        saved_output, output = None, None
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
                intermediates.append(None)
                continue
            
            if module_type in {'intersect', 'union', 
                        'equal', 'equal_integer', 'less_than', 'greater_than'}:
                output = module(output, saved_output)  # these modules take two feature maps
            elif module_type in {'relate'}:
                output = module(output, feature, query, self.edge_cat_vectors, edge_vector)
            else:
                output = module(output, feature, query)

            if module_type in {'intersect', 'union'}:
                intermediates.append(None)
            if module_type in {'intersect', 'union', 'relate', 'same', 'filter'}:
                intermediates.append((
                    module_type+'_'+self.vocab['question_idx_to_token'][module_input.item()],
                    output.data.cpu().numpy()
                    ))
     
        _, pred = self.classifier(output.unsqueeze(0)).max(1)
        return (self.vocab['answer_idx_to_token'][pred.item()], intermediates)

