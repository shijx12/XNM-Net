import torch
import torch.nn as nn

from . import composite_modules as modules
from . import basic_modules


class XNMNet(nn.Module):
    def __init__(self,
                 vocab,
                 dim_pre_v,
                 dim_v,
                 num_edge_cat=9,
                 num_attribute=15,
                 fc_dim=256):
        """

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
        
        # The classifier takes the output of the last module
        # and produces a distribution over answers
        self.classifier = nn.Sequential(
            nn.Linear(dim_v, fc_dim),
            nn.ReLU(),
            nn.Linear(fc_dim, 28)  # note no softmax here
            )
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.constant_(layer.bias, val=0)

        self.edge_cat_vectors = nn.Parameter(torch.Tensor(num_edge_cat, dim_v))
        # nn.init.orthogonal_(self.edge_cat_vectors.data)
        nn.init.normal_(self.edge_cat_vectors.data, mean=0, std=0.01)

        # encode program_input, which must be included in question tokens
        self.word_embedding = nn.Embedding(len(vocab['question_token_to_idx']), dim_v)
        nn.init.normal_(self.word_embedding.weight, mean=0, std=0.01)

        # map onehot node representation to feature vectors of dim_v
        self.map_pre_to_v = nn.Sequential(
            nn.Linear(dim_pre_v, dim_v),
            nn.ReLU()
            )
        for layer in self.map_pre_to_v:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.constant_(layer.bias, val=0)


        self.function_modules = {}  # holds our modules
        self.vocab = vocab
        self.dim_v = dim_v
        self.num_attribute = num_attribute
        # go through the vocab and add all the modules to our model
        for module_name in vocab['program_token_to_idx']:
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
                module = modules.ComparisonModule(dim_v)
            elif module_name in {'exist', 'count'}:
                module = modules.ExistOrCountModule(dim_v)
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


    def forward(self, programs, program_inputs, conn_matrixes, cat_matrixes, pre_v):
        """
        Args:
            conn_matrixes [list of Tensor]
            cat_matrixes [list of Tensor]
            pre_v [list of Tensor] (batch_size, num_node, dim_pre_v) : pre representation of each graph
            programs [list of Tensor]
            program_inputs [list of Tensor]
        """
        batch_size = len(pre_v)
        device = programs[0].device
        # assert batch_size == len(programs) == len(program_inputs)

        final_module_outputs = []
        for n in range(batch_size):
            feat_input = self.map_pre_to_v(pre_v[n])
            num_node = feat_input.size(0)
            saved_output, output = None, None
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
     
            final_module_outputs.append(output)
            
        final_module_outputs = torch.stack(final_module_outputs)
        return self.classifier(final_module_outputs)


    def forward_and_return_intermediates(self, programs, program_inputs, conn_matrixes, cat_matrixes, pre_v):

        assert len(pre_v) == 1, 'only support intermediates of batch size 1'
        device = programs[0].device
        intermediates = []

        feat_input = self.map_pre_to_v(pre_v[0])
        num_node = feat_input.size(0)
        saved_output, output = None, None
        for i in range(-1, -len(programs[0])-1, -1):
            module_type = self.vocab['program_idx_to_token'][programs[0][i].item()]
            if module_type in {'<NULL>', '<START>', '<END>', '<UNK>', 'unique'}:
                continue  # the above are no-ops in our model
            module_input = program_inputs[0][i] # <NULL> if no input for module
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
                output = module(output, cat_matrixes[0], self.edge_cat_vectors, query, feat_input)
            elif module_type == 'same':
                output = module(output, cat_matrixes[0], self.edge_cat_vectors, query, conn_matrixes[0])
            elif module_type == 'filter':
                output = module(output, conn_matrixes[0], feat_input, query)
            else:
                raise Exception("Invalid module type")
            
            if module_type in {'intersect', 'union'}:
                intermediates.append(None)
            if module_type in {'intersect', 'union', 'relate', 'same', 'filter'}:
                intermediates.append((
                    module_type+'_'+self.vocab['question_idx_to_token'][module_input.item()], 
                    output.data.cpu().numpy()
                    ))
     
        _, pred = self.classifier(output.unsqueeze(0)).max(1)
        return (self.vocab['answer_idx_to_token'][pred.item()], intermediates)

