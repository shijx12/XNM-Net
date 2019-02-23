import torch

def todevice(tensor, device):
    if isinstance(tensor, list) or isinstance(tensor, tuple):
        assert isinstance(tensor[0], torch.Tensor)
        return [todevice(t, device) for t in tensor]
    elif isinstance(tensor, torch.Tensor):
        return tensor.to(device)

def invert_dict(d):
    return {v: k for k, v in d.items()}

def convert_david_program_to_mine(program, vocab):
    # program is a list of str
    # vocab is my vocab
    functions, value_inputs = [], []
    for f in program:
        value_input = '<NULL>'
        function = f
        if f in {'equal_shape', 'equal_color', 'equal_size', 'equal_material'}:
            function = 'equal'
        elif 'query' in f:
            function = 'query'
            value_input = f[6:] # <cat> of query_<cat>
        elif 'same' in f:
            function = 'same'
            value_input = f[5:] # <cat> of same_<cat>
        elif 'filter_' in f:
            function = 'filter'
            value_input = f[f.find('[')+1:-1]
        elif 'relate' in f:
            function = 'relate'
            value_input = f[f.find('[')+1:-1]
        
        functions.append(function)
        value_inputs.append(value_input)
    functions = [vocab['program_token_to_idx'][f] for f in functions]
    value_inputs = [vocab['question_token_to_idx'][v] for v in value_inputs]
    return functions, value_inputs

