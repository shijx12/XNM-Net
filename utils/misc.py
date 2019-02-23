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

def convert_to_one_hot(indices, num_classes):
    batch_size = indices.size(0)
    indices = indices.unsqueeze(1)
    one_hot = indices.data.new(batch_size, num_classes).zero_().scatter_(1, indices.data, 1)
    return one_hot

def reverse_padded_sequence(inputs, lengths, batch_first=False):
    """Reverses sequences according to their lengths.
    Inputs should have size ``T x B x *`` if ``batch_first`` is False, or
    ``B x T x *`` if True. T is the length of the longest sequence (or larger),
    B is the batch size, and * is any number of dimensions (including 0).
    Arguments:
        inputs (Variable): padded batch of variable length sequences.
        lengths (list[int]): list of sequence lengths
        batch_first (bool, optional): if True, inputs should be B x T x *.
    Returns:
        A Variable with the same size as inputs, but with each sequence
        reversed according to its length.
    """

    if not batch_first:
        inputs = inputs.transpose(0, 1)
    if inputs.size(0) != len(lengths):
        raise ValueError('inputs incompatible with lengths.')
    reversed_indices = [list(range(inputs.size(1)))
                        for _ in range(inputs.size(0))]
    for i, length in enumerate(lengths):
        if length > 0:
            reversed_indices[i][:length] = reversed_indices[i][length-1::-1]
    reversed_indices = torch.LongTensor(reversed_indices).unsqueeze(2).expand_as(inputs).to(inputs.device)
    reversed_inputs = torch.gather(inputs, 1, reversed_indices)
    if not batch_first:
        reversed_inputs = reversed_inputs.transpose(0, 1)
    return reversed_inputs
