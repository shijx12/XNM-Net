import torch
import torch.nn as nn
import numpy as np
from itertools import chain
from utils.misc import reverse_padded_sequence

class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.num_vocab = kwargs['num_vocab']
        self.word_dim = kwargs['word_dim']
        self.lstm_dim = kwargs['lstm_dim']

        self.embedding = nn.Embedding(self.num_vocab, self.word_dim)
        self.forward_lstm = nn.LSTM(self.word_dim, self.lstm_dim // 2)
        self.backward_lstm = nn.LSTM(self.word_dim, self.lstm_dim // 2)
        self.reset_parameters()

    def forward(self, input_seqs, input_seq_lens):
        """
            Input:
                input_seqs: [seq_max_len, batch_size]
                input_seq_lens: [batch_size]
        """
        embedded = self.embedding(input_seqs) # [seq_max_len, batch_size, word_dim]
        forward_outputs = self.forward_lstm(embedded)[0] # [seq_max_len, batch_size, dim/2]
        reversed_embedded = reverse_padded_sequence(embedded, input_seq_lens)
        reversed_backward_outputs = self.backward_lstm(reversed_embedded)[0]
        backward_outputs = reverse_padded_sequence(reversed_backward_outputs, input_seq_lens) # [seq_max_len, batch_size, dim/2]
        outputs = torch.cat([forward_outputs, backward_outputs], dim=2) # [seq_max_len, batch_size, dim]
        # indexing outputs via input_seq_lens
        hidden = []
        for i, l in enumerate(input_seq_lens):
            hidden.append(outputs[l-1, i])
        hidden = torch.stack(hidden) # (batch_size, dim)
        return outputs, hidden, embedded

    def reset_parameters(self):
        nn.init.normal_(self.embedding.weight, mean=0, std=0.01)
        for name, param in chain(self.forward_lstm.named_parameters(), 
                        self.backward_lstm.named_parameters()):
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.kaiming_normal_(param)



