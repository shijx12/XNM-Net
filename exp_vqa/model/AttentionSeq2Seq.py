import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
from itertools import chain
from utils.misc import reverse_padded_sequence
from IPython import embed


class EncoderRNN(nn.Module):
    def __init__(self, input_encoding_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.forward_gru = nn.GRU(input_encoding_size, hidden_size)
        for name, param in self.forward_gru.named_parameters():
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.xavier_uniform_(param)

    def forward(self, input_seqs, input_embedded, input_seq_lens):
        embedded = input_embedded
        outputs = self.forward_gru(embedded)[0] # [seq_max_len, batch_size, dim]
        # indexing outputs via input_seq_lens
        hidden = []
        for i, l in enumerate(input_seq_lens): # batch
            hidden.append(outputs[l-1, i])
        hidden = torch.stack(hidden) # (batch, hidden_size)
        return hidden, outputs


class BiGRUEncoder(nn.Module):
    def __init__(self, dim_word, dim_hidden):
        super().__init__()
        self.forward_gru = nn.GRU(dim_word, dim_hidden//2)
        self.backward_gru = nn.GRU(dim_word, dim_hidden//2)
        for name, param in chain(self.forward_gru.named_parameters(), self.backward_gru.named_parameters()):
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)

    def forward(self, input_seqs, input_embedded, input_seq_lens):
        """
            Input:
                input_seqs: [seq_max_len, batch_size]
                input_seq_lens: [batch_size]
        """
        embedded = input_embedded # [seq_max_len, batch_size, word_dim]
        forward_outputs = self.forward_gru(embedded)[0] # [seq_max_len, batch_size, dim_hidden/2]
        backward_embedded = reverse_padded_sequence(embedded, input_seq_lens)
        backward_outputs = self.backward_gru(backward_embedded)[0]
        backward_outputs = reverse_padded_sequence(backward_outputs, input_seq_lens)
        outputs = torch.cat([forward_outputs, backward_outputs], dim=2) # [seq_max_len, batch_size, dim_hidden]
        # indexing outputs via input_seq_lens
        hidden = []
        for i, l in enumerate(input_seq_lens):
            hidden.append(
                torch.cat([forward_outputs[l-1, i], backward_outputs[0, i]], dim=0)
                )
        hidden = torch.stack(hidden) # (batch_size, dim)
        return hidden, outputs


class EncoderBiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, input_encoding_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, input_encoding_size)
        self.forward_gru = nn.GRU(input_encoding_size, hidden_size // 2)
        self.backward_gru = nn.GRU(input_encoding_size, hidden_size // 2)
        self.reset_parameters()

    def forward(self, input_seqs, input_embedded, input_seq_lens):
        # embedded = self.embedding(input_seqs)
        embedded = input_embedded
        forward_outputs = self.forward_gru(embedded)[0] # [seq_max_len, batch_size, dim/2]
        reversed_embedded = reverse_padded_sequence(embedded, input_seq_lens)
        reversed_backward_outputs = self.backward_gru(reversed_embedded)[0]
        backward_outputs = reverse_padded_sequence(reversed_backward_outputs, input_seq_lens) # [seq_max_len, batch_size, dim/2]
        outputs = torch.cat([forward_outputs, backward_outputs], dim=2) # [seq_max_len, batch_size, dim]
        # indexing outputs via input_seq_lens
        hidden = []
        for i, l in enumerate(input_seq_lens): # batch
            hidden.append(outputs[l-1, i])
        hidden = torch.stack(hidden).unsqueeze(0) # (1, batch, hidden_size)
        return outputs, hidden, embedded

    def reset_parameters(self):
        init.normal_(self.embedding.weight, mean=0, std=0.01)
        for name, param in chain(self.forward_gru.named_parameters(), self.backward_gru.named_parameters()):
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.kaiming_normal_(param)



class AttnDecoderRNN(nn.Module):
    def __init__(self, word_size, hidden_size, output_size, output_encoding_size, device, max_decoder_len=0,
                 assembler_w=None, assembler_b=None, assembler_p = None,EOStoken=-1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.output_encoding_size = output_encoding_size
        self.device = device
        self.max_decoder_len = max_decoder_len

        self.embedding = nn.Embedding(self.output_size, self.output_encoding_size) # module/layout embedding
        self.gru = nn.GRU(self.output_encoding_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size * 2, self.output_size)
        self.encoderLinear = nn.Linear(self.hidden_size, self.hidden_size)
        self.decoderLinear = nn.Linear(self.hidden_size, self.hidden_size)
        self.attnLinear = nn.Linear(self.hidden_size, 1)
        self.assembler_w = torch.FloatTensor(assembler_w).to(self.device)
        self.assembler_b = torch.FloatTensor(assembler_b).to(self.device)
        self.assembler_p = torch.FloatTensor(assembler_p).to(self.device)
        self.batch_size = 0
        self.EOS_token = EOStoken
        
        self.fc_q_list = [] # W_1^{(t)} q + b_1
        for t in range(self.max_decoder_len):
            self.fc_q_list.append(nn.Linear(self.hidden_size, self.hidden_size))
            self.add_module('fc_q_%d'%t, self.fc_q_list[t])
        self.fc_q_cat_h = nn.Linear(2*self.hidden_size, self.hidden_size) # W_2 [q;h] + b_2

        self.reset_parameters()
    
    def reset_parameters(self):
        init.normal_(self.embedding.weight, mean=0, std=0.01)
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.kaiming_normal_(param)
        for layer in chain(self.fc_q_list, (self.out, self.encoderLinear, self.decoderLinear, self.attnLinear, self.fc_q_cat_h)):
            init.kaiming_normal_(layer.weight)
            init.constant_(layer.bias, 0.0)

    '''
        compute if a token is valid at current sequence
        decoding_state [N,3]
        assembler_w [3,output_size, 4 ]
        assembler_b [output_size, 4]
        output [N, output_size]
    '''
    def _get_valid_tokens(self, decoding_state, assembler_W, assembler_b):

        batch_size = decoding_state.size(0)
        expanded_state = decoding_state.view(batch_size,3,1,1).expand(batch_size, 3, self.output_size, 4)

        expanded_w= assembler_W.view(1,3, self.output_size,4).expand(batch_size, 3, self.output_size, 4)

        tmp1 = torch.sum(expanded_state * expanded_w, dim=1)
        expanded_b = assembler_b.view(1,-1,4).expand(batch_size,-1,4)
        tmp2= tmp1 - expanded_b
        tmp3 = torch.min(tmp2,dim=2)[0]
        token_invalidity = torch.lt(tmp3, 0).to(self.device)
        return token_invalidity


    '''
        update the decoding state, which is used to determine if a token is valid
        decoding_state [N,3]
        assembler_p [output_size, 3]
        predicted_token [N,output_size]
        output [N, 3]
    '''
    def _update_decoding_state(self, decoding_state, predicted_token, assembler_P):
        decoding_state = decoding_state + torch.mm(predicted_token , assembler_P)
        return decoding_state


    '''
        for a give state compute the lstm hidden layer, attention and predicted layers
        can handle the situation where seq_len is 1 or >1 (i.e., s=using groudtruth layout)
        
        input parameters :
            time: int, time step of decoder
            previous_token: [decoder_len, batch], decoder_len=1 for step-by-step decoder
            previous_hidden_state: (h_n), dimmension:(num_layers * num_directions, batch, hidden_size), where num_layers=1 and num_directions=1
            encoder_outputs : outputs from LSTM in encoder[seq_len, batch, hidden_size * num_directions]
            encoder_lens: list of input sequence lengths
            decoding_state: the state used to decide valid tokens
        
        output parameters : 
            predicted_token: [decoder_len, batch]
            Att_weighted_text: batch,out_len,txt_embed_dim
            log_seq_prob: [batch]
            neg_entropy: [batch]
    '''
    def _step_by_step_attention_decoder(self, time, embedded, previous_hidden_state, cum_attention, encoder_embedded,
                                        encoder_outputs, encoder_hidden, encoder_lens, decoding_state,target_variable,sample_token):

        ##step1 run LSTM to get decoder hidden state
        seq_len = encoder_outputs.size(0)
        batch_size = encoder_outputs.size(1)
        hidden_size = encoder_outputs.size(2)

        out_len = embedded.size(0)

        output, hidden = self.gru(embedded, previous_hidden_state)
        ##step2: use function in Eq(2) of the paper to compute attention
        ##size encoder_outputs (seq_len,batch_size,hidden_size)==>(out_len,seq_len,batch_size,hidden_size)
        encoder_outputs_expand = encoder_outputs.view(1, seq_len, batch_size, hidden_size).expand(out_len, seq_len,batch_size,hidden_size)
        encoder_transform = self.encoderLinear(encoder_outputs_expand)

        ##size output (out_len,batch_size,hidden_size)
        output_expand = output.view(out_len, 1, batch_size, hidden_size).expand(out_len, seq_len, batch_size, hidden_size)
        output_transfrom = self.decoderLinear(output_expand)

        ##raw_attention size (out_len,seq_len,batch_size)
        raw_attention = self.attnLinear(F.tanh(encoder_transform + output_transfrom)).view(out_len, seq_len,batch_size)  ## Eq2
        # raw_attention = torch.sum(encoder_transform * output_transfrom, dim=3)

        # (out_len, seq_len, batch_size)==>(batch_size,out_len,seq_len)
        raw_attention = raw_attention.permute(2, 0, 1)

        ##mask the end of the question
        if encoder_lens is not None:
            mask = np.ones((batch_size, out_len, seq_len))
            for i, v in enumerate(encoder_lens):
                mask[i, :, 0:v] = 0
            mask_tensor = torch.ByteTensor(mask)
            mask_tensor = mask_tensor.to(self.device)
            raw_attention.data.masked_fill_(mask_tensor, -float('inf'))

        attention = F.softmax(raw_attention, dim=2)  ##(batch,out_len,seq_len)
        

        ##c_t = \sum_{i=1}^I att_{ti}h_i t: decoder time t, and encoder time i
        ## (seq_len,batch_size,hidden_size) ==>(batch_size,seq_len,hidden_size)
        encoder_batch_first = encoder_outputs.permute(1, 0, 2)
        context = torch.bmm(attention, encoder_batch_first)

        ##(out_len,batch,hidden_size) --> (batch,out_len,hidden_size)
        output_batch_first = output.permute(1, 0, 2)

        ##(batch,out_len,hidden_size*2)
        combined = torch.cat((context, output_batch_first), dim=2).permute(1, 0, 2)

        ## [out_len,batch,out_size]
        output_logit = self.out(combined)

        ##get the valid token for current position based on previous token to perform a mask for next prediction
        ## token_validity [batch, output_size]
        token_invalidity = self._get_valid_tokens(decoding_state=decoding_state,
                                                assembler_W=self.assembler_w,
                                                assembler_b=self.assembler_b)
        token_invalidity.view_as(output_logit)# It seems that out_len must be 1
        output_logit.data.masked_fill_(token_invalidity, -float('inf'))

        ## [batch,out_size]
        probs = F.softmax(output_logit, dim=2).view(-1, self.output_size)

        if target_variable is not None: # fetch from target_variable
            predicted_token = target_variable[time, :].view(-1,1)
        elif sample_token:
            predicted_token = probs.multinomial()
        else:
            predicted_token = torch.max(probs, dim=1)[1].view(-1, 1)
        # predicted_token [batch, 1]

        #print('attention')
        #embed()

        # one-hot vector of predicted_token [batch_size, self.output_size]
        tmp = torch.zeros(batch_size, self.output_size).to(self.device)
        predicted_token_encoded = tmp.scatter_(1, predicted_token, 1.0)

        updated_decoding_state = self._update_decoding_state(decoding_state=decoding_state,
                                                             predicted_token=predicted_token_encoded,
                                                             assembler_P=self.assembler_p)

        ## compute the negative entropy
        token_neg_entropy = torch.sum(probs.detach() * torch.log(probs + 0.000001), dim=1) # why probs.detach ?

        ## compute log_seq_prob
        selected_token_log_prob =torch.log(torch.sum(probs * predicted_token_encoded, dim=1)+ 0.000001)

        ################################################
        # compute function input attention 
        if False:
            function_input_query = self.embedding(predicted_token.view(batch_size)) # (batch, output_encoding_size)
            assert self.output_encoding_size == self.hidden_size
            function_input_query = function_input_query.view(batch_size, self.output_encoding_size, 1)
            encoder_outputs_expand = encoder_outputs.view(seq_len, batch_size, 1, self.hidden_size)
            raw_attention = torch.matmul(encoder_outputs_expand, function_input_query).view(seq_len, batch_size) # (seq_len, batch_size)
            raw_attention = raw_attention.permute(1, 0).view_as(mask_tensor) # (batch_size, out_len, seq_len)
            raw_attention.data.masked_fill_(mask_tensor, -float('inf'))
            # cum_attention 
            if cum_attention is not None:
                cum_attention_mask = cum_attention.gt(0.5)
                raw_attention.data.masked_fill_(cum_attention_mask, -float('inf'))
            attention = F.softmax(raw_attention, dim=2)  ##(batch,out_len,seq_len)
        if False: # similar to snmn, each timestamp has one net
            q_i = self.fc_q_list[time](encoder_hidden)
            q_i_h = torch.cat([q_i.squeeze(0), output.squeeze(0)], dim=1) # [batch_size, 2*dim_h]
            query = self.fc_q_cat_h(q_i_h).view(batch_size, self.hidden_size, 1) # (batch_size, dim_h, 1)
            encoder_outputs_expand = encoder_outputs.view(seq_len, batch_size, 1, self.hidden_size)
            raw_attention = torch.matmul(encoder_outputs_expand, query).view(seq_len, batch_size) # (seq_len, batch_size)
            raw_attention = raw_attention.permute(1, 0).view_as(mask_tensor) # (batch_size, out_len, seq_len)
            raw_attention.data.masked_fill_(mask_tensor, -float('inf'))
            attention = F.softmax(raw_attention, dim=2)  ##(batch,out_len,seq_len)


        ###############################################

        return predicted_token.permute(1, 0), hidden, attention, updated_decoding_state,token_neg_entropy, selected_token_log_prob





    def forward(self,encoder_embedded, encoder_hidden,encoder_outputs,encoder_lens,target_variable,sample_token):
        self.batch_size = encoder_outputs.size(1)
        total_neg_entropy = 0
        total_seq_prob = 0

        ## set initiate step:
        time = 0
        next_input = torch.FloatTensor(np.zeros((1, self.batch_size, self.output_encoding_size))).to(self.device)
        next_decoding_state = torch.FloatTensor([[0, 0, self.max_decoder_len]]).expand(self.batch_size, 3).contiguous().to(self.device)
        loop_state = True
        previous_hidden = encoder_hidden 
        cum_attention=None

        while time < self.max_decoder_len :
            predicted_token, previous_hidden, attention, next_decoding_state, neg_entropy, log_seq_prob = \
                self._step_by_step_attention_decoder(time=time,
                    embedded= next_input, # imply that out_len=1
                    previous_hidden_state=previous_hidden, 
                    cum_attention=cum_attention,
                    encoder_embedded=encoder_embedded,
                    encoder_outputs=encoder_outputs,
                    encoder_hidden=encoder_hidden,
                    encoder_lens=encoder_lens, decoding_state=next_decoding_state,target_variable= target_variable,sample_token=sample_token)

            if time == 0:
                predicted_tokens = predicted_token
                total_neg_entropy = neg_entropy
                total_seq_prob = log_seq_prob
                attention_total = attention
                cum_attention = attention
            else:
                predicted_tokens = torch.cat((predicted_tokens, predicted_token))
                total_neg_entropy += neg_entropy
                total_seq_prob += log_seq_prob
                attention_total = torch.cat((attention_total, attention), dim=1)
                # cum_attention = cum_attention + attention
                cum_attention = attention

            time +=1
            next_input =self.embedding(predicted_token)
            loop_state = torch.ne(predicted_token, self.EOS_token).any()

        return predicted_tokens, attention_total, total_neg_entropy, total_seq_prob




class attention_seq2seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(attention_seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seqs, input_embedded, input_seq_lens,target_variable,sample_token):
        encoder_outputs, encoder_hidden, txt_embedded = self.encoder(input_seqs, input_embedded, input_seq_lens)
        decoder_results, attention, neg_entropy, log_seq_prob = self.decoder(target_variable=target_variable,
                                                                    encoder_embedded=input_embedded,
                                                                    encoder_hidden= encoder_hidden,
                                                                    encoder_outputs= encoder_outputs,
                                                                    encoder_lens=input_seq_lens, sample_token=sample_token
                                                                             )
        ##using attention from decoder and txt_embedded from the encoder to get the attention weighted text
        ## txt_embedded [seq_len,batch,input_encoding_size]
        ## attention [batch, out_len,seq_len]
        txt_embedded_perm = txt_embedded.permute(1,0,2)
        att_weighted_text = torch.bmm(attention, txt_embedded_perm)
        
        return encoder_hidden, decoder_results, attention, att_weighted_text, neg_entropy, log_seq_prob
        # decoder_results (T, batch_size)
        # attention (batch,out_len,seq_len)
        # att_weighted_text (batch_size, T, input_encoding_size)
        # neg_entropy (batch_size, )
