# import libraries
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random
from scipy.special import expit

class attention(nn.Module):
    def __init__(self, hidsize):
        super(attention, self).__init__()
        
        self.hidsize = hidsize
        self.match1 = nn.Linear(2*hidsize, hidsize)
        self.match2 = nn.Linear(hidsize, hidsize)
        self.match3 = nn.Linear(hidsize, hidsize)
        self.match4 = nn.Linear(hidsize, hidsize)
        self.to_weight = nn.Linear(hidsize, 1, bias=False)

    def forward(self, hidden_state, out_encoders):
        bat_siz, seq_len, feat_n = out_encoders.size()
        hidden_state = hidden_state.view(bat_siz, 1, feat_n).repeat(1, seq_len, 1)
        matching_inputs = torch.cat((out_encoders, hidden_state), 2).view(-1, 2*self.hidsize)

        x = self.match1(matching_inputs)
        x = self.match2(x)
        x = self.match3(x)
        x = self.match4(x)
        attention_weights = self.to_weight(x)
        attention_weights = attention_weights.view(bat_siz, seq_len)
        attention_weights = F.softmax(attention_weights, dim=1)
        cntxt = torch.bmm(attention_weights.unsqueeze(1), out_encoders).squeeze(1)
        
        return cntxt


class RNN_encoder(nn.Module):
    def __init__(self, input_size, hidsize, drpot_percentage=0.3):
        super(RNN_encoder, self).__init__()

        
        self.input_size = input_size
        self.hidsize = hidsize

       
        self.compress = nn.Linear(in_features=input_size, out_features=hidsize)
        self.drpot = nn.drpot(drpot_percentage)
        self.gru = nn.GRU(input_size=hidsize, hidsize=hidsize, batch_first=True)

    def forward(self, input):
        bat_siz, seq_len, feat_n = input.size()    
        input = input.view(-1, feat_n)
        input = self.compress(input)
        input = self.drpot(input)
        input = input.view(bat_siz, seq_len, self.hidsize) 

        output, hidden_state = self.gru(input)

        return output, hidden_state


class RNN_decoder(nn.Module):
    def __init__(self, hidsize, output_size, vocab_size, word_dim, help=None, drpot_percentage=0.3):
        super(RNN_decoder, self).__init__()

       
        self.hidsize = hidsize 
        self.output_size = output_size 
        self.vocab_size = vocab_size
        self.word_dim = word_dim
        self.help = help

       
        self.embedding = nn.Embedding(vocab_size, word_dim)
        self.drpot = nn.drpot(drpot_percentage)
        self.gru = nn.GRU(hidsize+word_dim, hidsize, batch_first=True)
        self.attention = attention(hidsize)
        self.to_final_output = nn.Linear(hidsize, output_size)


    def forward(self, last_hid_st_encoder, out_encoder, targets=None, mode='train', tr_stps=None):
        
        _, bat_siz, _ = last_hid_st_encoder.size()
        curr_hid_st_decoder = self.initialize_hidden_state(last_hid_st_encoder)
        decoder_curr_inp_word = Variable(torch.ones(bat_siz, 1)).long() 
        decoder_curr_inp_word = decoder_curr_inp_word.cuda() if torch.cuda.is_available() else decoder_curr_inp_word
        sequence_logprob = []
        sequence_pred = []

       
        targets = self.embedding(targets) 
        _, seq_len, _ = targets.size()

        for i in range(seq_len-1): 
            threshold = self._get_teacher_learning_ratio(training_stps=tr_stps)
            curr_inp_word = targets[:, i] if random.uniform(0.05, 0.995) > threshold \
                else self.embedding(decoder_curr_inp_word).squeeze(1)

            
            cntxt = self.attention(curr_hid_st_decoder, out_encoder)
            gru_input = torch.cat([curr_inp_word, cntxt], dim=1).unsqueeze(1)
            gru_output, curr_hid_st_decoder = self.gru(gru_input, curr_hid_st_decoder)
            log_probablity = self.to_final_output(gru_output.squeeze(1))
            sequence_logprob.append(log_probablity.unsqueeze(1))
            decoder_curr_inp_word = log_probablity.unsqueeze(1).max(2)[1]

        
        sequence_logprob = torch.cat(sequence_logprob, dim=1)
        sequence_pred = sequence_logprob.max(2)[1]
        return sequence_logprob, sequence_pred


    def infer(self, last_hid_st_encoder, out_encoder):
        _, bat_siz, _ = last_hid_st_encoder.size()
        curr_hid_st_decoder = self.initialize_hidden_state(last_hid_st_encoder)
        decoder_curr_inp_word = Variable(torch.ones(bat_siz, 1)).long()  # <SOS> (batch x word index)
        decoder_curr_inp_word = decoder_curr_inp_word.cuda() if torch.cuda.is_available() else decoder_curr_inp_word
        sequence_logprob = []
        sequence_pred = []
        seq_len_assum = 28
        
        for i in range(seq_len_assum-1):
            curr_inp_word = self.embedding(decoder_curr_inp_word).squeeze(1)
            cntxt = self.attention(curr_hid_st_decoder, out_encoder)
            gru_input = torch.cat([curr_inp_word, cntxt], dim=1).unsqueeze(1)
            gru_output, curr_hid_st_decoder = self.gru(gru_input, curr_hid_st_decoder)
            log_probablity = self.to_final_output(gru_output.squeeze(1))
            sequence_logprob.append(log_probablity.unsqueeze(1))
            decoder_curr_inp_word = log_probablity.unsqueeze(1).max(2)[1]

        sequence_logprob = torch.cat(sequence_logprob, dim=1)
        sequence_pred = sequence_logprob.max(2)[1]
        return sequence_logprob, sequence_pred


    def initialize_hidden_state(self, last_enhidst):
        if last_enhidst is None:
            return None
        else:
            return last_enhidst


    def _get_teacher_learning_ratio(self, training_stps):
        return (expit(training_stps/20 +0.85))

class MODELS(nn.Module):
    def __init__(self, encoder, decoder):
        super(MODELS, self).__init__()
        self.encoder = encoder
        self.decoder = decoder


    def forward(self, avi_feats, mode, target_sentences=None, tr_stps=None):
        out_encoders, last_hid_st_encoder = self.encoder(avi_feats)
        if mode == 'train':
            sequence_logprob, sequence_pred = self.decoder(last_hid_st_encoder = last_hid_st_encoder, out_encoder = out_encoders,
                targets = target_sentences, mode = mode, tr_stps=tr_stps)
        elif mode == 'inference':
            sequence_logprob, sequence_pred = self.decoder.infer(last_hid_st_encoder=last_hid_st_encoder, out_encoder=out_encoders)
        else:
            raise KeyError('mode is not valid')
        return sequence_logprob, sequence_pred

    
if __name__ == '__main__':
    from train_seq2seq import dictionary

    json_file = 'MLDS_hw2_1_data/testing_label.json'
    numpy_file = 'MLDS_hw2_1_data/testing_data/feat'
    help = Vocabulary(json_file, min_word_count=5)
    input_data = Variable(torch.randn(3, 80, 4096).view(-1, 80, 4096))
    encoder = RNN_encoder(input_size=4096, hidsize=512)
    decoder = RNN_decoder(hidsize=512, output_size=1700, vocab_size=1700, word_dim=128, help=help)
    model = MODELS(encoder=encoder, decoder=decoder)
    ground_truth = Variable(torch.rand(3, 27)).long()
    for stp in range(50, 100):
        seq_prob, seq_predict = model(avi_feats=input_data, mode='train', target_sentences=ground_truth, tr_stps=stp)
        if stp % 10 == 0:
            print(seq_prob.size())
            print(seq_predict.size())


















