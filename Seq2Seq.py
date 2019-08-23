import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']

word_list = " ".join(sentences).split()
word_list = list(set(word_list))
word_dict = {w: i for i, w in enumerate(word_list)}
number_dict = {i: w for i, w in enumerate(word_list)}
n_class = len(word_dict)  # vocab list

# Parameter
n_hidden = 128

def make_batch(sentences):
    input_batch = [np.eye(n_class)[[word_dict[n] for n in sentences[0].split()]]]
    output_batch = [np.eye(n_class)[[word_dict[n] for n in sentences[1].split()]]]
    target_batch = [[word_dict[n] for n in sentences[2].split()]]

    # make tensor
    return torch.Tensor(input_batch), torch.Tensor(output_batch), torch.LongTensor(target_batch)

class seq2seq(nn.Module):
    def __init__(self):
        super(seq2seq, self).__init__()
        self.encoder = nn.RNN(n_class, n_hidden, batch_first = True)
        self.decoder = nn.RNN(n_class, n_hidden, batch_first = True)
        self.attn = nn.Linear(n_hidden, n_hidden)
        self.out = nn.Linear(n_hidden * 2, n_class)

    def get_att_weight(self, encoder_hidden_states, decoder_hidden_state):
        n_batch = encoder_hidden_states.size(0)
        n_step = encoder_hidden_states.size(1) #[batch, seq_len, hidden_size]
        attn_scores = torch.zeros(n_batch, n_step)
        for i in range(n_batch, n_step):
            temp_encoder_hidden_state = encoder_hidden_states[:, i, :].unsqueeze(1)
            attn_scores[:, i] = self.get_att_score(temp_encoder_hidden_state, decoder_hidden_state)
        return F.softmax(attn_scores, dim = 1) #[batch, seq_len]
    
    
    def get_att_score(self, encoder_hidden_state, decoder_hidden_state):
        encoder_hidden_state = self.attn(encoder_hidden_state)
        #[batch, 1, hidden_size] * [batch, hidden_size, 1]
        score = torch.matmul(encoder_hidden_state, decoder_hidden_state.transpose(1, 2))
        score = score.squeeze()
        return score #[batch]

    def forward(self, encoder_inputs, decoder_inputs):
        encoder_hidden_states, hidden_state= self.encoder(encoder_inputs)
        """
        encoder_hidden_states:[ batch, seq_len, num_directions * hidden_size]
        hidden_state:[ batch, num_layers * num_directions, hidden_size] ????????????????
        """
        batch_size = decoder_inputs.size(0)
        n_step = decoder_inputs.size(1) # seq_len
        output = torch.empty([batch_size, n_step, n_class])
        for i in range(n_step):
            temp_decoder_input = decoder_inputs[:, i, :].unsqueeze(1)
            #print('*********',aa.size(), hidden_state.size())
            decoder_output, hidden_state = self.decoder(temp_decoder_input, hidden_state)
            attn_weight = self.get_att_weight(encoder_hidden_states, decoder_output)
            attn_weight = attn_weight.unsqueeze(1)
            #[batch, 1, n_step] * [batch, n_step, hidden_size]
            context = torch.matmul(attn_weight, encoder_hidden_states)
            context = context.squeeze(1)
            decoder_output = decoder_output.squeeze(1)
            full_state = torch.cat((context, decoder_output), 1)
            output[:, i, :] = self.out(full_state)
        return output

model = seq2seq()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

input_batch, output_batch, target_batch = make_batch(sentences)
#print(input_batch.size(),output_batch.size(), target_batch.size())

#training
for epoch in range(1000):
    output = model(input_batch, output_batch)
    n_step = input_batch.size(1)
    loss = 0
    for i in range(n_step):
        loss += criterion(output[:, i, :], target_batch[:,i])
    if (epoch + 1) % 200 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


            

        