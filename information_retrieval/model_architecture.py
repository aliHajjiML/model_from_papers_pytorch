import torch.nn as nn
import torch.nn.functional as F
import torch
class EncoderRNN(nn.Module):

    def __init__(self, input_size, hidden_size, token_numbers, embedding, n_layers=1, dropout=0):
        # call the init function of nn.Module
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.input_size = input_size
        self.token_numbers = token_numbers
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.gru = nn.GRU(input_size = self.input_size, hidden_size = self.hidden_size, num_layers = self.n_layers, dropout= dropout, bidirectional=True, batch_first= True)
        self.conv_k1 = nn.Conv1d(in_channels = self.token_numbers, out_channels = self.token_numbers, kernel_size = 2, padding= 0, padding_mode = 'zeros', stride = 2)
        self.conv_k2 = nn.Conv1d(in_channels = self.token_numbers, out_channels = self.token_numbers, kernel_size = 3, padding= 1, padding_mode = 'zeros', stride = 2)
        self.conv_k3 = nn.Conv1d(in_channels = self.token_numbers, out_channels = self.token_numbers, kernel_size = 4, padding= 1, padding_mode = 'zeros', stride = 2)
        self.max_pooling = nn.MaxPool2d(kernel_size= (self.token_numbers,1))
        self.avg_pooling = nn.AvgPool2d(kernel_size= (self.token_numbers,1))


    def forward(self, input_seq, input_lengths, hidden= None):

        # input_seq is word index (BatchSize*SeqMaxLength)
        # the output should take this form (BatchSize*SeqMaxLength*EmbeddingDim)
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module
        # input lengths: Max length of each sentence in the batch (BatchSize)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # Concat bidirectional GRU outputs
        outputs = torch.cat((outputs[:, :, :self.hidden_size],outputs[:, : ,self.hidden_size:]), 2)
        # Permute dimension
        outputs_reshape = outputs.permute(1, 0, 2)
        # Apply multiple convolution on hidden states
        outputs_reshape_relu_k1 = F.relu(self.conv_k1(outputs_reshape))
        outputs_reshape_relu_k2 = F.relu(self.conv_k2(outputs_reshape))
        outputs_reshape_relu_k3 = F.relu(self.conv_k3(outputs_reshape))
        # Apply Max Pooling
        outputs_max_pooling_k1 = self.max_pooling(outputs_reshape_relu_k1)
        outputs_max_pooling_k2 = self.max_pooling(outputs_reshape_relu_k2)
        outputs_max_pooling_k3 = self.max_pooling(outputs_reshape_relu_k3)
        # Apply Average Pooling to get level 1 and 2 encoding vectors, and concatenation to get level 3 encoding vector.
        outputs_level1_encoding = self.avg_pooling(embedded)
        outputs_level2_encoding = self.avg_pooling(outputs_reshape)
        outputs_level3_encoding = torch.cat((outputs_max_pooling_k1[:, :, :],outputs_max_pooling_k2[:, : , :], outputs_max_pooling_k3[:, : , :]), 2)

        return outputs_level1_encoding, outputs_level2_encoding, outputs_level3_encoding