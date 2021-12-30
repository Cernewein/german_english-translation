from vars import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from collections import OrderedDict

class Encoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(DROPOUT)
        self.embedding = nn.Embedding(
            num_embeddings = VOCAB_SIZE,
            embedding_dim = EMBEDDING_DIM
        )
        self.lstm = nn.LSTM(
            input_size = EMBEDDING_DIM,
            hidden_size = HIDDEN_DIM,
            num_layers = LSTM_LAYERS,
            bidirectional = False,
            dropout = DROPOUT
        )

    def forward(self, input):
        # Encoder takes whole sequence
        # input shape: (seq_length, batch_size)
        embedded = self.dropout(self.embeddings(input))
        # embedding shape: (seq_length, batch_size, EMBEDDING_SIZE)

        output, (hidden, cell) = self.lstm(embedded)

        return hidden, cell


class Decoder(pl.LightningModule):
    def __init__(self):
        super(Decoder, self).__init__()
        self.dropout = nn.Dropout(DROPOUT)
        self.embeddings = nn.Embedding(
            num_embeddings = VOCAB_SIZE,
            embedding_dim=EMBEDDING_DIM
        )
        self.lstm = nn.LSTM(
            input_size = EMBEDDING_DIM,
            hidden_size = HIDDEN_DIM,
            num_layers = LSTM_LAYERS,
            bidirectional = False,
            dropout = DROPOUT
        )
        self.fc = nn.Linear(
            in_features = HIDDEN_DIM,
            out_features = VOCAB_SIZE
        )
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, cell):
        # input shape is batch size, we need to transform it into (1, batch size)
        # Decoder takes in one word at a time
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embeddings(input))
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        output = self.fc(output.squeeze(1))
        output = self.logsoftmax(output)
        return output, (hidden, cell)


class Seq2Seq(pl.LightningModule):
    def __init__(self) -> None:
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, input_sequence, ):
        encoder_output, (encoder_hidden, encoder_cell) = self.encoder(input)
        decoder_output, (decoder_hidden, decoder_cell) = self.decoder()