from vars import *
from preprocess_data import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import pytorch_lightning as pl


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
            embedding_dim = EMBEDDING_DIM
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
        # output is shape (1, batch size, HIDDEN_DIM)
        # output size is squeezed to remove the first dimension
        output = self.fc(output.squeeze(0))
        # shape of output is (batch size, VOCAB_SIZE)
        output = self.logsoftmax(output)
        return output, hidden, cell


class Seq2Seq(pl.LightningModule):
    def __init__(self) -> None:
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.train, self.validation, self.test = Multi30k()

    def forward(self, input_sequence, target_sentence, teacher_force_ratio = 0.5):
        batch_size = input_sequence.shape[1]
        target_len = target_sentence.shape[0]
        target_vocab_size = len(german.vocab)
        hidden, cell = self.encoder(input_sequence)
        outputs = torch.zeros(target_len, batch_size, target_vocab_size)

        # Initialize target sequence with start token
        x = target_sentence[0]
    
        # Looping through the target until target length
        for token_index in range(1, target_len):
            decoder_output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[token_index] = decoder_output
            generated_tokens = decoder_output.argmax(1)
            x = target_sentence[token_index] if random.random() < teacher_force_ratio else generated_tokens # Applying teacher forcing

        return outputs
            
    # configure optimizer
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y, ignore_index = pad_idx)
        self.log("train_loss", loss )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y, ignore_index = pad_idx)
        self.log("validation_loss", val_loss)
        return val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y, ignore_index = pad_idx)
        return loss

"""     def train_dataloader(self):
        # Data loader
        train_loader = torch.utils.data.DataLoader(
            dataset=self.train, batch_size = BATCH_SIZE, num_workers=4, shuffle=False
        )
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            dataset=self.validation, batch_size = BATCH_SIZE, num_workers=4, shuffle=False
        )
        return val_loader

    def test_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            dataset=self.test, batch_size = BATCH_SIZE, num_workers=4, shuffle=False
        )
        return val_loader """