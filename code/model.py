from vars import *
from preprocess_data import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import pytorch_lightning as pl
from torch.utils.data import DataLoader

class Encoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(DROPOUT)
        self.embedding = nn.Embedding(
            num_embeddings = SRC_VOCAB_SIZE,
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
        embedded = self.dropout(self.embedding(input))
        # embedding shape: (seq_length, batch_size, EMBEDDING_SIZE)

        output, (hidden, cell) = self.lstm(embedded)

        return hidden, cell


class Decoder(pl.LightningModule):
    def __init__(self):
        super(Decoder, self).__init__()
        self.dropout = nn.Dropout(DROPOUT)
        self.embedding = nn.Embedding(
            num_embeddings = TGT_VOCAB_SIZE,
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
            out_features = TGT_VOCAB_SIZE
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input, hidden, cell):
        # input shape is batch size, we need to transform it into (1, batch size)
        # Decoder takes in one word at a time
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        # output is shape (1, batch size, HIDDEN_DIM)
        # output size is squeezed to remove the first dimension
        output = self.fc(output.squeeze(0))
        # shape of output is (batch size, VOCAB_SIZE)
        output = self.softmax(output)
        return output, hidden, cell


class Seq2Seq(pl.LightningModule):
    def __init__(self) -> None:
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, input_sequence, target_sentence, teacher_force_ratio = 0.5):
        batch_size = input_sequence.shape[1]
        target_len = target_sentence.shape[0]
        target_vocab_size = TGT_VOCAB_SIZE
        hidden, cell = self.encoder(input_sequence)
        outputs = torch.zeros(target_len, batch_size, target_vocab_size)

        # Initialize target sequence with start token
        x = target_sentence[0]
    
        # Looping through the target until target length
        for token_index in range(1, target_len):
            decoder_output, hidden, cell = self.decoder(x, hidden, cell)
            generated_tokens = decoder_output.argmax(1)
            outputs[token_index] = decoder_output
            x = target_sentence[token_index] if random.random() < teacher_force_ratio else generated_tokens # Applying teacher forcing
        return outputs
            
    # configure optimizer
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x, y)
        loss = F.cross_entropy(y_hat.transpose(0,2).transpose(0,1), y.transpose(0,1), ignore_index = PAD_IDX)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size = BATCH_SIZE)
        return loss

    def train_dataloader(self):
        # Data loader
        train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
        train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)
        return train_dataloader


"""     def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y, ignore_index = PAD_IDX)
        self.log("validation_loss", val_loss)
        return val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y, ignore_index = PAD_IDX)
        return loss 

   def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            val_iter, batch_size = BATCH_SIZE, collate_fn = collate_fn
        )
        return val_loader

    def test_dataloader(self):
        test_loader = torch.utils.data.DataLoader(
            test_iter, batch_size = BATCH_SIZE, collate_fn = collate_fn
        )
        return test_loader """

