from vars import *
from model import *
from preprocess_data import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from timeit import default_timer as timer
from torch.utils.data import DataLoader

model = Seq2Seq()
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

def translate(model: pl.LightningModule, src_sentence: str, tgt_sentence: str):
    model.eval()
    src = sequential_text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    tgt = sequential_text_transform[SRC_LANGUAGE](tgt_sentence).view(-1, 1)
    tgt_tokens = model(src, tgt, teacher_force_ration = 0)
    return " ".join(vocab_transformer[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")

def train_epoch(model, optimizer):
    model.train()
    losses = 0
    train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    i = 1
    for src, tgt in train_dataloader:

        print(f"Batch number {i}")
        i +=1
        if i > MAX_BATCH_PROCESS:
            break
        logits = model(src, tgt)

        optimizer.zero_grad()

        #model output is shape seq_len, batch_size, target_vocab_size
        # in order to compute the loss, we need to transpose it to 
        #  batch_size, target_vocab_size, seq_len

        # Same logic for target, it is shape seq_len, batch_size and needs to be converted to batch_size, seq_len
        # See https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html?highlight=crossentropyloss#torch.nn.CrossEntropyLoss
        # for more details
        loss = loss_fn(logits.transpose(0,2).transpose(0,1), tgt.transpose(0,1))
        
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(train_dataloader)


def evaluate(model):
    model.eval()
    losses = 0
    print("Model evaluation")
    val_iter = Multi30k(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    i = 1
    for src, tgt in val_dataloader:

        logits = model(src, tgt)
        print(f"Batch number {i}")
        i +=1
        
        optimizer.zero_grad()

        
        loss = loss_fn(logits.transpose(0,2).transpose(0,1), tgt.transpose(0,1))
        
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(val_dataloader)

if __name__ == '__main__':
    logger = TensorBoardLogger("tb_logs", name="my_model")
    trainer = Trainer(logger = logger, max_epochs = NUM_EPOCHS)
    trainer.fit(model)
    trainer.validate(model)
    trainer.test(model)