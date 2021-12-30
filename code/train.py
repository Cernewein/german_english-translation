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

if __name__ == '__main__':
    model = Seq2Seq()
    
    logger = TensorBoardLogger("tb_logs", name="my_model")
    trainer = Trainer(logger = logger, max_epochs = NUM_EPOCHS)
    trainer.fit(model, train_iterator, valid_iterator)

