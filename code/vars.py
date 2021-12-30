import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_dir = "./data/"


VOCAB_SIZE = 10000
DROPOUT = 0.5
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EMBEDDING_DIM = 128
HIDDEN_DIM = 128
LSTM_LAYERS = 1


## tokens variable
PAD_TOKEN = '[PAD]'
UNKNOWN_TOKEN = '[UNK]'
START_TOKEN = '[START]'
STOP_TOKEN = '[STOP]'
SENTENCE_START = '<s>' # start sentence
SENTENCE_END = '</s>' # end sentence