import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_dir = "./data/"


VOCAB_SIZE = 10000
DROPOUT = 0.1

## tokens variable
PAD_TOKEN = '[PAD]'
UNKNOWN_TOKEN = '[UNK]'
START_TOKEN = '[START]'
STOP_TOKEN = '[STOP]'
SENTENCE_START = '<s>' # start sentence
SENTENCE_END = '</s>' # end sentence