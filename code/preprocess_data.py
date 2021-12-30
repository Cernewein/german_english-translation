from torchtext.datasets import Multik30k
from torchtext.data import Field, BucketIterator
import spacy
from vars import *

# Load spacy languages models
spacy_ger = spacy.load('de_core_news_sm')
spacy_eng = spacy.load('en_core_web_sm')

# Define tokenization
def tokenizer_ger(text):
    return [token.text for token in spacy_ger.tokenizer(text)]

def tokenizer_eng(text):
    return [token.text for token in spacy_eng.tokenizer(text)]

# Define torchtext Fields
german = Field(tokenize = tokenizer_ger, use_vocab = True, lower = True, init_token = '<sos>', eos_token = '<eos>')
english = Field(tokenize = tokenizer_eng, use_vocab = True, lower = True, init_token = '<sos>', eos_token = '<eos>')

# Load Multi30k dataset
train_data, validation_data, test_data = Multik30k.splits(exts = ('.de', '.en'), fields = (german, english))

# Build Vocab

german.build(train_data, max_size = VOCAB_SIZE, min_freq = 2)
english.build(train_data, max_size = VOCAB_SIZE, min_freq = 2)