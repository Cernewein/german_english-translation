# Taken from https://pytorch.org/tutorials/beginner/translation_transformer.html
import torch
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import Multi30k
from typing import Iterable, List

SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'

# Place-holders
token_transformer = {}
vocab_transformer = {}
sequential_text_transform = {}

# Create source and target language token_transformer.
token_transformer[SRC_LANGUAGE] = get_tokenizer('spacy', language='de_core_news_sm')
token_transformer[TGT_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')


# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']





# helper function in order to perform sequential transformations on text
# If for example transforms are tokenization, conversion to numerical values, conversion to tensor,
# this function returns a function taking txt_input as input that will apply these transformations sequentially on txt_input
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

# helper function to yield list of tokens
def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

    for data_sample in data_iter:
        yield token_transformer[language](data_sample[language_index[language]])

for language in [SRC_LANGUAGE, TGT_LANGUAGE]:
    # Training data Iterator
    train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    vocab_transformer[language] = build_vocab_from_iterator(yield_tokens(train_iter, language),
                                                    min_freq=1,
                                                    specials=special_symbols,
                                                    special_first=True)

    # Set UNK_IDX as the default index. This index is returned when the token is not found.
    # If not set, it throws RuntimeError when the queried token is not found in the Vocabulary.
    vocab_transformer[language].set_default_index(UNK_IDX)
    
    sequential_text_transform[language] = sequential_transforms(token_transformer[language], #Tokenization
                                               vocab_transformer[language], #Numericalization
                                               tensor_transform) # Add BOS/EOS and create tensor

# function to collate data samples into batch tensors
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        # Transforming each sentence in batch into numbers based on vocab contained in a tensor
        src_batch.append(sequential_text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(sequential_text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

    # Padding tensors so that all are of equal length
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch

SRC_VOCAB_SIZE = len(vocab_transformer[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_transformer[TGT_LANGUAGE])