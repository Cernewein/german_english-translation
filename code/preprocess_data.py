from lxml import etree
from bs4 import BeautifulSoup as bs
import os
import re
from pickle import *
from progress.bar import Bar
from somajo import SoMaJo
from vars import *
from collections import Counter


def load_xml_file(filename:str):
    with open(filename, 'r') as f:
        # Read each line in the file, readlines() returns a list of lines
        content = f.readlines()
        # Combine the lines in the list into a string
        content = "".join(content)
        xml_file = bs(content, "lxml")
    return xml_file


def save_pickle(data, filename):
    filename = os.path.join(data_dir, filename)
    with open(filename, 'wb+') as f:
        dump(data, f)

def process_sentences(sentence_pair):
    english_tokenizer = SoMaJo("en_PTB", split_camel_case=True)
    german_tokenizer = SoMaJo("de_CMC", split_camel_case=True)
    english_tokenized = english_tokenizer.tokenize_text([sentence_pair.find("tuv", {"xml:lang" : "en"}).seg.text])
    german_tokenized = german_tokenizer.tokenize_text([sentence_pair.find("tuv", {"xml:lang" : "de"}).seg.text])
    for english_sentence in english_tokenized:
        english_tokens = [token.text for token in english_sentence]
    for german_sentence in german_tokenized:
        german_tokens = [token.text for token in german_sentence]
    return english_tokens, german_tokens

def load_sentences(source_file):
    data = []
    file = load_xml_file(source_file)
    sentence_pairs = file.find_all("tu")
    bar = Bar("Processing data", max=len(sentence_pairs))
    for sentence_pair in sentence_pairs:
        english_text, german_text = process_sentences(sentence_pair)
        data.append({"german": german_text,
                     "english": english_text})
        bar.next()
    bar.finish()
    return data

def create_vocab(data) :
    english_words = []
    german_words = []
    bar = Bar("Creating vocab files", max=len(data))
    for sentence_pairs in data:
        english_words.extend(sentence_pairs["english"])
        german_words.extend(sentence_pairs["german"])
        bar.next()
    bar.finish()
    top_n_words_german = [word[0] for word in Counter(german_words).most_common(VOCAB_SIZE)]
    top_n_words_english = [word[0] for word in Counter(english_words).most_common(VOCAB_SIZE)]
    with open(data_dir + "vocab_german", "wb") as f:
        dump(top_n_words_german,f)
    with open(data_dir + "vocab_english", "wb") as f:
        dump(top_n_words_english,f)

if __name__ == '__main__':
    filename = "./data/raw/de-en_truncated.xml"
    outfile = "./data/sentences_tokenized.pkl"
    data = load_sentences(filename)
    create_vocab(data)
    save_pickle(data, filename)