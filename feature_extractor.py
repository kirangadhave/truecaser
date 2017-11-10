import pandas as pd
import numpy as np
from nltk import word_tokenize, sent_tokenize, pos_tag_sents

class Word:
    def __init__(self, word, position, pos = "UNK"):
        self.is_first_word = 0
        self.is_proper_noun = 0
        self.is_capital = 0

        if position == 0:
            self.is_first_word = 1
        if "NNP" in pos:
            self.is_proper_noun = 1
        if word[0].isupper():
            self.is_capital = 1

        self.vector = [self.is_first_word, self.is_proper_noun, self.is_capital]

    def print(self):
        print(self.vector)


def sentence_vectorize(sentence):
    for word in sent_tokenize(sentence):
        print(word)
