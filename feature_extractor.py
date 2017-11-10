import pandas as pd
import numpy as np

class Word:
    def __init__(self, word, position, pos = "UNK"):
        self.actual_word = word
        self.word_position = position
        self.pos = pos

    def print(self):
        print(self.actual_word, self.word_position, self.pos)

def word_featurizer(word):
    word.print()

word_featurizer(Word("test_word", 0))
