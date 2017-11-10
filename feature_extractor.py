import pandas as pd
import numpy as np


class Word:
    def __init__(self, word, position, pos = "UNK"):
        self.actual_word = word
        self.word_position = position
        self.pos = pos
        self.first_capital = False
        if self.actual_word[0].isupper():
            self.first_capital = True

    def print(self, pad_length=16):
        print(str("{:<" + str(pad_length) + "}").format("Actual Word: "), self.actual_word)
        print(str("{:<" + str(pad_length) + "}").format("Position: "), self.word_position)
        print(str("{:<" + str(pad_length) + "}").format("Part of Speech: "), self.pos)
        print(str("{:<" + str(pad_length) + "}").format("Is Capital: "), str(self.first_capital))


def word_featurizer(word):
    word.print()


word_featurizer(Word("Testing", 0))
