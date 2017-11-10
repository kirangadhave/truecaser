import pandas as pd
import numpy as np


class Word:
    def __init__(self, word, position, pos = "UNK"):
        self.actual_word = word
        self.word_position = position
        self.pos = pos

    def print(self, pad_length=16):
        print(str("{:<" + str(pad_length) + "}").format("Actual Word: "), self.actual_word)
        print(str("{:<" + str(pad_length) + "}").format("Position: "), self.word_position)
        print(str("{:<" + str(pad_length) + "}").format("Part of Speech: "), self.pos)


def word_featurizer(word):
    word.print()


def main():
    word_featurizer(Word("testing", 0))


if __name__ == "__main__":
    main()