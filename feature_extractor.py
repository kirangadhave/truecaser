from nltk import word_tokenize, sent_tokenize, pos_tag


class Word:
    def __init__(self, word, position, pos = "UNK"):
        self.word = word
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

def vectorize_text(text):
    vectors = []
    tagged_words = []

    for sentence in sent_tokenize(text):
        for i,word in enumerate(pos_tag(word_tokenize(sentence))):
            vectors.append(Word(word[0], i, word[1]))
    return [w.word for w in vectors],[w.vector for w in vectors]
