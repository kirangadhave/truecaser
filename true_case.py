import pandas as pd
import numpy as np
import sys
from feature_extractor import vectorize_text
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from nltk.corpus import gutenberg

train_text = gutenberg.raw('austen-emma.txt')

test_text = gutenberg.raw('austen-sense.txt')

# print(len(test_text))
# print(len(test_text.lower()))

# Create training vectors
train_vectors = pd.DataFrame(vectorize_text(test_text)[1], columns=["first_word", "proper_noun", "label"])
X_train = train_vectors.iloc[:, 0:2].values
Y_train = train_vectors.iloc[:, 2].values

# Create test vectors
test_vectors = pd.DataFrame(vectorize_text(test_text)[1], columns=["first_word", "proper_noun", "label"])
X_test = test_vectors.iloc[:, 0:2].values
a = vectorize_text(test_text)[0]
Y_test = test_vectors.iloc[:, 2].values
test_vectors = pd.DataFrame(vectorize_text(test_text, True)[1], columns=["first_word", "proper_noun", "label"])
X_test = test_vectors.iloc[:, 0:2].values
b = vectorize_text(test_text, True)[0]
# a.sort()
# b.sort()
# print(list(zip(list(set(a)), list(set(b)))))

# mask = np.ones(len(Y_test), dtype = bool)
# mask[[0]] = False
# # mask[[len(Y_test) - 1]] = False
# Y_test = Y_test[mask]


# Train Support Vector Machine
clf = svm.SVC(kernel='linear', gamma = 10)
clf.fit(X_train, Y_train)
preds = clf.predict(X_test)

print(len(preds), len(Y_test))

print(accuracy_score(Y_test, preds)*100)