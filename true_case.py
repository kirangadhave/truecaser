import pandas as pd
import numpy as np
from feature_extractor import vectorize_text
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score




# Create training vectors
train_vectors = pd.DataFrame(vectorize_text(test_text)[1], columns=["first_word", "proper_noun", "label"])
print(train_vectors)
X_train = train_vectors.iloc[:, 0:2].values
Y_train = train_vectors.iloc[:, 2].values

# Create test vectors
test_vectors = pd.DataFrame(vectorize_text(test_text)[1], columns=["first_word", "proper_noun", "label"])
X_test = test_vectors.iloc[:, 0:2].values
Y_test = test_vectors.iloc[:, 2].values
test_vectors = pd.DataFrame(vectorize_text(test_text.lower())[1], columns=["first_word", "proper_noun", "label"])
X_test = test_vectors.iloc[:, 0:2].values


# Train Support Vector Machine
clf = svm.SVC(kernel='linear', gamma = 10)
clf.fit(X_train, Y_train)
preds = clf.predict(X_test)
print(accuracy_score(Y_test, preds)*100)