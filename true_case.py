import pandas as pd
import numpy as np
from feature_extractor import vectorize_text
from sklearn.tree import DecisionTreeClassifier

text = '''
Packages are namespaces which contain multiple packages and modules themselves. They are simply directories, but with a twist.

Each package in Python is a directory which MUST contain a special file called __init__.py. This file can be empty, and it indicates that the directory it contains is a Python package, so it can be imported the same way a module can be imported.

If we create a directory called foo, which marks the package name, we can then create a module inside that package called bar. We also must not forget to add the __init__.py file inside the foo directory.
'''

text_vectors = pd.DataFrame(vectorize_text(text), columns=["first_word", "proper_noun", "label"])

X = text_vectors.iloc[:, 0:2]
Y = text_vectors.iloc[:, 2]

clf = DecisionTreeClassifier()