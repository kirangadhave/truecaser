import pandas as pd
import numpy as np
from feature_extractor import vectorize_text
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

train_text = '''
Packages are namespaces which contain multiple packages and modules themselves. They are simply directories, but with a twist.

Each package in Python is a directory which MUST contain a special file called __init__.py. This file can be empty, and it indicates that the directory it contains is a Python package, so it can be imported the same way a module can be imported.

If we create a directory called foo, which marks the package name, we can then create a module inside that package called bar. We also must not forget to add the __init__.py file inside the foo directory.
'''

test_text = '''
While drawing in class to avoid listening to a story from his painfully boring teacher at Stagwood School, 12-year old Cal sees a frog staring at him through the window. Odder than that is the fact that this frog happens to be wearing glasses. 

Cal and his best friend, the tactless but loyal Soy, learn that the frog (who prefers the name Deli) has sought them out for a reason. When a school administrator named Ream reveals himself to be a dragon, the boys discover that fairytales are real, and that there is magic afoot in Stagwood. With Ream on their tail, the trio must unearth a powerful tool protected by riddles and rile (the magic that fuels nightmares) to save the fate of all fairytales past. Their only means on conveyance, Cal's now-flying bed, takes them on a journey beyond the home of the fairies (a cloud floating somewhere over Iceland) to set things right. But, before Cal can defeat Ream and his kidnapped army of fairies, he has to deal with Soy's knack for arguing with magical creatures, discover the truth about Deli's identity, and earn his place as the hero of the story. 
 

The Guardians of Lore is a middle grade novel that centers around two life-long friends, infusing humor and fantasy-based riddles into a modern fairytale.
'''

train_vectors = pd.DataFrame(vectorize_text(train_text), columns=["first_word", "proper_noun", "label"])
X_train = train_vectors.iloc[:, 0:2].values
Y_train = train_vectors.iloc[:, 2].values

test_vectors = pd.DataFrame(vectorize_text(test_text), columns=["first_word", "proper_noun", "label"])
X_test = test_vectors.iloc[:, 0:2].values
Y_test = test_vectors.iloc[:, 2].values
test_vectors = pd.DataFrame(vectorize_text(test_text.lower()), columns=["first_word", "proper_noun", "label"])
X_test = test_vectors.iloc[:, 0:2].values


clf = DecisionTreeClassifier(criterion='gini', random_state=100)
clf.fit(X_train, Y_train)

preds = clf.predict(X_test)

print(accuracy_score(Y_test, preds)*100)