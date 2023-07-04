# Train a logistic regression classifier to check whether a
# flower is a iris Virginica or not
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np

iris = datasets.load_iris()
# slicing this states that keep all the rows 3rd column in x
x = iris['data'][:,3:]
# print(iris['data'])
# print(x)
y = (iris['target'] == 2).astype(int)
# print(y)

# Train a Logistic regression classifier
clf = LogisticRegression()
clf.fit(x,y)
example = clf.predict([[2.6]]);
print(example)


# using matplotlib to plot the visualisation
x_new = np.linspace(0,3,1000).reshape(-1,1)
y_prob = clf.predict_proba(x_new)
print(y_prob)
plt.plot(x_new,y_prob[:,1],"g--",label="virginica")
plt.show()

# print(list(iris.keys()))
# print(iris['data']

# target ->label
# attributes -> features
# print(iris['target'])
# print(iris['DESCR'])
