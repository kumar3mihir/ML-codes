import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np

iris = datasets.load_iris()

# Slicing the dataset to get the desired features and target
x = iris.data[:, 3:]  # Use only the 4th column (petal width) as the feature
y = (iris.target == 2).astype(int)  # Set the target label to 1 if the flower is Iris Virginica, 0 otherwise

# Train a Logistic Regression classifier
clf = LogisticRegression()
clf.fit(x, y)

# Predict the target probabilities for the given input
example = [[2.6]]
y_prob = clf.predict_proba(example)
print(y_prob)

# Plot the visualization
x_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_prob = clf.predict_proba(x_new)
print(y_prob)
plt.plot(x_new, y_prob[:, 1], label='Iris Virginica')
plt.plot(x_new, y_prob[:, 0], label='Not Iris Virginica')
plt.xlabel("Petal Width")
plt.ylabel("Probability")
plt.legend()
plt.show()







