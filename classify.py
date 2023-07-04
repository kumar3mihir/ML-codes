# loading required modules
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

# loading datasets
iris = datasets.load_iris()
features = iris.data
labels = iris.target
# printing description and features
# print(features[0],labels[0])
# print(iris.DESCR)
# training classifiers
clf = KNeighborsClassifier()
clf.fit(features,labels)

prods = clf.predict([[19, 12, 13, 1]])
print(prods)
