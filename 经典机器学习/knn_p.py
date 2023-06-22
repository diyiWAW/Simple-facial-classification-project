import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier

X, y = make_classification(n_samples=1000, n_features=2, n_informative=2,
                           n_redundant=0, n_clusters_per_class=1)
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()