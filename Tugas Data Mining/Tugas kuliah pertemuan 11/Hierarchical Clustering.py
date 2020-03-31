# Import the Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import Dataset
dataset = pd.read_csv('fifa20rating.csv')
X = dataset.iloc[:, [5, 6]].values

import scipy.cluster.hierarchy as sch
# Create dendogram
dendogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendogram')
plt.xlabel('Player')
plt.ylabel('')
plt.show()


from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)
# Visualising the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'blue')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'red')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'purple')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'orange')
plt.title('Clusters of Fifa20 Rating')
plt.xlabel('Potential players')
plt.ylabel('Overall players')
plt.legend()
plt.show()
