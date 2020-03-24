# K-means Clustering

# Import the Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import Dataset
dataset = pd.read_csv('fifa20rating.csv')
X = dataset.iloc[:, [5, 6]].values


np.random.seed(200)
k = 3
#centroids[i] = [x,y]
centroids = {
    i+1: [np.random.randint(0,80), np.random.randint(0, 80)]
    for i in range(k)
}

# Assignment Stage
def assignment(X, centroids):
    for i in centroids.keys():
        #
        X['distance_from_{}'.format(i)] = (
            np.sqrt(
                (X['x'] - centroids[i][0]) ** 2
                + (X['y'] - centroids[i][1]) ** 2
            )
        )
    centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
    X['closest'] = X.loc[:, centroid_distance_cols].idxmin(axis=1)
    X['closest'] = X['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    return X



# Using the elbow method
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init= 'k-means++', random_state= 52)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('wcss')
plt.show()

# Fitting K-Means to dataset
kmeans = KMeans(n_clusters= 5, init= 'k-means++', random_state= 52)
y_kmeans = kmeans.fit_predict(X)

#Visualiting The Clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of Fifa 20 FootballPlayer')
plt.xlabel('Potential Player')
plt.ylabel('Overall Player')
plt.legend()
plt.show()