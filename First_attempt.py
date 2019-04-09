import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('Cust_Segmentation.csv')

df = df.drop('Address', axis=1)

X = df.values[:, 1:]
X = np.nan_to_num(X)
clus_dataset = StandardScaler().fit_transform(X)

k_means = KMeans(init='k-means++', n_clusters=3, n_init=12)
k_means.fit(clus_dataset)
labels = k_means.labels_
print(labels)

df['clus_km'] = labels

_ = plt.scatter(df['Age'], df['Income'])

_ = plt.scatter(X[:, 0], X[:, 3], c=labels.astype(np.float), alpha=0.5)
plt.show()


from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0,0,.95,1], elev=48, azim=134)

plt.cla()
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')

ax.scatter(X[:, 1], X[:, 0], X[:, 3], c=labels.astype(np.float))

def ecdf(s):
    y = np.arange(0, len(s) +1) / len(s)
    x = np.sort(s)
    return x, y

df.groupby('clus_km').mean()

inc = df['Income']
print(df['Income'])
x, y = ecdf(inc)