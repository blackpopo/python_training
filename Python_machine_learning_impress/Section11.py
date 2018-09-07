
# coding: utf-8

# In[ ]:


#classtaring no tutors
#k-means algorithm << protoptype-base 
#other base >> hierarchical base, density-base
#elbow method, silhoutte plot
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=150, n_features=2,
                 centers=3, cluster_std=0.5,
                 shuffle=True, random_state=0)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(X[:, 0], X[:, 1], c="white", marker="o", edgecolor="black", s=50)
plt.grid()
plt.tight_layout()
plt.show()


# In[ ]:


#k-means 1,select centroids
#2, attach centroids
#3, move centroids
#4, repeat from 2 to 3 till eta or max_iter
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3, init="random", n_init=10,
           max_iter=300, tol=1e-04, random_state=0)
y_km = km.fit_predict(X)


# In[ ]:


plt.scatter(X[y_km==0,0], X[y_km==0,1], s=50, c="lightgreen", 
            edgecolor="black", marker="s", label="cluster 1")
plt.scatter(X[y_km==1,0], X[y_km==1,1], s=50, c="orange", 
            edgecolor="black", marker="s", label="cluster 2")
plt.scatter(X[y_km==2,0], X[y_km==2,1], s=50, c="lightblue", 
            edgecolor="black", marker="s", label="cluster 3")
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1],
           s=250, marker="*", c="red", edgecolor="black", label="centroids")
plt.legend(scatterpoints=1)
plt.grid()
plt.tight_layout()
plt.show()


# In[ ]:


#k-means++ method
#1 dataset M initialized
#2 select ransom mu
#3 evaluate minimam sentroids
#4 use probability distribution
#init="k-means++"
#fuzzy clustering >> fuzzy k-means
print("Distribution: {:2f}".format(km.inertia_))


# In[ ]:


#elbow method
distortion = []
for i in range(1,11):
    km =KMeans(n_clusters=i, init="k-means++", n_init=10, max_iter=300, random_state=0)
    km.fit(X)
    distortion.append(km.inertia_)
    
plt.plot(range(1,11), distortion, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("Distortion")
plt.tight_layout()
plt.show()


# In[ ]:


#silhouette analysis
km = KMeans(n_clusters=3, init="k-means++", n_init=10, tol=1e-04, max_iter=300,
           random_state=0)
y_km = km.fit_predict(X)

import numpy as np
from matplotlib import cm
from sklearn.metrics import silhouette_samples
cluster_labels = np.unique(y_km)
print(cluster_labels)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(X, y_km, metric="euclidean")
print(silhouette_vals)
y_ax_lower, y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km == c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(float(i) / n_clusters)
    plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0,
            edgecolor="none", color=color)
    yticks.append((y_ax_lower + y_ax_upper) / 2.0)
    y_ax_lower += len(c_silhouette_vals)

silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg, color="red", linestyle="--")
plt.yticks(yticks, cluster_labels + 1)
plt.ylabel("Cluster")
plt.xlabel("Silhouette coefficient")
plt.tight_layout()
plt.show()


# In[ ]:


km = KMeans(n_clusters=2, init="k-means++", n_init=10, max_iter=300,
           tol=1e-04, random_state=0)
y_km = km.fit_predict(X)
plt.scatter(X[y_km==0, 0], X[y_km==0, 1], s=50, c="lightgreen",
           edgecolor="black", marker="s", label="cluster1")
plt.scatter(X[y_km==1, 0], X[y_km==1, 1], s=50, c="orange",
           edgecolor="black", marker="s", label="cluster2")
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
           s=250, marker="*", c="red", label="centroids")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


# In[ ]:


cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
silhouetters = silhouette_samples(X, y_km, metric="euclidean")
y_ax_lower, y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km==c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(float(i) / n_clusters)
    plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0, 
            edgecolor="none", color=color)
    yticks.append((y_ax_lower + y_ax_upper) / 2.0)
    y_ax_lower += len(c_silhouette_vals)
    #cm.jet change coloer default map
    
silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg, color="red", linestyle="--")
plt.yticks(yticks, cluster_labels + 1)
plt.ylabel("Cluster")
plt.xlabel("silhouette coefficient")
plt.tight_layout()
plt.show()


# In[ ]:


#hierarchical clustering
#agglomerative, divisive
import pandas as pd 
import numpy as np
np.random.seed(123)
variables = ["X", "Y", "Z"]
labels = ["id_0", "id_1", "id_2", "id_3", "id_4"]
X = np.random.random_sample([5,3])*10
#each elements * 10
df = pd.DataFrame(X, columns=variables, index=labels)
df


# In[ ]:


#SciPy >> spatial.distance module >> pdist function
from scipy.spatial.distance import pdist, squareform
row_dist = pd.DataFrame(squareform(pdist(df, metric="euclidean")),
                       columns=labels, index=labels)
row_dist


# In[ ]:


#agglomerative >>scipy.cluster.hierarchy submodule >> linkage function
from scipy.cluster.hierarchy import linkage
help(linkage)


# In[ ]:


row_clusters = linkage(pdist(df, metric="euclidean"), method="complete")
pd.DataFrame(row_clusters,
            columns=["row label 1", "row label 2", "distance", "No."],
            index=["cluster of {}".format(i+1) for i in range(row_clusters.shape[0])])


# In[ ]:


from scipy.cluster.hierarchy import dendrogram
row_dendr = dendrogram(row_clusters, labels=labels)
plt.ylabel("Euclidean distance")
plt.tight_layout()
plt.show()


# In[ ]:


#heat map
fig = plt.figure(figsize=(8,8), facecolor="white")
axd = fig.add_axes([0.09, 0.1, 0.2, 0.6])
row_dendr = dendrogram(row_clusters, orientation="left")
df_rowclust = df.iloc[row_dendr["leaves"][::-1]]
axm = fig.add_axes([0.23, 0.1, 0.6, 0.6])
cax = axm.matshow(df_rowclust, interpolation="nearest", cmap="hot_r")
axd.set_xticks([])
axd.set_yticks([])
for i in axd.spines.values():
    i.set_visible(False)
fig.colorbar(cax)
axm.set_xticklabels([" "] + list(df_rowclust.columns))
axm.set_yticklabels([" "] + list(df_rowclust.index))
plt.show()


# In[ ]:


#AgglomerativeClustering
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=3, affinity="euclidean", linkage="complete")
labels = ac.fit_predict(X)
print("Cluster labels: {}".format(labels))


# In[ ]:


ac = AgglomerativeClustering(n_clusters=2, affinity="euclidean", linkage="complete")
labels = ac.fit_predict(X)
print("Cluster labels: {}".format(labels))


# In[ ]:


#DBSCAN >> density based spatial clustering of application with noise
#labeling as core, border and noise
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
plt.scatter(X[:,0], X[:,1])
plt.tight_layout()
plt.show()


# In[ ]:


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,3))
km = KMeans(n_clusters=2, random_state=0)
y_km = km.fit_predict(X)
ax1.scatter(X[y_km==0, 0], X[y_km==0, 1], c="lightgreen",
            edgecolor="black", marker="o", s=40, label="cluster1")
ax1.scatter(X[y_km==1, 0], X[y_km==1, 1], c="red",
            edgecolor="black", marker="s", s=40, label="cluster2")
ax1.set_title("K-means clustering")

ac = AgglomerativeClustering(n_clusters=2, affinity="euclidean", linkage="complete")
y_ac = ac.fit_predict(X)
ax2.scatter(X[y_ac==0, 0], X[y_ac==0, 1], c="lightgreen",
            edgecolor="black", marker="o", s=40, label="cluster1")
ax2.scatter(X[y_ac==1, 0], X[y_ac==1, 1], c="red",
            edgecolor="black", marker="s", s=40, label="cluster2")
ax2.set_title("Agglomerative clustering")
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:


from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.2, min_samples=5, metric="euclidean")
y_db = db.fit_predict(X)
plt.scatter(X[y_db==0,0], X[y_db==0,1], c="lightblue", 
           edgecolor="black", marker="o", s=40, label="cluster 1")
plt.scatter(X[y_db==1,0], X[y_db==1,1], c="orange",
           edgecolor="black", marker="s", s=40, label="cluster 2")
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:


#curse of demention

