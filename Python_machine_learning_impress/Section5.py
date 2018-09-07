
# coding: utf-8

# In[ ]:


import pandas as pd
df_wine = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",
    header=None)
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=0.3, stratify=y, random_state=0)
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)


# In[ ]:


import numpy as np
cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print("\nEigenvalues \n{}".format(eigen_vals))
#eigenvalues >> 固有値　eigenvectors >> 固有ベクトル
#cov >> 共分散行列


# In[ ]:


tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
import matplotlib.pyplot as plt
#np.cumsum >> 要素の差分を足し合わせる
plt.bar(range(1,14), var_exp, alpha=0.5, align="center", label="individual explained variance")
plt.step(range(1,14), cum_var_exp, where="mid", label="cumulative explained variance")
plt.ylabel("explained variance ratio")
plt.xlabel("principal component index")
plt.legend(loc="best")
plt.tight_layout()
plt.show()


# In[ ]:


#特徴変換
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i])
              for i in range(len(eigen_vals))]
eigen_pairs.sort(key=lambda k: k[0], reverse=True)
#タプルを作り　代入している
w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
               eigen_pairs[1][1][:, np.newaxis]))
print("matrix W:",w)


# In[ ]:


X_train_std[0].dot(w)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
X_train_pca = X_train_std.dot(w)
colors = ["r","b","g"]
markers = ["s","x","o"]
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train==l, 0], X_train_pca[y_train==l, 1],c=c, label=l, marker=m)
plt.xlabel("pc 1")
plt.ylabel("pc 2")
plt.legend(loc="lower left")
plt.tight_layout()
plt.show()
print(np.unique(y_train))


# In[ ]:


from matplotlib.colors import ListedColormap
def plot_decision_regions(X, y, classifier, resolution=0.2):
    #marker & colormap
    markers = ("s","x", "o", "^","v")
    colors = ("red", "blue", "green", "gray", "cyan")
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                          np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],y=X[y == cl, 1],
                   alpha=0.6, c=cmap(idx),
                   edgecolor="black",marker=markers[idx],
                   label=cl)
        
#colormapのクラスの作成 


# In[ ]:


#PCA >> 主成分分析
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
lr = LogisticRegression()
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
lr.fit(X_train_pca, y_train)
plot_decision_regions(X_train_pca, y_train, classifier=lr)
plt.xlabel("pc1")
plt.ylabel("pc2")
plt.legend(loc="lower left")
plt.tight_layout()
plt.show()


# In[ ]:


plot_decision_regions(X_test_pca, y_test, classifier=lr)
plt.xlabel("pc1")
plt.ylabel("pc2")
plt.legend(loc="lower left")
plt.tight_layout()
plt.show()


# In[ ]:


pca = PCA(n_components=None)
X_train_pca = pca.fit_transform(X_train_std)
pca.explained_variance_ratio_


# In[ ]:


pca = PCA()
X_train_pca = pca.fit_transform(X_train_std)
pca.explained_variance_ratio_
#当時と違いすでに自動ソートされるように実装されている


# In[ ]:


#線形判別分析　データ圧縮
np.set_printoptions(precision=4)
mean_vecs = []
for label in range(1,4):
    mean_vecs.append(np.mean(X_train_std[y_train==label], axis=0))
    print("mv{},{}".format(label, mean_vecs[label-1]))


# In[ ]:


d = 13 
S_W = np.zeros((d,d))
for label, mv in zip(range(1,4), mean_vecs):
    #label >> 1~3
    class_scatter = np.zeros((d,d))
    for row in X_train_std[y_train == label]:
        row, mv = row.reshape(d, 1), mv.reshape(d, 1)
        #row ,mv >> 13 X 1 の　matrix
        class_scatter += (row - mv).dot((row - mv).T)
    S_W += class_scatter
print("scaled within-class scatter matrix,{},{}".format(S_W.shape[0], S_W.shape[1]))


# In[ ]:


print("class label distribution:{}".format(np.bincount(y_train)[1:]))


# In[ ]:


d = 13
S_W = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
    class_scatter = np.cov(X_train_std[y_train==label].T)
    S_W += class_scatter
print("scaled within-class scatter matrix {}x{}".format(S_W.shape[0], S_W.shape[1]))


# In[ ]:


mean_overall = np.mean(X_train_std, axis=0)
d = 13
S_B = np.zeros((d,d))
for i, mean_vec in enumerate(mean_vecs):
    n = X_train[y_train == i +1, :].shape[0]
    mean_vec = mean_vec.reshape(d, 1)
    mean_oveall = mean_overall.reshape(d, 1)
    S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)
print("between-class scatter {}X{}".format(S_B.shape[0], S_B.shape[1]))


# In[ ]:


eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
print(eigen_vals, eigen_vecs)


# In[ ]:


eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
               for i in range(len(eigen_vals))]
#np.abs >> 絶対値　absolute 
eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)
for eigen_val in eigen_pairs:
    print(eigen_val[0])


# In[ ]:


tot = sum(eigen_vals.real)
discr = [(i/tot) for i in sorted(eigen_vals.real, reverse=True)]
cum_discr = np.cumsum(discr)
#discrimity >> 判別性（固有ベクトルの大きさの順番）
plt.bar(range(1, 14), discr, alpha=0.5, align="center", label="individal discrimity")
plt.step(range(1, 14), cum_discr, where="mid", label="cumulative discrimity")
plt.ylabel("discriminability ratio")
plt.xlabel("Liner Discrimination")
plt.ylim([-0.1, 1.1])
plt.legend(loc="best")
plt.tight_layout()
plt.show()


# In[ ]:


w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real,
              eigen_pairs[1][1][:, np.newaxis].real))
print ("matrix W:\n", w)


# In[ ]:


X_train_ida = X_train_std.dot(w)
colors = ["r", "b", "g"]
markers = ["s","x","o"]
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_ida[y_train==l, 0],
               X_train_ida[y_train==l, 1] * (-1),#1 l の九月をしっかりスる。
               c=c , label=l, marker=m)
plt.xlabel("ld 1")
plt.ylabel("ld 2")
plt.tight_layout()
plt.legend(loc="upper right")
plt.show()
#print(X_train_ida,y_train)


# In[ ]:


#PVA + karneltric
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np
def rbf_kernel_pca(X, gamma, n_components):
    #gamma >> tuninng papamator
    #n_componexts >> 主成分の個数
    #X_pc >> 射影データ
    sq_dists = pdist(X, "sqeuclidean")
    #ユークリッド距離の指定
    mat_sq_dists = squareform(sq_dists)
    #実際の計算
    K = exp(-gamma * mat_sq_dists)
    N = K.shape[0]
    one_n = np.ones((N,N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    eigvals, eigvecs = eigh(K)
    eigvals, eigvecs= eigvals[::-1], eigvecs[:,::-1]
    #順番の入れ替え
    X_pc = np.column_stack((eigvecs[:,i] for i in range(n_components)))
    return X_pc
    


# In[ ]:


from sklearn.datasets import make_moons
X, y = make_moons(n_samples=100, random_state=123)
plt.scatter(X[y==0, 0], X[y==0, 1], color="red", marker="^", alpha=0.5)
plt.scatter(X[y==1, 0], X[y==1, 1], color="blue",marker="o", alpha=0.5)
plt.tight_layout()
plt.show()
#線形分離不可能


# In[ ]:


from sklearn.decomposition import PCA 
scikit_pca = PCA(n_components=2)
X_spca = scikit_pca.fit_transform(X)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))
ax[0].scatter(X_spca[y==0, 0], X_spca[y==0, 1],
             color="red", marker="^", alpha=0.5)
ax[0].scatter(X_spca[y==1, 0], X_spca[y==1, 1],
             color="red", marker="^", alpha=0.5)
ax[1].scatter(X_spca[y==0, 0], np.zeros((50,1))+0.02,
             color="red", marker="^", alpha=0.5)
ax[1].scatter(X_spca[y==1, 0], np.zeros((50,1))-0.02,
             color="blue", marker="o", alpha=0.5)
ax[0].set_xlabel("PC1")
ax[0].set_ylabel("PC2")
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
plt.tight_layout()
plt.show()


# In[ ]:


from matplotlib.ticker import FormatStrFormatter
X_kpca = rbf_kernel_pca(X, gamma=15, n_components=2)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7,3))
ax[0].scatter(X_kpca[y==0, 0], X_kpca[y==0, 1],
             color="red", marker="^", alpha=0.5)
ax[0].scatter(X_kpca[y==1, 0], X_kpca[y==1, 1],
             color="blue", marker="o", alpha=0.5)
ax[1].scatter(X_kpca[y==0, 0], np.zeros((50,1))+0.02,
             color="red", marker="^", alpha=0.5)
ax[1].scatter(X_kpca[y==1, 0], np.zeros((50,1))-0.02,
             color="blue", marker="o", alpha=0.5)
ax[0].set_xlabel("pc1")
ax[0].set_ylabel("pc2")
ax[1].set_yticks([])
ax[1].set_ylim([-1,1])
ax[1].set_xlabel("pc1")
plt.show()


# In[ ]:


from sklearn.datasets import make_circles
X, y =make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)
plt.scatter(X[y==0, 0], X[y==0, 1], color="red", marker="^", alpha=0.5)
plt.scatter(X[y==1, 0], X[y==1, 1], color="blue", marker="o", alpha=0.5)
plt.tight_layout()
plt.show()


# In[ ]:


scikit_pca = PCA(n_components=2)
X_pca = scikit_pca.fit_transform(X)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7,3))
ax[0].scatter(X_pca[y==0, 0],X_pca[y==0, 1],
             color="red", marker="^", alpha=0.5)
ax[0].scatter(X_pca[y==1, 0],X_pca[y==1, 1],
             color="blue", marker="o", alpha=0.5)
ax[1].scatter(X_pca[y==0, 0], np.zeros((500,1))+0.02,
             color="red", marker="^", alpha=0.5)
ax[1].scatter(X_pca[y==1, 0], np.zeros((500,1))-0.02,
             color="blue", marker="o", alpha=0.5)
ax[0].set_xlabel("pc1")
ax[0].set_ylabel("pc2")
ax[1].set_xlabel("pc1")
ax[1].set_ylim([-1,1])
ax[1].set_yticks([])
plt.tight_layout()
plt.show()


# In[ ]:


X_kpca = rbf_kernel_pca(X, gamma=15,n_components=2)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7,3))
ax[0].scatter(X_kpca[y==0, 0],X_kpca[y==0, 1],
             color="red", marker="^", alpha=0.5)
ax[0].scatter(X_kpca[y==1, 0],X_kpca[y==1, 1],
             color="blue", marker="o", alpha=0.5)
ax[1].scatter(X_kpca[y==0, 0], np.zeros((500,1))+0.02,
             color="red", marker="^", alpha=0.5)
ax[1].scatter(X_kpca[y==1, 0], np.zeros((500,1))-0.02,
             color="blue", marker="o", alpha=0.5)
ax[0].set_xlabel("pc1")
ax[0].set_ylabel("pc2")
ax[1].set_xlabel("pc1")
ax[1].set_ylim([-1,1])
ax[1].set_yticks([])
plt.tight_layout()
plt.show()


# In[ ]:


from scipy.spatial.distance import pdist, squareform
from scipy import exp
import numpy as np
def rbf_kernel_pca(X, gamma, n_components):
    sq_dists = pdist(X, "sqeuclidean")
    mat_sq_dists = squareform(sq_dists)
    K = exp(-gamma * mat_sq_dists)
    N = K.shape[0]
    one_n = np.ones((N,N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) - one_n.dot(K) + one_n.dot(K).dot(one_n)
    eigvals, eigvecs = eigh(K)
    eigvals, eigvecs =eigvals[::-1], eigvecs[:,::-1]
    alphas = np.column_stack((eigvecs[:,i] for i in range(n_components)))
    lambdas = [eigvals[i] for i in range(n_components)]
    return alphas, lambdas


# In[ ]:


X, y =make_moons(n_samples=100, random_state=123)
alphas, lambdas = rbf_kernel_pca(X, gamma=15, n_components=1)
print(alphas[:5], lambdas)


# In[ ]:


x_new = X[25]
print(x_new)
x_proj = alphas[25]
print(x_proj)
def project_x(x_new, X, gamma, alphas, lambdas):
    pair_dict = np.array([np.sum((x_new-row)**2) for row in X])
    k = np.exp(-gamma*pair_dict)
    return k.dot(alphas / lambdas)


# In[ ]:


x_reproj = project_x(x_new, X, gamma=15, alphas=alphas, lambdas=lambdas)
x_reproj


# In[ ]:


plt.scatter(X[y==0, 0],np.zeros((50)), color="red", marker="^", alpha=0.5)
plt.scatter(X[y==1, 0],np.zeros((50)), color="blue", marker="o", alpha=0.5)
plt.scatter(x_proj, 0, color="black", label="original X[25]",
           marker="^", s=500)
plt.scatter(x_reproj, 0, color="green", label="remapped X[25]",
           marker="x", s=500)
plt.legend(scatterpoints=1)
plt.tight_layout()
plt.show()


# In[ ]:


from sklearn.decomposition import KernelPCA
X, y = make_moons(n_samples=100, random_state=123)
scikit_kpca = KernelPCA(n_components=2, kernel="rbf", gamma=15)
X_kernelpca = scikit_kpca.fit_transform(X)
plt.scatter(X_kernelpca[y==0, 0], X_kernelpca[y==0, 1],
           color="red", marker="^", alpha=0.5)
plt.scatter(X_kernelpca[y==1, 0], X_kernelpca[y==1, 1],
           color="blue", marker="o", alpha=0.5)
plt.xlabel("pc1")
plt.ylabel("pc2")
plt.tight_layout()
plt.show()

