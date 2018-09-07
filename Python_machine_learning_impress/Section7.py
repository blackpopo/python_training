
# coding: utf-8

# In[ ]:


#ensamble method
from scipy.special import comb
#comb 二項演算子　組み合わせのやつ
import math
def ensamble_error(n_classifier, error):
    k_start = int(math.ceil(n_classifier / 2.0))
    probs = [comb(n_classifier, k)* 
            error**k * (1-error)**(n_classifier-k)
            for k in range(k_start, n_classifier+1)]
    return sum(probs)
ensamble_error(n_classifier=11, error=0.25)


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
error_range = np.arange(0.0, 1.01, 0.01)
ens_errors = [ensamble_error(n_classifier=11, error=error) 
             for error in error_range]

plt.plot(error_range, ens_errors, label="ensamble errors", linewidth=2)
plt.plot(error_range, error_range, label="base error", linestyle="--", linewidth=2)
plt.xlabel("base error")
plt.ylabel("base / ensamble error")
plt.legend(loc="upper left")
plt.grid(alpha=0.3)
plt.show()


# In[ ]:


#多数決分類気
import numpy as np
#np.bincount >> array中の各々の整数の数の個数を合計したarrayを作る
# from random import shuffle
# X = np.arange(1,10).tolist()
# shuffle(X)
# print(X, np.bincount(np.array(X))) >>> [9, 5, 1, 7, 3, 8, 2, 4, 6] [0 1 1 1 1 1 1 1 1 1]
np.argmax(np.bincount([0, 0, 1], weights=[0.2, 0.2, 0.6]))
# np.bincount([0, 0, 1],weights=[0.2, 0.2, 0.6]) >> [0.4, 0.6]


# In[ ]:


ex = np.array([[0.9, 0.1],
              [0.8, 0.2],
              [0.4, 0.6]])
p = np.average(ex, axis=0, weights=[0.2, 0.2, 0.6])
p


# In[ ]:


from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import six
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
import numpy as np
import operator 
#sklearn.base >> scikitlearning の　base class が　baseestimator
#classifiermaxin >> 　class の　継承により働くclass
#label encoder >> いろんなものを　分類数に応じて変換するもの
#externals >> I/O の制御をする
#_name_estimator >> 不明
#operator >> 組み込み関数の制御

class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
    
    def __init__(self, classifiers, vote="classlabel", weights=None):
        self.classifiers = classifiers
        self.named_classifiers = {key: value for key, value in _name_estimators(classifiers)}
        self.vote = vote
        self.weights = weights
        
    def fit(self, X, y):
        self.labelenc = LabelEncoder()
        self.labelenc.fit(y)
        self.classes = self.labelenc.classes_
        self._classifiers = []
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X, self.labelenc.transform(y))
            self._classifiers.append(fitted_clf)
        return self
        

    def predict(self, X):
        #class label の予測
        if self.vote == "probability":
            maj_vote = np.maxarg(self.predict_proba(X), axis=1)
        else:
            predictions = np.asarray([clf.predict(X) for clf in self._classifiers]).T
            maj_vote = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)),
                                          axis=1, arr=predictions) 
            maj_vote = self.labelenc.inverse_transform(maj_vote)
            return maj_vote
    #asarray >> nandarray のコピーを作るかどうか
    def predict_proba(self, X):
        probas = np.asarray([clf.predict_proba(X) for clf in self._classifiers])
        avg_proba = np.average(probas, axis=0, weights=self.weights)
        return avg_proba
    
    def get_params(self, deep=True):
        if not deep:
            return super(MajorityVoteClassifier, self).get_params(deep=False)
        else:
            out = self.named_classifiers.copy()
            for name, step in six.iteritems(self.named_classifiers):
                for key, value in six.iteritems(step.get_params(deep=True)):
                    out["{}__{}".format(name, value)] = value
            return  out
                


# In[ ]:


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
iris = datasets.load_iris()
X, y = iris.data[50:, [1, 2]], iris.target[50:]
le = LabelEncoder()
y = le.fit_transform(y)
print(X[:10])


# In[ ]:


X_train, X_test, y_train, y_test =         train_test_split(X, y, test_size=0.5, random_state=1, stratify=y)
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import numpy as np
clf1 = LogisticRegression(penalty="l2", C=0.001, random_state=1)
clf2 = DecisionTreeClassifier(max_depth=1, criterion="entropy", random_state=1)
clf3 = KNeighborsClassifier(n_neighbors=1, p=2, metric="minkowski")
pipe1 = Pipeline([["sc", StandardScaler()], ["clf", clf1]])
pipe3 = Pipeline([["sc", StandardScaler()], ["clf", clf3]])
clf_labels = ["Logistic Regression", "Decision tree", "KNN"]
print("10 fold classfier: \n")
for clf, label in zip([pipe1, clf2, pipe3], clf_labels):
    scores = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10, scoring="roc_auc")
    print("roc_auc:{} +- {} {}".format(scores.mean(), scores.std(), label))
    


# In[ ]:


mv_clf = MajorityVoteClassifier(classifiers=[pipe1, clf2, pipe3])
clf_labels += ["majority voting"]
all_clf = [pipe1, clf2, pipe3, mv_clf]
for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10, scoring="roc_auc")
    print("roc auc: {} +- {} {}".format(scores.mean(), scores.std(), label))


# In[ ]:


#ensamble classifier のチューニングと評価
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
#auc >> roc曲線下の面積
colors = ["black", "orange", "blue", "green"]
linestyles = [":", "--", "-.", "-"]
for clf, label, clr, ls in zip(all_clf, clf_labels, colors, linestyles):
    y_pred = clf.fit(X_train, y_train).predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_pred)
    roc_auc = auc(x=fpr, y=tpr)
    plt.plot(fpr, tpr, color=clr, linestyle=ls, label="{} (auc = {})".format(label, roc_auc))

plt.legend(loc="lower right")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=2)
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.grid(alpha=0.5)
plt.show()


# In[ ]:


sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
from itertools import product
#2つ以上ののリストをうまくタプルにして、組み合わせて表示する！
X_min = X_train_std[:, 0].min() - 1
X_max = X_train_std[:, 0].max() + 1
y_min = X_train_std[:, 1].min() - 1
y_max = X_train_std[:, 1].max() + 1
#描画するサイズの決定
xx, yy = np.meshgrid(np.arange(X_min, X_max, 0.1), np.arange(y_min, y_max, 0.1))
f, axarr = plt.subplots(nrows=2, ncols=2, sharex="col", sharey="row", figsize=(7,5))
print(f, axarr)
for idx, clf, tt in zip(product([0, 1], [0, 1]), all_clf, clf_labels):
    clf.fit(X_train_std, y_train)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.3)
    axarr[idx[0], idx[1]].scatter(X_train_std[y_train==0, 0],
                                 X_train_std[y_train==0, 1],
                                 c="blue", marker="^", s=50)
    axarr[idx[0], idx[1]].scatter(X_train_std[y_train==1, 0],
                                 X_train_std[y_train==1, 1],
                                 c="green", marker="o", s=50)
    axarr[idx[0], idx[1]].set_title(tt)
plt.text(-3.5, -5, s="sepal width", ha="center", va="center", fontsize=12)
plt.text(-12.5, 4.5, s="sepal length", ha="center", va="center", fontsize=12, rotation=90)
plt.show() 
    


# In[ ]:


mv_clf.get_params()


# In[ ]:


from sklearn.model_selection import GridSearchCV
params = {"decisiontreeclassifier__max_depth":[1, 2],
         "pipeline-1__clf__C":[0.01, 1.0, 100.0]}
grid = GridSearchCV(estimator=mv_clf, param_grid=params, cv=10, scoring="roc_auc")
grid.fit(X_train, y_train)


# In[ ]:


for r, _ in enumerate(grid.cv_results_["mean_test_score"]):
    print("{} +- {} {}".format(grid.cv_results_["mean_test_score"][r],
                              grid.cv_results_["std_test_score"][r] / 2.0,
                              grid.cv_results_["params"][r]))


# In[ ]:


print("best params:{}".format(grid.best_params_))
print("accuracy:{}".format(grid.best_score_))


# In[ ]:


#bagging
#trainingsample >> 複数に分ける。それぞれで決定木を成長させ、多数決をとる。
import pandas as pd
df_wine = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header=None)
df_wine.columns = ["Class label", "Alcohole", "Malic acid", "Ash", "Alcalinity of ash",
                   "Magnesium", "Total phenols", "Flavanoids", "Nonflavanoid phenols",
                  "Proanthocyanins", "Color intensity", "Hue", "OD280/OD315 of diluted wines",
                  "Proline"]
df_wine = df_wine[df_wine["Class label"] != 1]
y = df_wine["Class label"].values
X = df_wine[["Alcohole", "OD280/OD315 of diluted wines"]].values
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_teat, y_train, y_teat = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)


# In[ ]:


from sklearn.ensemble import BaggingClassifier
tree = DecisionTreeClassifier(criterion="entropy", max_depth=None, random_state=1)
bag = BaggingClassifier(base_estimator=tree, n_estimators=500, max_samples=1.0,
                       max_features=1.0, bootstrap=True, bootstrap_features=False,
                       n_jobs=1, random_state=1)


# In[ ]:


from sklearn.metrics import accuracy_score
tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)
tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print("decision tree train/test accuracies {} / {}".format(tree_train, tree_test))


# In[ ]:


bag = bag.fit(X_train, y_train)
y_trian_pred = bag.predict(X_train)
y_test_pred = bag.predict(X_test)
bag_train = accuracy_score(y_train, y_train_pred)
bag_test = accuracy_score(y_true=y_test, y_pred=y_test_pred)
print(bag_test)
print("bagging train/test accuracies {} / {}".format(bag_train, bag_test))


# In[ ]:


X_min = X_train[:, 0].min() - 1
X_max = X_train[:, 0].max() + 1
y_min = X_train[:, 1].min() - 1
y_max = X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(X_min, X_max, 0.1),
                    np.arange(y_min, y_max, 0.1))
f, axarr = plt.subplots(nrows=1, ncols=2, sharex="row", sharey="col", figsize=(8, 3))
for ind, clf, tt  in zip([0, 1], [tree, bag], ["decision tree", "bagging"]):
    clf.fit(X_train, y_train)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    axarr[ind].contourf(xx, yy, Z, alpha=0.3)
    axarr[ind].scatter(X_train[y_train==0, 0],
                      X_train[y_train==0, 1], c="blue", marker="^")
    axarr[ind].scatter(X_train[y_train==1, 0],
                      X_train[y_train==1, 1], c="green", marker="o")
    axarr[ind].set_title(tt)
    
axarr[0].set_ylabel("Alcohole", fontsize=12)
plt.text(10.2, -0.5, s="OD280/OD315 of diluted wines", ha="center", va="center", fontsize=12)
plt.tight_layout()
plt.show()
                     


# In[ ]:


#ensemble>>>boosting>>>adaboost
from sklearn.ensemble import AdaBoostClassifier
tree = DecisionTreeClassifier(criterion="entropy", max_depth=1, random_state=1)
ada = AdaBoostClassifier(base_estimator=tree, n_estimators=500, learning_rate=0.1,
                        random_state=1)
tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)
tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print("decision tree train/test accuracies {} / {}".format(tree_train, tree_test))


# In[ ]:


ada = ada.fit(X_train, y_train)
y_train_pred = ada.predict(X_train)
y_test_pred = ada.predict(X_test)
ada_train = accuracy_score(y_train, y_train_pred)
ada_test = accuracy_score(y_test, y_test_pred)
print("adaboost train/test accuracies {} / {}".format(ada_train, ada_test))


# In[ ]:


X_min = X_train[:, 0].min() - 1
X_max = X_train[:, 0].max() + 1
y_min = X_train[:, 1].min() - 1
y_max = X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(X_min, X_max, 0.1),
                    np.arange(y_min, y_max, 0.1))
f, axarr = plt.subplots(nrows=1, ncols=2, sharex="row", sharey="col", figsize=(8, 3))
for ind, clf, tt  in zip([0, 1], [tree, ada], ["decision tree", "Ada Boost"]):
    clf.fit(X_train, y_train)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    axarr[ind].contourf(xx, yy, Z, alpha=0.3)
    axarr[ind].scatter(X_train[y_train==0, 0],
                      X_train[y_train==0, 1], c="blue", marker="^")
    axarr[ind].scatter(X_train[y_train==1, 0],
                      X_train[y_train==1, 1], c="green", marker="o")
    axarr[ind].set_title(tt)
    
axarr[0].set_ylabel("Alcohole", fontsize=12)
plt.text(10.2, -0.5, s="OD280/OD315 of diluted wines", ha="center", va="center", fontsize=12)
plt.tight_layout()
plt.show()

