
# coding: utf-8

# In[ ]:


import pandas as pd
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",header=None)
from sklearn.preprocessing import LabelEncoder
X = df.loc[:,2:].values
y = df.loc[:,1].values
le = LabelEncoder()
y = le.fit_transform(y)
print(le.classes_)


# In[ ]:


le.transform(le.classes_)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                   stratify=y, random_state=1)


# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
pipe_lr = make_pipeline(StandardScaler(),
                       PCA(n_components=2),
                       LogisticRegression(random_state=1))
pipe_lr.fit(X_train, y_train)
y_pred = pipe_lr.predict(X_test)
print("test accuracy:{}".format(pipe_lr.score(X_test, y_test)))
#pipeline >> 便利なwrapperツール 


# In[ ]:


#holdout strategy >> training data, check data, test data の　3つに分け
#training data & check data で　hyper paramertor を調節する。
#層化　>> stratified >> サンプルの比率を一定に保つ
import numpy as np
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=10, random_state=1).split(X_train, y_train)
#print([i for i in kfold])
scores = []
for k, (train, test) in enumerate(kfold):
    pipe_lr.fit(X_train[train], y_train[train])
    score = pipe_lr.score(X_train[test], y_train[test])
    scores.append(score)
    print("Fold:{}, Class dict:{}, Accuracy:{}".format(k+1, np.bincount(y_train[train]), score))
print("\nCV accuracy:{} +- {}".format(np.mean(scores), np.std(scores)))


# In[ ]:


from sklearn.model_selection import cross_val_score
#cv >> data の分割数, n_jobs >> 使用するCPUの数
scores = cross_val_score(estimator=pipe_lr, X=X_train, y=y_train, cv=10, n_jobs=10)
print("CV accuracy scores:{}".format(scores))
print("CV accuracy:{} +- {}".format(np.mean(scores), np.std(scores)))


# In[ ]:


#learning curve & validation curve
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
pipe_lr = make_pipeline(StandardScaler(),LogisticRegression(penalty="l2", random_state=1))
train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_lr,
                                                       X=X_train,
                                                       y=y_train,
                                                       train_sizes=np.linspace(0.1, 1.0, 10),
                                                        cv=10, n_jobs=10)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.plot(train_sizes, train_mean, color="blue",
         markersize=5, marker="o",label="train accuracy")
plt.fill_between(train_sizes, 
                train_std+train_mean,
                train_mean-train_std,
                alpha=0.5, color="blue")
plt.plot(train_sizes, test_mean, color="green",linestyle="--",
         markersize=5, marker="s",label="validation accuracy")
plt.fill_between(train_sizes, 
                test_std+test_mean,
                test_mean-test_std,
                alpha=0.5, color="green")
plt.grid()
plt.xlabel("Number of train sample")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")
plt.ylim([0.8,1.05])
plt.tight_layout()
plt.show()


# In[ ]:


from sklearn.model_selection import validation_curve
param_range = [10**c for c in range(-3,3)]
train_scores, test_scores = validation_curve(estimator=pipe_lr,
                                            X=X_train, y=y_train,
                                            param_name="logisticregression__C",
                                            param_range=param_range,cv=10)
#logisticregression の後の_ は2つ分用意する！
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.plot(param_range, train_mean, color="blue", marker="^", markersize=5, label="training accuracy")
plt.fill_between(param_range, train_mean+train_std, train_mean-train_std, alpha=0.5, color="blue")
plt.plot(param_range, test_mean, color="green", marker="o", markersize=5, label="validation accuracy")
plt.fill_between(param_range, test_mean+test_std, test_mean-test_std, alpha=0.5, color="green")
plt.grid()
plt.xscale("log")
plt.legend("upper left")
plt.xlabel("Parametor C")
plt.ylabel("Accuracy")
plt.ylim([0.8, 1.05])
plt.tight_layout()
plt.show()


# In[ ]:


#hyper paramator を　grid search　という、手法で最適化する。
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=1))
param_range = [10**c for c in range(-4,5)]
param_grid = [{"svc__C": param_range, "svc__kernel": ["linear"]},
               {"svc__C": param_range, "svc__gamma":param_range, "svc__kernel":["rbf"]}]
gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring="accuracy", cv=10, n_jobs=-1)
gs = gs.fit(X_train, y_train)
print(gs.best_score_)
print(gs.best_params_)


# In[ ]:


clf = gs.best_estimator_
clf.fit(X_train, y_train)
print("test accuracy: {}".format(clf.score(X_test, y_test)))


# In[ ]:


#入れ子式の交差検証　>> out roop = {test : trainning}, inner roop={training: check}
#out roop ト　inner roop ヲ　テキギイレカエル
gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid,
                 scoring="accuracy", cv=2)
scores = cross_val_score(gs, X_train, y_train, scoring="accuracy", cv=5)
print("CV accuracy:{} +- {}".format(np.mean(scores), np.std(scores)))


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0),
                 param_grid={"max_depth":[1,2,3,4,5,6,7,None]},
                 scoring="accuracy", cv=5)
scores = cross_val_score(gs, X_train, y_train, scoring="accuracy", cv=5)
print("CV accuracy:{} +- {}".format(np.mean(scores), np.std(scores)))


# In[ ]:


#confusion matrix >> true&false:実際に当たったか　positive&negative:どう予測したか
from sklearn.metrics import confusion_matrix
pipe_svc.fit(X_train, y_train)
y_pred = pipe_svc.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)
#TP FN
#FP TN


# In[ ]:


fig, ax = plt.subplots(figsize=(2.5,2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i,j], va="center", ha="center")
        
plt.xlabel("predicted label")
plt.ylabel("true label")
plt.tight_layout()
plt.show()


# In[ ]:


#適合率　ト　再現率
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score, recall_score
print("precision:{}".format(precision_score(y_true=y_test, y_pred=y_pred)))
print("recll:{}".format(recall_score(y_true=y_test, y_pred=y_pred)))
print("F1:{}".format(f1_score(y_true=y_test, y_pred=y_pred)))
# pre = tp / (tp + fp)
# tec = tp / (fn + tp)
# f1 = 2 * (pre*rec) / (pre+rec)


# In[ ]:


from sklearn.metrics import f1_score, make_scorer
scorer = make_scorer(f1_score, pos_label=0)
gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, 
               scoring=scorer, cv=10, n_jobs=-1)
gs.fit(X_train, y_train)
print(gs.best_score_)
print(gs.best_params_)


# In[ ]:


#Area Under the Curve >> roc 曲線
from sklearn.metrics import roc_curve, auc
from scipy import interp
pipe_lr = make_pipeline(StandardScaler(),PCA(n_components=2),
                       LogisticRegression(penalty="l2", random_state=1, C=100.0))
X_train2 = X_train[:,[4,14]]
cv = list(StratifiedKFold(n_splits=3, random_state=1).split(X_train, y_train))
fig = plt.figure(figsize=(7, 5))
mean_tpr = 0.0
mean_fpr = np.linspace(0,1,100)
all_tpr = []
for i, (train, test) in enumerate(cv):
    probas = pipe_lr.fit(X_train2[train], y_train[train]).predict_proba(X_train2[test])
    fpr, tpr, thresholds = roc_curve(y_train[test], probas[:, 1], pos_label=1)
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label = "roc fold {} (areas = {})".format(i+1, roc_auc))
plt.plot([0, 1], [0, 1], linestyle="--",
        color=(0.6, 0.6, 0.6), label="random guessing")
mean_tpr /= len(cv)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, "k--", label="mean ROC (area = {})".format(mean_auc), lw=2)
plt.plot([0, 0, 1], [0, 1, 1], linestyle=":", color="black", label="perfect performance")
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel("false positive")
plt.ylabel("true positive")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()


# In[ ]:


pre_score = make_scorer(score_func=precision_score, pos_label=1, greater_is_better=True, average="micro")
print(pre_score)
#average argment により、他クラス分類に対応している。


# In[ ]:


#bad samples
X_imb = np.vstack((X[y==0], X[y==1][:40]))
y_imb = np.hstack((y[y==0], y[y==1][:40]))
print(X_imb,"\n", y_imb)


# In[ ]:


y_pred = np.zeros(y_imb.shape[0])
np.mean(y_pred == y_imb) * 100
#true > 1, false > 0 として合計を算出し、要素の数だけ割っている


# In[ ]:


from sklearn.utils import resample
print("number of class 1 sample before: ", X_imb[y_imb == 1].shape[0])
X_upsampled, y_upsampled = resample(X_imb[y_imb == 1], y_imb[y_imb == 1],
                                   replace=True, n_samples=X_imb[y_imb == 0].shape[0],
                                   random_state=123)
print("number of class 1 samples after: ", X_upsampled.shape[0])


# In[ ]:


X_bal = np.vstack((X[y==0], X_upsampled))
y_bal = np.hstack((y[y==0], y_upsampled))
#np.vstack >> axis1, np.hstack >>axis0 で結合


# In[ ]:


y_pred = np.zeros(y_bal.shape[0])
np.mean(y_pred == y_bal) * 100

