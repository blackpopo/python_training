
# coding: utf-8

# In[ ]:


import pandas as pd
from io import StringIO
csv_data = '''A,B,C,D
              1.0,2.0,3.0,4.0
              5.0,6.0,,8.0
              10.0,11.0,12.0,'''

df = pd.read_csv(StringIO(csv_data))
df


# In[ ]:


df.values


# In[ ]:


df.isnull().sum()


# In[ ]:


df.dropna()


# In[ ]:


df.dropna(axis = 1)


# In[ ]:


df.dropna(how = "all")


# In[ ]:


df.dropna(thresh = 4)


# In[ ]:


df.dropna(subset = ["C"])


# In[ ]:


from sklearn.preprocessing import Imputer
imr = Imputer(missing_values = "NaN",strategy = "mean",axis = 0)
imr = imr.fit(df.values)
imputed_data = imr.transform(df.values)
imputed_data


# In[ ]:


import pandas as pd
df = pd.DataFrame([
    ["green","M",10.01,"class1"],
    ["red","L",13.5,"class2"],
    ["blue","XL",15.4,"class1"]
])
df.columns = ["color","size","price","classlabel"]
#colulms >> df の　property として設定
df


# In[ ]:


df.loc[1]


# In[ ]:


size_map = {"XL":3,"L":2,"M":1}
df["size"] = df["size"].map(size_map)
df


# In[ ]:


inv_size_map = {v:k for k ,v in size_map.items()}
df["size"].map(inv_size_map)


# In[ ]:


import numpy as np
class_map = {label:idx for idx,label in enumerate(np.unique(df["classlabel"]))}
class_map


# In[ ]:


df["classlabel"] = df["classlabel"].map(class_map)
df


# In[ ]:


inv_class_map = {v:k for k,v in class_map.items()}
df["classlabel"] = df["classlabel"].map(inv_class_map)
df


# In[ ]:


from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder() #LabelEncoder クラスのインスタンスを作る
y = class_le.fit_transform(df["classlabel"].values)
y


# In[ ]:


class_le.inverse_transform(y)


# In[ ]:


X = df[["color","size","price"]].values
color_le = LabelEncoder()
X[:,0] = color_le.fit_transform(X[:,0])
X


# In[ ]:


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features = [0])
#OneHotEncoder >> 名義特徴量を0,1 に分ける
ohe.fit_transform(X).toarray()


# In[ ]:


pd.get_dummies(df[["price","color","size"]])


# In[ ]:


pd.get_dummies(df[["price","color","size"]])
ohe.fit_transform(X).toarray()[:,1:]


# In[ ]:


df_wine = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",header = None)
df_wine.columns = ["Class label","Alcohol","Malic acid","Ash","Alcalinity of ash","Magnesium","Totoal phenols","Flabanoids","Nonflavanoid phenols","Proanthocyanins","Color intensity","Hue","OD280/OD315 of diluted wine","Proline"]
print("Classlabels",np.unique(df_wine["Class label"]))
df_wine.head()


# In[ ]:


from sklearn.model_selection import train_test_split
X,y = df_wine.iloc[:,1:].values,df_wine.iloc[:,0].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state=0 ,stratify = y)
#stratify >> stratify 中のラベルが　test_data_set,train_data_set それぞれの比率として同じになるようにするもの
#ilocについて、ilocはPandasのメソッドの一つ。引数によって指定された行、または列を取り出す。\
#.values はpandas のプロパティの一つ


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.fit_transform(X_test)
print(X_train_norm[0:5,:])


# In[ ]:


#標準化　と　正規化
ex = np.array([0,1,2,3,4,5])
print("stndardizet",(ex - ex.mean()) / ex.std())
print("normalized",(ex - ex.min()) / (ex.max() - ex.min()))


# In[ ]:


from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)
print(X_train_std[:5,:])
print(X_test_std[:5,:])


# In[ ]:


from sklearn.linear_model import LogisticRegression
LogisticRegression(penalty="l1")


# In[ ]:


lr = LogisticRegression(penalty="l1", C=1.0)
lr.fit(X_train_std, y_train)
print("training accuracy:", lr.score(X_train_std, y_train))
print("test accuracy: ", lr.score(X_test_std, y_test))


# In[ ]:


lr.intercept_


# In[ ]:


lr.coef_


# In[ ]:


import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.subplot(1,1,1)
colors = ["blue","green","red","cyan","magenta","yellow","black","pink",
          "lightgreen","lightblue","gray","indigo","orange"]
weights, params = [], []
for c in np.arange(-4.0, 6.0):
    lr = LogisticRegression(penalty="l1", C=10.0 ** c, random_state=0)
    lr.fit(X_train_std,y_train)
    weights.append(lr.coef_[1])
    params.append(10 ** c)
weights = np.array(weights)
for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, weights[:, column], label=df_wine.columns[column+1], color=color)
plt.axhline(0, color="black", linestyle="--", linewidth=3)
plt.xlim(10 ** (-5), 10 ** 5)
plt.ylabel("weights coefficients")
plt.xlabel("C")
plt.xscale("log")
plt.legend(loc="upper left")
ax.legend(loc="upper left", bbox_to_anchor=(1.38, 1.03), ncol=1, fancybox=True)
plt.show()


# In[ ]:


from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

class SBS():
    #future selection & feature extraction
    
    def __init__(self, estimator, k_features, scoring=accuracy_score,
                 test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size= test_size
        self.random_state = random_state
    
    def fit(self, X, y):
        X_train, X_test, y_train, y_test =            train_test_split(X, y, test_size=self.test_size, 
                             random_state=self.random_state)
        dim = X_train.shape[1]
        self._indices = tuple(range(dim))
        self._subsets = [self._indices]            
        score = self.calc_score(X_train, y_train, X_test, y_test, self._indices)
        self._scores = [score]
        
        while dim > self.k_features:
            scores = []
            subsets = []
            for p in combinations(self._indices, r=dim-1):
                score = self.calc_score(X_train, y_train, X_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores)
            self._indices = subsets[best]
            self._subsets.append(self._indices)
            dim -= 1
            self._scores.append(scores[best])
        
        self.k_score = self._scores[-1]
        return self
    
    def transform(self, X):
        return X[:, self._indices]
    
    def calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score
        
        


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
knn = KNeighborsClassifier(n_neighbors=5)
sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, y_train)


# In[ ]:


k_feat = [len(k) for k in sbs._subsets]
plt.plot(k_feat, sbs._scores, marker="o")
plt.ylim([0.7, 1.02])
plt.ylabel("accuracy")
plt.xlabel("number of features")
plt.grid()
plt.tight_layout()
plt.show()
#random_state = 0 (X_train　の設定)にすると　まったく違うグラフになる。


# In[ ]:


k3 = list(sbs._subsets[10])
print(df_wine.columns[1:][k3])


# In[ ]:


knn.fit(X_train_std, y_train)
print("training accuracy: ",knn.score(X_train_std, y_train))
print("test aiccuracy:", knn.score(X_test_std, y_test))


# In[ ]:


knn.fit(X_train_std[:, k3], y_train)
print("training accuracy", knn.score(X_train_std[:,k3], y_train))
print("test accuracy", knn.score(X_test_std[:,k3] , y_test))


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
feat_labels = df_wine.columns[1:]
forest = RandomForestClassifier(n_estimators=500, random_state=1)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("{:0g}),{:<30},{:f}".format(f+1, feat_labels[indices[f]], importances[indices[f]]))


# In[ ]:


plt.title("Feature Importance")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), feat_labels[indices], rotation=90)
plt.xlim([-1,X_train.shape[1]])
plt.tight_layout()
plt.show()


# In[ ]:


from sklearn.feature_selection import SelectFromModel
#特徴量洗濯のモデル
sfm = SelectFromModel(forest, threshold=0.1,prefit=True)
X_selected = sfm.transform(X_train)
print("number of smaples", X_selected.shape[0])
for f in range(X_selected.shape[1]):
    print("{:f2},{:<30},{f}".format(f+1, feat_labels[indices[f]], importances[indices[f]]))

