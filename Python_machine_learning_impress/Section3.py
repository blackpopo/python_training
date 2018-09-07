
# coding: utf-8

# In[ ]:


from sklearn import datasets
import numpy as np
iris = datasets.load_iris()
X = iris.data[:,[2,3]]
y = iris.target
print("Class labels:",np.unique(y))
print(iris)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 1,stratify = y)
#stratify >> 層かプリンティング（クラスラベルの比率を一定にする）　…賢い
print("label counts in y:",np.bincount(y))
print("label counts in y_train:",np.bincount(y_train))
print("label counts in y_test:",np.bincount(y_test))


# In[ ]:


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(X_train)
X_train_std = ss.transform(X_train)
X_test_std = ss.transform(X_test)
# print(X_train_std,X_test_std)
#fit と　transform　は全く別のメソッド

from sklearn.linear_model import Perceptron
ppn = Perceptron(n_iter = 40,eta0 = 0.1,random_state = 1)
ppn.fit(X_train_std,y_train)


# In[ ]:


y_pred = ppn.predict(X_test_std)
print("miss_classified:%d" % (y_test != y_pred).sum())
# %で埋め込み数字　sprintf　というらしい


# In[ ]:


from sklearn.metrics import accuracy_score
print("Accuracy: %.2f" % accuracy_score(y_test,y_pred))
print("Accuracy: %.2f" % ppn.score(X_test_std,y_test))
get_ipython().run_line_magic('whos', '')


# In[ ]:


from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
def plot_decision_regions(X,y,classifier,test_idx = None,resolution = 0.02):
    markers = ("s","x","o","^","v")
    colors = ("red","blue","lightgreen","gray","cyan")
    cmap =  ListedColormap(colors[:len(np.unique(y))])
    x1_min,x1_max = X[:,0].min() - 1 ,X[:,0].max() + 1
    x2_min,x2_max = X[:,1].min() - 1 ,X[:,1].max() + 1
    xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),np.arange(x2_min,x2_max,resolution))
    Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1,xx2,Z,alpha = 0.3 ,cmap = cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())
    for idx ,cl in enumerate(np.unique(y)):
        plt.scatter(x = X[y == cl,0] , y = X[y == cl,1] ,alpha = 0.8 , c = colors[idx] ,marker = markers[idx] , label = cl, edgecolor = "black" )
    if test_idx:
        X_test,y_test = X[test_idx,:],y[test_idx]
        plt.scatter(X_test[:,0],X_test[:,1],c = "",edgecolor = "black",alpha = 1,marker = "o",s = 100, label = "test_set")
                    
    


# In[ ]:


X_combined_std = np.vstack((X_train_std,X_test_std))
y_combined = np.hstack((y_train,y_test))
plot_decision_regions(X = X_combined_std,y = y_combined,classifier = ppn,test_idx = range(105,150))
plt.xlabel("petal length")
plt.ylabel("petal width")
plt.legend("upper left")
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import math
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

z = np.arange(-7,7,0.1)
phi_z = sigmoid(z)
plt.plot(z,phi_z)
plt.axvline(0.0,color = "k") #axvline >> 垂直線
plt.ylim(-0.1,1.1)
plt.xlabel("z")
plt.ylabel("$\phi$") # phiの記号の記述
plt.yticks([0.0,0.5,1.0])
ax = plt.gca() #gca >> graphic of axis
ax.yaxis.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:


#class label が　0 or 1　の場合
def cost_1(z):
    return -np.log(sigmoid(z))
def cost_0(z):
    return -np.log(1-sigmoid(z))

z = np.arange(-10,10,0.1)
phi_z = sigmoid(z)
c1 = [cost_1(x) for x in z]
plt.plot(phi_z,c1,label = "J(w) y = 1")
c0 = [cost_0(x) for x in z]
plt.plot(phi_z,c0,label = "J(w) y = -1")
plt.ylim(0.0,5.1)
plt.xlim([0,1])
plt.xlabel("$\phi$(z)")
plt.ylabel("J(w)")
plt.legend(loc = "upper left")
plt.tight_layout()
plt.show()


# In[ ]:


class LogisticRegressionGD(object):
    def __init__(self,eta = 0.05,n_iter = 100,random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        
    def fit(self,X,y):
        rgen = np.random.RandomState(self.random_state)
        self.w = rgen.normal(loc = 0.0,scale = 0.01,size =1+ X.shape[1])
        self.gosagun = []
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w[1:] += self.eta * X.T.dot(errors)
            self.w[0] += self.eta * errors.sum() #.sum() >> 破壊的メソッド
            cost = -y.dot(np.log(output)) - (1-y).dot(np.log(1-output))
            self.gosagun.append(cost)
        return self
            
    def net_input(self,X):
        return np.dot(X,self.w[1:]) + self.w[0]
    
    def activation(self,z):
        return 1.0 / (1.0 + np.exp(-np.clip(z,-250,250))) #z >250 の時　z -250　に変換する
    
    def predict(self,X):
        return np.where(self.activation(self.net_input(X)) >= 0.5,1,-1)


# In[ ]:


X_train_01_subset = X_train[ (y_train == 0) | (y_train == 1) ]
y_train_01_subset = y_train[ (y_train == 0) | (y_train == 1) ]
lrgd = LogisticRegressionGD(eta = 0.05,n_iter = 1000,random_state = 1)
lrgd.fit(X_train_01_subset,y_train_01_subset)
plot_decision_regions(X_train_01_subset,y_train_01_subset,classifier = lrgd)
plt.xlabel("petal length")
plt.ylabel("petal width")
plt.legend(loc = "upper left")
plt.tight_layout()
plt.show()


# In[ ]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C = 100.0,random_state = 1)
lr.fit(X_train_std,y_train)
plot_decision_regions(X_combined_std,y_combined,classifier = lr,test_idx = range(105,150))
plt.xlabel("petal length")
plt.ylabel("petal width")
plt.legend(loc  = "upper left")
plt.tight_layout()
plt.show()


# In[ ]:


lr.predict_proba(X_test_std) #probability of 1st,2nd,3rd


# In[ ]:


lr.predict_proba(X_test_std[:10,:]).argmax(axis = 1)


# In[ ]:


lr.predict(X_test_std[:10,:])


# In[ ]:


weights  ,params = [],[]
for c in np.arange(-5,5):
    lr = LogisticRegression(C = 10.0 ** c,random_state = 1)
    lr.fit(X_train_std,y_train)
    weights.append(lr.coef_[1])
    params.append(10.0 ** c)
    
weights = np.array(weights)
plt.plot(params,weights[:,0],label = "petal length")
plt.plot(params,weights[:,1],label = "petal width")
plt.ylabel("weight coefficient")
plt.xlabel("c")
plt.xscale("log")
plt.show()


# In[ ]:


from sklearn.svm import SVC
svm = SVC(kernel = "linear" , C=1.0 ,random_state = 1)
svm.fit(X_train_std,y_train)
plot_decision_regions(X_combined_std,y_combined,classifier = svm,test_idx = range(105,150))
plt.xlabel("petal length")
plt.ylabel("petal width")
plt.legend(loc = "upper left")
plt.tight_layout()
plt.show()


# In[ ]:


from sklearn.linear_model import SGDClassifier
ppn = SGDClassifier(loss = "perceptron")
lr = SGDClassifier(loss = "log")
svm = SGDClassifier(loss = "hinge")


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
np.random.seed(1)
X_xor = np.random.randn(200,2)#random.randn(nand.arrayの形)
y_xor = np.logical_xor(X_xor[:,0] > 0 ,X_xor[:,1] > 0)#論理和でTrue　or False を割り当てる
y_xor = np.where(y_xor,1,-1) #True >> 1 ,False >> 1の割り当て
plt.scatter(X_xor[y_xor == 1,0],X_xor[y_xor ==1,1] ,c = "b", marker = "x",label = "1")
plt.scatter(X_xor[y_xor == -1,0],X_xor[y_xor == -1,1],c = "r",marker = "s",label = "-1")
plt.xlim([-3,3])
plt.ylim([-3,3])
plt.legend(loc = "best")
plt.tight_layout()
plt.show()


# In[ ]:


svm =  SVC(kernel = "rbf",random_state = 1 ,gamma = 0.10,C = 10.0)
svm.fit(X_xor,y_xor)
plot_decision_regions(X_xor,y_xor,classifier = svm)
plt.tight_layout()
plt.show()


# In[ ]:


svm = SVC(kernel = "rbf",random_state = 1,gamma = 0.2 ,C = 1.0)
svm.fit(X_train_std,y_train)
plot_decision_regions(X_combined_std,y_combined,classifier = svm,test_idx = range(105,150))
plt.xlabel("petal lenght")
plt.ylabel("petal width")
plt.tight_layout()
plt.show()


# In[ ]:


svm = SVC(kernel = "rbf" ,random_state = 1,gamma = 100 ,C = 1.0)#rbf >> Radial Basis Function 動型基底カーネル（ガウスカーネル）
svm.fit(X_train_std,y_train)
plot_decision_regions(X_combined_std,y_combined,classifier = svm,test_idx = range(105,149))
plt.xlabel("petal length")
plt.ylabel("petal width")
plt.legend(loc = "upper left")
plt.show()


# In[ ]:


#decision Tree 
#informatin gain 1.Gini impurity 2.entropy 3.classification error
import matplotlib.pyplot as plt
import numpy as np
def gini(p):
    return (p)*(1 - (p)) + (1 - p) * (1 - (1 - p))

def entropy(p):#エントロピー
    return - p * np.log2(p) - (1 - p) * np.log2((1 - p))

def error(p):#分類誤差
    return 1 - np.max([p,1 - p])

x = np.arange(0.0 ,1.0 ,0.01)
ent = [entropy(p) if p != 0 else None for p in x]
sc_ent = [e * 0.5 if e else None for e in ent]
err = [error(i) for i in x]
fig = plt.figure()
ax = plt.subplot(111)
functions = [ent,sc_ent,gini(x),err]
names = ["entropy","scaled_entropy","Gini Impurity","misclassification error"]
linestyles = ["-","-","--","-."]
colors = ["black","red","green","cyan"]
for i , lab ,ls, c in zip(functions,names,linestyles,colors):
    line = ax.plot(x,i,label = lab ,linestyle = ls, color = c, lw =2)
ax.legend(loc = "upper center",bbox_to_anchor = (0.5,1.15),ncol = 5 ,fancybox = True, shadow = False)
ax.axhline(y = 0.5 ,linewidth = 1 ,color = "k", linestyle = "--")
ax.axhline(y = 1.0 ,linewidth = 1, color = "k",linestyle = "--")
plt.xlabel("p")
plt.ylabel("Inpurity index")
plt.show()


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion = "gini",max_depth  = 4 ,random_state = 1)
tree.fit(X_train,y_train)
X_combined = np.vstack((X_train,X_test))
y_combined = np.hstack((y_train,y_test))
plot_decision_regions(X_combined,y_combined,classifier = tree,test_idx = range(105,150))
plt.xlabel("petal length")
plt.ylabel("petal width")
plt.tight_layout()
plt.show()


# In[ ]:


from pydotplus import graph_from_dot_data 
from sklearn.tree import export_graphviz

dot_data = export_graphviz(tree,filled = True,rounded = True ,class_names = ["Setosa","Versicolor","Virginica"],feature_names = ["petal length","petal width"],out_file = None)
graph = graph_from_dot_data(dot_data)
graph.write_png("tree.png")


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(criterion = "gini",n_estimators = 25,random_state = 1,n_jobs = 2)
#n_estimators >> 決定木の数
forest.fit(X_train,y_train)
plot_decision_regions(X_combined,y_combined,classifier = forest,test_idx = range(105,150)) 
plt.xlabel("petal length")
plt.ylabel("petal width")
plt.tight_layout()
plt.show()


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

#minkowski >> ユークリッド距離
knn.fit(X_train_std,y_train)
plot_decision_regions(X_combined_std,y_combined,classifier = knn,test_idx = range(105,150))
plt.xlabel("petal length")
plt.ylabel("petal width")
plt.legend("upper left")
plt.tight_layout()
plt.show()

