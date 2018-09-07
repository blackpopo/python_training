
# coding: utf-8

# In[ ]:


import numpy as np
class Perceptron(object):
    def __init__(self , eta = 0.01 , n_iter = 50, random_status = 1):
        self.eta = eta #eta >> 学習率
        self.n_iter =  n_iter # n__iter >> 学習回数
        self.random_status = random_status # random__status >> 乱数シード
        
    def fit(self ,X,Y):
        rgen = np.random.RandomState(self.random_status) 
        self.w = rgen.normal(loc = 0.0 , scale = 0.01 ,size = 1 + X.shape[1])
        self.errors = []
#         print(self.w)
        
        for _ in range(self.n_iter):
            error = 0
            for xi , target in zip(X,Y):
                update = self.eta * (target  - self.predict(xi))
#                print(target ,self.predict(xi))
                self.w[1:] += update * xi
                self.w[0] += update # x0 == 1
                error += int(update != 0.0)
            self.errors.append(error)
#         print(self.errors)
        return self

    
    def net_input(self,X):
#         print(np.dot(X,self.w[1:])+self.w[0])
        return np.dot(X,self.w[1:])+self.w[0]
    
    def predict(self,X):
        return np.where(self.net_input(X) >= 0.0 ,1 ,-1)


# In[ ]:


v1 = np.array([1,2,3])
v2 = 0.5 * v1
print(v2)
np.arccos(v1.dot(v2)/(np.linalg.norm(v1)+np.linalg.norm(v2)))


# In[ ]:


import pandas as pd
df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",header = None)
# df >> Data Frame
df


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
y = df.iloc[0:100,4].values
y = np.where(y == "Iris-setosa",-1,1)
X = df.iloc[0:100,[0,2]].values
plt.scatter(X[:50,0],X[:50,1],color = "blue",marker = "o",label = "setosa")
plt.scatter(X[50:100,0],X[50:100,1],color = "red",marker = "x", label= "versicolor")
plt.xlabel("sepal")
plt.ylabel("petal")
plt.legend(loc="upper left")
plt.show
# print(X)


# In[ ]:


ppn = Perceptron(eta = 0.01,n_iter = 10)
ppn.fit(X,y)
plt.plot(range(1,len(ppn.errors)+1),ppn.errors,marker= "o")
plt.xlabel("Epochs")
plt.ylabel("N of update")
plt.show()


# In[ ]:


from matplotlib.colors import ListedColormap

def plot_decision_regions(X,y,classifier,resolution = 0.02):
    markers = ("s","x","o","^","v")
    colors = ("red","blue","green","gray","cyan")
    cmap = ListedColormap(colors[:len(np.unique(y))]) #unique >> setが他のarrayに直す
    
    x1_min ,x1_max = X[:,0].min() -1, X[:,0].max() +1
    x2_min, x2_max = X[:,1].min() -1 ,X[:,1].max() +1
    
    xx1 , xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),np.arange(x2_min,x2_max,resolution))
    Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T) #ravel >> 　1次元型のarrayにする
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1,xx2,Z,alpha = 0.3,cmap= cmap) # contourf >> 等高線
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())
    
    for idx ,cl in enumerate(np.unique(y)):
        plt.scatter(x= X[y == cl,0],y=X[y == cl,1],alpha = 0.8, c = colors[idx],marker = markers[idx],label=cl,edgecolor = "black")
    
    


# In[ ]:


plot_decision_regions(X,y,classifier=ppn)
plt.xlabel("sepal")
plt.ylabel("petal")
plt.legend(loc = "upper left")
plt.show()


# In[ ]:


class AdalineGD(object):
    #eta >> 学習率
    #cost >> リスト w>>一次元配列
    #X >> shape =[n_samples , n_features]
    #y >> shape = n_samples
    def __init__(self,eta = 0.01,n_iter = 50,random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        
    def fit(self,X,y):
        rgen = np.random.RandomState(self.random_state)
        self.w = rgen.normal(loc = 0.0,scale = 0.01,size = 1 + X.shape[1])
        self.costs = []
        
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input) #activation >> 活性化関数
            errors = (y - output)
            self.w[1:] += self.eta * X.T.dot(errors) #vector multiplation
            self.w[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2
            self.costs.append(cost)
        return self
    
    def net_input(self,X):
        return np.dot(X,self.w[1:]) + self.w[0]
    
    def activation(self,X):
        return X
    
    def predict(self,X):
        return np.where(self.activation(self.net_input(X)) >= 0.0,1,-1)
    



# In[ ]:


fig ,ax = plt.subplots(nrows = 1,ncols = 2,figsize = (10,4))

ada1 = AdalineGD(n_iter = 10,eta = 0.01).fit(X,y)
ax[0].plot(range(1,len(ada1.costs) + 1),np.log10(ada1.costs),marker = "o")
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("log(Sum_of_errors)")
ax[0].set_title("Adaline eta = 0.01")

ada2 = AdalineGD(n_iter = 10,eta = 0.0001).fit(X,y)
ax[1].plot(range(1,len(ada2.costs) + 1),np.log10(ada2.costs),marker = "o")
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("log(sum_of_errors)")
ax[1].set_title("Adaline eta = 0.0001")

plt.show()


# In[ ]:


X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()
ada = AdalineGD(n_iter = 15,eta = 0.01)
print(X_std,y)
ada.fit(X_std,y)
plot_decision_regions(X_std,y,classifier = ada)
plt.title("Adaline Grad")
plt.xlabel("sepal")
plt.ylabel('petal')
plt.legend(loc = "upper left")
plt.tight_layout()
plt.show()

plt.plot(range(1,len(ada.costs) + 1),ada.costs,marker = "o")
plt.xlabel("Epochs")
plt.ylabel("sum-of-errors")
plt.show()


# In[ ]:


#確率的勾配降下法
from numpy.random import seed

class AdalineSGD(object):
    #shuffle >> True の時trainin data　をシャッフルする
    def __init__(self,eta = 0.01,n_iter = 10,shuffling = True,random_state = None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_init = False
        self.shuffling = shuffling
        self.random_state = random_state
        
    def fit(self,X,y):
        self.init_weights(X.shape[1])
        self.costs = []
        for i in range(self.n_iter):
            if self.shuffling:
                X,y = self.shuffle(X,y)
            cost = []
            for xi ,target in zip(X,y):
                cost.append(self.update_weights(xi,target))
            avg_cost = sum(cost) / len(y)
            self.costs.append(avg_cost)
#             print(self.costs)
        return self
    
    def partical_fit(self,X,y):
        if not self.w_init:  #wの初期化がFalseの時実行
            self.init_weights(X.shape[1])
        if y.ravel().shape[0] > 1: #yの要素が一つより大きい
            for xi,target in zip(X,y):
                self.update_weights(xi,target)
        else:
            self.update_weights(X,y) #X,y が一次元配列
        return self
    
    def shuffle(self,X,y):
        r = self.rgen.permutation(len(y)) # len(y)の配列のランダム準コピーを作る
        return X[r] , y[r] #ndarray を突っ込むとその順番になる
    
    def init_weights(self,m):
        self.rgen = np.random.RandomState(self.random_state)
        self.w = self.rgen.normal(loc = 0.0,scale = 0.01,size = 1 + m)
        self.w_init = True
        
    def update_weights(self,xi,target):
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w[1:] += self.eta * xi.dot(error)
        self.w[0] += self.eta * error
        cost = 0.5 * error**2
        return cost
    
    def net_input(self,X):
        return np.dot(X,self.w[1:]) + self.w[0]
    
    def activation(self,X):
        return X
    
    def predict(self,X):
        return np.where(self.activation(self.net_input(X)) >= 0.0 ,1 ,-1)
    
ada = AdalineSGD(n_iter = 15,eta = 0.01 ,random_state = 1)
ada.fit(X_std,y)
# print(X_std,y)
plot_decision_regions(X_std,y,classifier = ada)
plt.title("Adaline - stochastic gradient descent")
plt.xlabel("sepal")
plt.ylabel("petal")
plt.legend("lower right")
plt.tight_layout()
plt.show()

plt.plot(range(1,len(ada.costs) + 1),ada.costs,marker = "o")
plt.xlabel("Epochs")
plt.ylabel("Average Cost")
plt.show()


# In[ ]:


x = np.arange(5)
print(x)
rgen = np.random.RandomState(1)
print(rgen)
r = rgen.permutation(len(x))
print(type(r))
y = x[r]
print(y)
z = y[np.array([3,4])]
print(z)

