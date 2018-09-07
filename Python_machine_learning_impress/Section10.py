
# coding: utf-8

# In[ ]:


#import pandas as pd

# df = pd.read_csv("https://raw.githubusercontent.com/rabst/python-machine-learning-book-2nd-edition/blob/master/code/ch10/housing.data.txt", header=None, sep="\s+")
# df.columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]
# df.head


# In[ ]:


import pandas as pd

df = pd.read_csv("housing.data.txt", sep="\s+")
df.columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]
df.head()


# In[ ]:


#散布図データ解析　散布図行列　#seaborn package
import matplotlib.pyplot as plt
import seaborn as sb
cols = ["LSTAT", "INDUS", "NOX", "RM", "MEDV"]
sb.pairplot(df[cols], size=2.5)
plt.tight_layout()
plt.show()


# In[ ]:


#peaarson product-moment correlation coefficent
import numpy as np
cm = np.corrcoef(df[cols].values.T)
hm = sb.heatmap(cm, cbar=True, annot=True, square=True, fmt=".2f", annot_kws={"size": 15}, 
               yticklabels=cols, xticklabels=cols)
plt.tight_layout()
plt.show()
#cbar >> color bar, cm >> correlation map, annot_kws >> annotation_keywords


# In[ ]:


#最小二乗法、線形最小二乗法
#adaline >> adaptive linear neuron,, gd, sgd
class LinearRegressionGD(object):
    def __init__(self, eta=0.001, n_iter=20):
        self.eta = eta
        self.n_iter = n_iter
        
    def fit(self, X, y):
        self.w = np.zeros(1 + X.shape[1])
        self.costs = []
        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w[1:] += self.eta * X.T.dot(errors)
            self.w[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.costs.append(cost)
        return self
    
    def net_input(self, X):
        return np.dot(X, self.w[1:]) + self.w[0]
    
    def predict(self, X):
        return self.net_input(X)


# In[ ]:


#できるだけ２次配列に収納する
X = df[["RM"]].values
y = df["MEDV"].values
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()
lr = LinearRegressionGD()
lr.fit(X_std, y_std)


# In[ ]:


plt.plot(range(1, lr.n_iter+1), lr.costs)
plt.ylabel("SSE")
plt.xlabel("Epoch")
plt.show()


# In[ ]:


def lin_regplot(X, y, model):
    plt.scatter(X, y, c="steelblue", edgecolor="white", s=70)
    plt.plot(X, model.predict(X), color="black", lw=2)
    return



# In[ ]:


lin_regplot(X_std, y_std, lr)
plt.xlabel("ave of RM")
plt.ylabel("price of $1000s in MEDV")
plt.show()


# In[ ]:


num_rooms_std = sc_x.transform(np.array([5.0]).reshape(1, -1))
#room number なのに　5.0 なのは　float で正確に計算したいから
price_std = lr.predict(num_rooms_std)
print("price in $1000s: {}".format(sc_y.inverse_transform(price_std)))


# In[ ]:


print ("slope: {}".format(lr.w[1]))
print("intercept: {:.3f}".format(lr.w[0]))


# In[ ]:


from sklearn.linear_model import LinearRegression
slr = LinearRegression()
slr.fit(X, y)
y_pred = slr.predict(X)
print("slope;{:.3f}".format(slr.coef_[0]))
print("intercept:{:.3f}".format(slr.intercept_))


# In[ ]:


lin_regplot(X, y, slr)
plt.xlabel("ave of RM")
plt.ylabel("prince in $1000s in MEDV")
plt.show()


# In[ ]:


#take outside data away
#instead use RANSAC >> random sample concensus
#learn inlier 
from sklearn.linear_model import RANSACRegressor
#make instance
ransac = RANSACRegressor(LinearRegression(), max_trials=100, min_samples=50, loss="absolute_loss",
                        residual_threshold=5.0, random_state=0)
#residual_threshlold >> max distance from the line
ransac.fit(X, y)
#fit


# In[ ]:


#median absolute deviation >> one of deviation 
#in these days, there are other deviations for inlier
inlier_mask = ransac.inlier_mask_
#take inlier mask
print(inlier_mask)
outlier_mask = np.logical_not(inlier_mask)
print(outlier_mask)
line_X = np.arange(3, 10, 1)
line_y_ransac = ransac.predict(line_X[:, np.newaxis])
#change the demention from 1 to 2
plt.scatter(X[inlier_mask], y[inlier_mask], c="steelblue", edgecolor="white", marker="o", label="inliers" )
plt.scatter(X[outlier_mask], y[outlier_mask], c="limegreen", edgecolor="white", marker="x", label="outliers" )
plt.plot(line_X, line_y_ransac, color="black", lw=2)
plt.xlabel("RM")
plt.ylabel("MEDV")
plt.legend(loc = "upper left")
plt.show()


# In[ ]:


print("slope: {:3f}".format(ransac.estimator_.coef_[0]))
print("intercept: {:3f}".format(ransac.estimator_.intercept_))


# In[ ]:


#all variables
from sklearn.model_selection import train_test_split
X = df.iloc[:, :-1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
slr = LinearRegression()
slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)


# In[ ]:


#residal plot
plt.scatter(y_train_pred, y_train_pred-y_train, c="steelblue", edgecolor="white", label="trainig data")
plt.scatter(y_test_pred, y_test_pred-y_test, c="limegreen", edgecolor="white", label="test data")
plt.xlabel("predicted values")
plt.ylabel("residals")
plt.legend(loc="upper left")
plt.hlines(y=0, xmin=-10, xmax=50, color="black", lw=2)
plt.tight_layout()
plt.show()


# In[ ]:


from sklearn.metrics import mean_squared_error
print("mse train:{:3f}, test:{:3f}".format(mean_squared_error(y_train, y_train_pred), 
                                           mean_squared_error(y_test, y_test_pred)))


# In[ ]:


#r**2 errors
#r**2 =: 1 -(sse/sst)
from sklearn.metrics import r2_score
print("r2_train: {:3f}, _test: {:3f}".format(r2_score(y_train, y_train_pred),
                                            r2_score(y_test, y_test_pred)))


# In[ ]:


#Regularizaition >> ridgeregression, least absolute shrinkage and selection operator, elastic net
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=1.0)
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1.0)
from sklearn.linear_model import ElasticNet
elasnet = ElasticNet(alpha=1.0, l1_ratio=0.5)


# In[ ]:


from sklearn.preprocessing import PolynomialFeatures
#poly >> many, nomial >> demention
X = np.array([258.0, 270.0, 320.0, 342.0, 368.0, 396.0, 446.0, 480.0, 586.0])[:, np.newaxis]
y = np.array([236.4, 234.4, 252.8, 298.6, 314.2, 342.2, 360.8, 391.2, 390.8])
lr = LinearRegression()
pr = LinearRegression()
quadratic = PolynomialFeatures(degree=2)
X_quad = quadratic.fit_transform(X)


# In[ ]:


lr.fit(X, y)
X_fit = np.arange(250, 600, 10)[:, np.newaxis]
#from 1d to 2d
y_lin_fit = lr.predict(X_fit)


# In[ ]:


pr.fit(X_quad, y)
y_quad_fit = pr.predict(quadratic.fit_transform(X_fit))
plt.scatter(X, y, label="training points")
plt.plot(X_fit, y_lin_fit, label="linear fit", linestyle="--")
plt.plot(X_fit, y_quad_fit, label="quadratic fit")
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()


# In[ ]:


y_lin_pred = lr.predict(X)
y_quad_pred= pr.predict(X_quad)
print("training mse linear:{:3f} quadracic {:3f}".format(mean_squared_error(y, y_lin_pred),
                                                        mean_squared_error(y, y_quad_pred)))
print("training r^2 linear:{:3f} qeadratic {:3f}".format(r2_score(y, y_lin_pred),
                                                        r2_score(y, y_quad_pred)))


# In[ ]:


X = df[["LSTAT"]].values
y = df["MEDV"].values
regr = LinearRegression()

#make features in 2d and 3d
quadratic = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)
X_quad = quadratic.fit_transform(X)
X_cubic = cubic.fit_transform(X)
#linear model prediction
X_fit=np.arange(X.min(), X.max(), 1)[:, np.newaxis]
regr = regr.fit(X, y)
y_lin_fit  = regr.predict(X_fit)
linear_r2 = r2_score(y, regr.predict(X))

regr = regr.fit(X_quad, y)
y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
quadratic_r2 = r2_score(y, regr.predict(X_quad))

regr = regr.fit(X_cubic, y)
y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
cubic_r2 = r2_score(y, regr.predict(X_cubic))
#plotting
plt.scatter(X, y, label="training points", color="lightgray")
plt.plot(X_fit, y_lin_fit, label="linear(d1) $R^2={}".format(linear_r2), 
        color="blue", lw=2, linestyle="--")
plt.plot(X_fit, y_quad_fit, label="quadratic(d2) $R^2={}".format(quadratic_r2), 
        color="red", lw=2, linestyle="-")
plt.plot(X_fit, y_cubic_fit, label="cubic(d3) $R^2={}".format(cubic_r2), 
        color="yellow", lw=2, linestyle=":")
plt.xlabel("% lower status [LSTAT]")
plt.ylabel("price in $1000s [MEDV]")
plt.legend(loc="upper left")
plt.show()


# In[ ]:


#f(x) = 2**(-x)
X_log = np.log(X)
y_sqrt = np.sqrt(y)
#calcurete squart
X_fit = np.arange(X_log.min()-1, X_log.max()+1, 1)[:, np.newaxis]
regr = regr.fit(X_log, y_sqrt)
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(y_sqrt, regr.predict(X_log))
plt.scatter(X_log, y_sqrt, label="training points", color="lightgray")
plt.plot(X_fit, y_lin_fit, label="linear (d=1) $R^2={}".format(linear_r2), color="blue", lw=2)
plt.xlabel("log(% loxer status of the population)")
plt.ylabel("$\sqrt{price /; in /; /$1000s [medv]}$")
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()


# In[ ]:


#random forest decision tree
from sklearn.tree import DecisionTreeRegressor
X = df[["LSTAT"]].values
y = df["MEDV"].values
tree = DecisionTreeRegressor(max_depth=3)
tree.fit(X, y)
sort_ind = X.flatten().argsort()
lin_regplot(X[sort_ind], y[sort_ind], tree)
plt.xlabel("% lower status of populatino [LSTAT]")
plt.ylabel("price in $1000s [MEDV]")
plt.show()


# In[ ]:


X = df.iloc[:, :-1].values
y = df["MEDV"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(n_estimators=1000, criterion="mse", random_state=1,
                              n_jobs=-1)
forest.fit(X_train, y_train)
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)
print("MSE train {:3f}, test {:3f}".format(mean_squared_error(y_train, y_train_pred),
                                          mean_squared_error(y_test, y_test_pred)))
print("R^2 train {:3f}, test {:3f}".format(r2_score(y_train, y_train_pred),
                                          r2_score(y_test, y_test_pred)))


# In[ ]:


plt.scatter(y_train_pred, y_train_pred-y_train, c="steelblue",
           edgecolor="white", marker="o", s=35, alpha=0.5, label="training data")
plt.scatter(y_test_pred, y_test_pred-y_test, c="limegreen",
           edgecolor="white", marker="s", s=35, alpha=0.5, label="test data")
plt.xlabel("predicted values")
plt.ylabel("Residuals")
plt.legend(loc="upper left")
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color="black")
plt.tight_layout()
plt.xlim([-10, 50])
plt.show()

