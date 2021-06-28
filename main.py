# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.subplots

path='C:/Users/Xiahanzhong/Desktop/Machine Learning/Exercise/1/ex1data1.txt'
data=pd.read_csv(path,header=None,names=['Population','Profit'])
data.head()
print(data)
# data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))
# plt.show()
def computeCost(X,y,theta):
    inner=np.power(((X*theta.T)-y),2)
    return np.sum(inner)/(2*len(X))
data.insert(0,'ones',1)


cols=data.shape[1]
print(cols)
X=data.iloc[:,:-1]
y=data.iloc[:,cols-1:cols]
X.head()
y.head()
X=np.matrix(X.values)
y=np.matrix(y.values)
theta=np.matrix(np.array([0,0]))
print("{} {} {}".format(X.shape,y.shape,theta.shape))
print("{}".format(computeCost(X,y,theta)))
def gradientdescent(X,y,theta,alpha,iters):
    temp=np.matrix(np.zeros(theta.shape))
    parameters=int(theta.ravel().shape[1])#拉成一维数组
    cost=np.zeros(iters)
    for i in range(iters):
        error = (X*theta.T)-y

        for j in range(parameters):
            term=np.multiply(error,X[:,j])
            temp[0,j]=theta[0,j]-(alpha/len(X))*np.sum(term)
        theta=temp
        cost[i]=computeCost(X,y,theta)
    return theta,cost
g,cost=gradientdescent(X,y,theta,0.01,1500)
print(g)
predict1 = [1,3.5]*g.T
print("predict1:",predict1)
predict2 = [1,7]*g.T
print("predict2:",predict2)
x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = g[0, 0] + (g[0, 1] * x)

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()