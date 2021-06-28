import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.subplots
import scipy.optimize as opt
path='C:/Users/Xiahanzhong/Desktop/Machine Learning/Exercise/ex2-logistic regression/ex2data1.txt'
data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
data.head()

# data.head()
# data_rows=data.shape[0]
# data_cols=data.shape[1]
# #print("{}*{}".format(data_rows,data_cols))
posi = data[data['Admitted'].isin([1])]
nega = data[data['Admitted'].isin([0])]

# fig, ax = plt.subplots(figsize=(12,8))
# ax.scatter(posi['Exam 1'], posi['Exam 2'], s=50, c='b', marker='o', label='Admitted')
# ax.scatter(nega['Exam 1'], nega['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')
# ax.legend()
# ax.set_xlabel('Exam 1 Score')
# ax.set_ylabel('Exam 2 Score')
#plt.show()
def sigmoid(z):
    return 1/(1+np.exp(-z))

def cost(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    return np.sum(first - second) / (len(X))
data.insert(0,'ones',1)
cols=data.shape[1]
X = data.iloc[:,0:cols-1]
y=data.iloc[:,cols-1:cols]
theta=np.zeros(3)
X=np.array(X.values)
y=np.array(y.values)

e=np.eye(3)


def gradient(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    error = sigmoid(X * theta.T) - y
    for i in range(parameters):
        term = np.multiply(error, X[:, i])
        grad[i] = np.sum(term) / len(X)

    return grad

result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))

Newtheta=np.zeros(3)
Newtheta=np.matrix(Newtheta)
Newtheta=result[0]
print(Newtheta)
plot_x1=np.linspace(20,100,100)
plot_y1=-1/Newtheta[2]*(Newtheta[0]+plot_x1*Newtheta[1])
fig,ax=plt.subplots(figsize=(14,9))
ax.plot(plot_x1,plot_y1,label='Boundary')
ax.scatter(posi['Exam 1'], posi['Exam 2'], s=50, c='b', marker='o', label='Admitted')
ax.scatter(nega['Exam 1'], nega['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
plt.show()
print(X)
def hfunc(theta,X):
    return sigmoid(np.dot(theta.T,X))
print(hfunc(Newtheta,[1,45,85]))

