import numpy as np
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
LAMBDA = 0.000001

iris = load_iris ()
data = iris.data
data = data / data.max()
data = np.vstack (( data [:, 0], np.ones (( data.shape [0])), data [:, 2])).T
_, idx = np.unique(data [:, 0], return_index =True)
data = data[idx]
X, y = data [:, :-1], data [:, -1]
beta = np.linalg.inv((X.T @ X + LAMBDA* np.identity(X.shape[1]))) @ (X.T @ y)
y__ = X @ beta
K = rbf_kernel(X)
alpha = np.linalg.inv((K + LAMBDA * np.identity(K.shape[0]))) @ y
y_ = K @ alpha
plt.plot(X[:,0],y,label='Sepal vs Petal')
plt.plot(X[:,0],y__,label='Ridge Regression')
plt.plot(X[:,0],y_,label='Kernel Regression')
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.tight_layout()
plt.show()


