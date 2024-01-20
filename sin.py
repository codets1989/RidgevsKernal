import numpy as np
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel

LAMBDA = 0.000001
X = np.linspace ( -2.5* np.pi , 2.5* np.pi).reshape ((-1, 1))
y = np.sin(X)
X = np.hstack ((X, np.ones ((X.shape [0], 1))))

beta = np.linalg.inv(X.T @ X + LAMBDA * np.identity(X.shape[1]))@(X.T @ y)
y__ = X @ beta

K = rbf_kernel(X)
alpha = np.linalg.inv((K + LAMBDA * np.identity(K.shape[0]))) @ y
y_ = K @ alpha

plt.plot(X[:,0],y,label='$y = \\sin(x)$')
plt.plot(X[:,0],y_+0.05, label='Kernel Regression')
plt.plot(X[:,0],y__,label='Ridge regression')
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.tight_layout()
plt.show()