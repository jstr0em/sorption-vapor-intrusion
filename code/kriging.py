import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor

# observation coordinates
X = np.array([
    [1, 2],
    [1.5, 4],
    [0.5, 1.5],
    [5,8],
    [4, 3],
    [2.5, 6.5],
])
y = np.array([0.5, 1.5, 4.5, 3, 2, 6]) # observation data

gpr = GaussianProcessRegressor() # regressor function
gpr.fit(X, y)

res = 100 # prediction resolution
x1, x2 = np.meshgrid(np.linspace(0, 7, res), np.linspace(0, 10, res)) # grid to predict values onto
xx = np.vstack([x1.reshape(x1.size), x2.reshape(x2.size)]).T # stacks gridpoints
y_pred = gpr.predict(xx) # predicts values onto the grid
y_pred = y_pred.reshape((res, res)) # reshapes predicted values


# plotting
plt.contourf(x1,x2, y_pred)
plt.scatter(X[:,0], X[:,1], c=y)
plt.show()
