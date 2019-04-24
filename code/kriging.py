import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
import pandas as pd
from utils import get_dropbox_path
import sqlite3

probes = pd.read_csv('./data/indianapolis_probes.csv')


query = " \
    SELECT \
        Value AS Concentration, \
        Location \
    FROM \
        VOC_Data_SoilGas_and_Air \
    WHERE \
        Variable = 'Chloroform' AND \
        StopDate = '2011-08-12' AND \
        Depth_ft = 13.0 \
;"
db = sqlite3.connect(get_dropbox_path() + '/var/Indianapolis.db')
data = pd.read_sql_query(query, db)



X = []
y = []
for loc in data['Location']:
    X.append(probes.loc[probes['Location']==loc][['x','y']].values[0])
    y.append(data.loc[data['Location']==loc]['Concentration'].values[0])


X = np.array(X)
y = np.array(y)
# observation coordinates
"""
X = np.array([
    [1, 2],
    [1.5, 4],
    [0.5, 1.5],
    [5,8],
    [4, 3],
    [2.5, 6.5],
])
"""
#y = np.array([0.5, 1.5, 4.5, 3, 2, 6]) # observation data

gpr = GaussianProcessRegressor() # regressor function
gpr.fit(X, y)

res = 200 # prediction resolution
x1, x2 = np.meshgrid(np.linspace(0, 20, res), np.linspace(0, 25, res)) # grid to predict values onto
xx = np.vstack([x1.reshape(x1.size), x2.reshape(x2.size)]).T # stacks gridpoints
y_pred = gpr.predict(xx) # predicts values onto the grid
y_pred = y_pred.reshape((res, res)) # reshapes predicted values


# plotting
plt.contourf(x1,x2, y_pred)
plt.scatter(X[:,0], X[:,1], c=y)
plt.show()
