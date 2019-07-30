import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn import datasets
import pandas as pd


df = pd.read_csv('../data/indianapolis.csv').dropna()

df = df.loc[df['Contaminant']=='Chloroform']

cols = ['OutdoorHumidity','BarometricPressure','OutdoorTemp','IndoorConcentration']

def assign_quantiles(x):

    quantiles = df['IndoorConcentration'].quantile(np.arange(0,1.1,0.25)).values
    for i in range(len(quantiles)-1):
        if (x >= quantiles[i]) and (x < quantiles[i+1]):
            return i


X = df[cols].values

df['Percentile'] = df['IndoorConcentration'].apply(assign_quantiles)

est = KMeans(n_clusters=4)

fig = plt.figure(figsize=(4, 3), dpi=300)
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
est.fit(X)
labels = est.labels_
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels.astype(np.float), edgecolor='k')

ax.set_xlabel(cols[0])
ax.set_ylabel(cols[1])
ax.set_zlabel(cols[2])


fig = plt.figure(figsize=(4, 3))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

seasons_to_int_dict = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Fall': 3}
season_to_int = lambda x: seasons_to_int_dict[x]

#labels = df['Season'].apply(season_to_int).values
labels = df['Percentile'].values

ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, edgecolor='k')

ax.set_title('Truth')
ax.set_xlabel(cols[0])
ax.set_ylabel(cols[1])
ax.set_zlabel(cols[2])

plt.show()
