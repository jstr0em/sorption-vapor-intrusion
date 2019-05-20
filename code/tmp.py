from plotly.offline import download_plotlyjs, init_notebook_mode,  iplot, plot
import numpy as np
import pandas as pd
import plotly.graph_objs as go


def get_lims_colors(surfacecolor):# color limits for a slice
    return np.min(surfacecolor), np.max(surfacecolor)

x = np.linspace(0,10,10)
y = np.linspace(0,10,10)
z = np.linspace(0,5,3)

X, Y, Z = np.meshgrid(x,y,z)
C = X**2+Z**2

times = ['2011/03/12','2012/03/12']

"""
iterables = [times, z]

index = pd.MultiIndex.from_product(iterables, names=['times', 'depth'])
# stores data in df
data_to_add = []
for time in times:
    for i in range(len(z)):
        data_to_add.append(C.T[i])

df = pd.Series(data_to_add, index=index)

# df is in the correct format now...
#print(df[(times[0], z[2])])
data = []
for time in times:
    for depth in z:
        data_now = dict(
            type='surface',
            surfacecolor=df[time, depth],
            colorscale='jet',
        )

        data.append(data_now)


steps = []

for i in range(len(data)):
    step = dict(
        method='restyle',
        args=['visible', [False] * len(data_slider)],
        label='Year {}'.format(i + 1960)
    )
    step['args'][1][i] = True
    steps.append(step)
"""
# alternative route
data = []
for i in range(3):
    data_now = dict(
        type='surface',
        visible=True,
        surfacecolor=C.T[i],
        z=Z.T[i],
        x=X.T[0],
        y=Y.T[0],
    )
    data.append(data_now)


plot(data,filename='tmp.html')
