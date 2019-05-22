from plotly.offline import download_plotlyjs, init_notebook_mode,  iplot, plot
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from kriging import Kriging, Data

def get_lims_colors(surfacecolor):# color limits for a slice
    return np.min(surfacecolor), np.max(surfacecolor)

x = np.linspace(0,10,10)
y = np.linspace(0,10,10)
z = np.linspace(0,5,3)

X, Y, Z = np.meshgrid(x,y,z)


C = X**2+Z**2

data_inst = Data()
input_data = data_inst.get_data(interpolate=True)
times = input_data['Date'].unique()

# only using the first three dates

times = times[[0,5,10]]
depths = input_data['Depth'].unique()


iterables = [times, depths]

index = pd.MultiIndex.from_product(iterables, names=['times', 'depth'])

data_to_add = []
for time in times:
    krig = Kriging(input_data[input_data['Date']==time])
    X, Y, Z, C = krig.x1, krig.x2, krig.x3, krig.pred
    for depth in depths:
        i = np.argmin(np.abs(Z[0][0] - depth))
        data_to_add.append(C.T[i])


df = pd.Series(data_to_add, index=index)


# alternative route
data = []
for time in times:
    vmin, vmax = 0, 50
    for i, depth in enumerate(depths):
        j = np.argmin(np.abs(Z[0][0] - depth))
        data_now = dict(
            type='surface',
            #visible=True,
            surfacecolor=df[(time, depth)],
            cmin=vmin,
            cmax=vmax,
            z=Z.T[j],
            x=X.T[0],
            y=Y.T[0],
            showscale=False,
            colorscale='Jet',
        )

        if i % len(depths) == 0:
            data_now['showscale'] = True
            #print('Show scale for: ', i)
        data.append(data_now)


steps = []

for j, time in enumerate(times):
    i0 = j*len(depths)
    iend = (j+1)*len(depths)
    step = dict(
        method='restyle',
        args=['visible', [False] * len(data)],
        label='Time {}'.format(time)
    )
    for i in range(i0,iend):
        step['args'][1][i] = True
    steps.append(step)

sliders = []
sliders.append(
    dict(
        active=0,
        currentvalue={'prefix': 'Time: '},
        pad={'t': 50},
        steps=steps,
    )
)


layout = dict(
    title='Slices in volumetric data',
    sliders=sliders,
)

fig=dict(data=data, layout=layout)


plot(fig,filename='tmp.html')
