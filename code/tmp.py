from plotly.offline import download_plotlyjs, init_notebook_mode,  iplot, plot
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from kriging import Kriging, Data


data_inst = Data()
input_data = data_inst.get_data(interpolate=True)
max_time = pd.read_csv('./data/indianapolis.csv')['Time'].max()

input_data = input_data[input_data['Date']<=max_time]
times = np.sort(input_data['Date'].unique())
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
    vmin, vmax = 0, np.max(list(df[(time,)]))

    if vmax <= 0.1:
        continue
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


print(len(data))
steps = []
j = 0
for time in times:
    vmin, vmax = 0, np.max(list(df[(time,)]))

    if vmax <= 0.1:
        continue
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
    j += 1


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
    title='Chloroform soil-gas concentration (ug/m^3)',
    sliders=sliders,
)

fig=dict(data=data, layout=layout)


plot(fig,filename='tmp.html')
