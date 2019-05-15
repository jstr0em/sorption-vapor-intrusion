from plotly.offline import download_plotlyjs, init_notebook_mode,  iplot, plot
#import plotly.plotly as py
import numpy as np
import pandas as pd

x = np.linspace(0,10,100)
"""
data = []
for amp in [0, 1, 2]:
    data_to_append = [dict(
        visible = False,
        line=dict(color='#00CED1', width=6),
        name = 'nu = '+str(step) + 'amp = '+str(amp),
        x = x,
        y = amp+np.sin(step*x)
    ) for step in np.arange(0,11)]
    data.append(data_to_append)

"""
freqs = np.arange(0,11)
data = [dict(
    visible = False,
    line=dict(color='#00CED1', width=6),
    name = 'nu = '+str(freq),
    x = x,
    y = np.sin(freq*x)
) for freq in freqs
]
data[5]['visible'] = True


sliders = []

steps = []
for i in range(len(freqs)):
    step = dict(
        method='restyle',
        args=['visible', [False]*len(freqs)],
    )
    step['args'][1][i] = True
    steps.append(step)

sliders.append(
    dict(
        active=5,
        currentvalue={'prefix': 'Frequency: '},
        pad={'t': 50},
        steps=steps,
    )
)


layout = dict(sliders=sliders)

fig = dict(data=data, layout=layout)


def pairwise(iterable):
    a = iter(iterable)
    return zip(a, a)

df = pd.DataFrame({'x': [0,1], 'y': [2,3], 'z': [4,5], 'u': [6,7]})


for x, y in pairwise(df.columns):
    print(x,y)
#plot(fig, filename='wave.html')

"""
steps = []
for i in range(len(data)):
    step = dict(
        method = 'restyle',
        args = ['visible', [False] * len(data)],
    )
    step['args'][1][i] = True # Toggle i'th trace to "visible"
    steps.append(step)

steps2 = []

sliders = [
    # slider 1
    dict(
        active = 10,
        currentvalue = {"prefix": "Frequency: "},
        pad = {"t": 50},
        steps = steps,
    ),
    # slider 2
    dict(
        active = 0,
        currentvalue = {"prefix": "Amplitude: "},
        pad = {"t": 150},
        steps = steps2,
    )
]

layout = dict(sliders=sliders)

fig = dict(data=data, layout=layout)

#plot(fig, filename='wave.html')
"""
