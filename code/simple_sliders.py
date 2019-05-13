from plotly.offline import download_plotlyjs, init_notebook_mode,  iplot, plot
#import plotly.plotly as py
import numpy as np


x = np.linspace(0,10)
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


print(data[0])
print(len(data[0]))
#data[5]['visible'] = True

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
