from plotly.offline import download_plotlyjs, init_notebook_mode,  iplot, plot
import numpy as np
import pandas as pd

def get_lims_colors(surfacecolor):# color limits for a slice
    return np.min(surfacecolor), np.max(surfacecolor)

x = np.linspace(0,10,10)
y = np.linspace(0,10,10)
z = np.linspace(0,5,3)

X, Y, Z = np.meshgrid(x,y,z)
C = X**2+Z**2
vmin, vmax = np.min(C), np.max(C)


data = [
    dict(type='surface',
        x=X.T[0],
        y=Y.T[0],
        z=Z.T[0],
        surfacecolor=C.T[0],
        colorscale='jet',
        colorbar=dict(thickness=20, ticklen=4),
        )
    ]




frames = []
for k in range(len(z)):
    surfcol=C.T[k]
    cmin, cmax=get_lims_colors(surfcol)
    vmin=min([cmin, vmin])
    vmax=max([cmax, vmax])
    frames.append(dict(data=[dict(z=Z.T[k],
                             surfacecolor=C.T[k])],
                       name='frame{}'.format(k)))
