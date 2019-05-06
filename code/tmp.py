import numpy as np
from plotly.offline import download_plotlyjs, init_notebook_mode,  iplot, plot
#init_notebook_mode(connected=True)


pl_clrsc= [[0.0, "rgb(20,29,67)"],
           [0.05, "rgb(25,52,80)"],
           [0.1, "rgb(28,76,96)"],
           [0.15, "rgb(23,100,110)"],
           [0.2, "rgb(16,125,121)"],
           [0.25, "rgb(44,148,127)"],
           [0.3, "rgb(92,166,133)"],
           [0.35, "rgb(140,184,150)"],
           [0.4, "rgb(182,202,175)"],
           [0.45, "rgb(220,223,208)"],
           [0.5, "rgb(253,245,243)"],
           [0.55, "rgb(240,215,203)"],
           [0.6, "rgb(230,183,162)"],
           [0.65, "rgb(221,150,127)"],
           [0.7, "rgb(211,118,105)"],
           [0.75, "rgb(194,88,96)"],
           [0.8, "rgb(174,63,95)"],
           [0.85, "rgb(147,41,96)"],
           [0.9, "rgb(116,25,93)"],
           [0.95, "rgb(82,18,77)"],
           [1.0, "rgb(51,13,53)"]]

def get_lims_colors(surfacecolor):# color limits for a slice
    return np.min(surfacecolor), np.max(surfacecolor)




volume=lambda x,y,z: (x-2*y)*np.exp(-x**2-0.7*y**2-z**2)

x=np.linspace(-2.2,2.2, 50)
y=np.linspace(-2.2,2.2, 50)
x,y=np.meshgrid(x,y)
z=-2*np.ones(x.shape)
surfcolor_z=volume(x,y,z)
vmin=np.min(surfcolor_z)
vmax=np.max(surfcolor_z)

base_trace=dict(type='surface',
                x=x,
                y=y,
                z=z,
                surfacecolor=surfcolor_z,
                colorscale=pl_clrsc,
                colorbar=dict(thickness=20, ticklen=4)
                )


frame_zval=np.arange(-2,1.75, 0.1)

print(base_trace)

frames=[]
for k in range(frame_zval.shape[0]):
    zz=frame_zval[k]*np.ones(x.shape)
    surfcol=volume(x,y,zz)
    cmin, cmax=get_lims_colors(surfcol)
    vmin=min([cmin, vmin])
    vmax=max([cmax, vmax])
    frames.append(dict(data=[dict(z=zz,
                             surfacecolor=volume(x,y,zz))],
                       name='frame{}'.format(k)),
                 )

"""
base_trace.update(cmin=vmin, cmax=vmax)



sliders=[dict(steps= [dict(method= 'animate',#Sets the Plotly method to be called when the
                                                #slider value is changed.
                           args= [[ 'frame{}'.format(k) ],#Sets the arguments values to be passed to
                                                              #the Plotly method set in method on slide
                                  dict(mode= 'immediate',
                                  frame= dict( duration=50, redraw= False ),
                                           transition=dict( duration= 0)
                                          )
                                    ],
                            label='{:.2f}'.format(frame_zval[k])
                             ) for k in range(frame_zval.shape[0])],
                transition= dict(duration= 0 ),
                x=0,#slider starting position
                y=0,
                currentvalue=dict(font=dict(size=12),
                                  prefix='z: ',
                                  visible=True,
                                  xanchor= 'center'
                                 ),
                len=1.0)#slider length)
           ]


axis = dict(showbackground=True,
            backgroundcolor="rgb(230, 230,230)",
            gridcolor="rgb(255, 255, 255)",
            zerolinecolor="rgb(255, 255, 255)",
            )


layout = dict(
         title='Slices in volumetric data',
    font=dict(family='Balto'),
         width=600,
         height=600,
         scene=dict(xaxis=(axis),
                    yaxis=(axis),
                    zaxis=dict(axis, **dict(range=[-2,1.75], autorange=False)),
                    aspectratio=dict(x=1,
                                     y=1,
                                     z=1
                                     ),
                    ),
         updatemenus=[dict(type='buttons', showactive=False,
                                y=1,
                                x=1.3,
                                xanchor='right',
                                yanchor='top',
                                pad=dict(t=0, r=10),
                                buttons=[dict(label='Play',
                                              method='animate',
                                              args=[None,
                                                    dict(frame=dict(duration=30,
                                                                    redraw=False),
                                                         transition=dict(duration=0),
                                                         fromcurrent=True,
                                                         mode='immediate'
                                                        )
                                                   ]
                                             )
                                        ]
                               )
                          ],
    sliders=sliders
        )


fig=dict(data=[base_trace], layout=layout, frames=frames)
"""
#plot(fig, validate=False)
