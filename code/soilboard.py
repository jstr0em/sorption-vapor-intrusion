from plotly.offline import download_plotlyjs, init_notebook_mode,  iplot, plot
from kriging import Kriging
import numpy as np


def get_lims_colors(surfacecolor):# color limits for a slice
    return np.min(surfacecolor), np.max(surfacecolor)


class Slices:
    def __init__(self):

        krig = Kriging()
        x1, x2, x3, grid, pred = krig.get_results()

        vmin, vmax = np.min(pred), np.max(pred)

        base_trace=dict(type='surface',
                x=x1.T[0],
                y=x2.T[0],
                z=x3.T[0],
                surfacecolor=pred.T[0],
                colorscale='jet',
                colorbar=dict(thickness=20, ticklen=4)
                )

        frames = []
        frame_zval = np.linspace(0,6, 50)
        for k in range(frame_zval.shape[0]):
            surfcol=pred.T[k]
            cmin, cmax=get_lims_colors(surfcol)
            vmin=min([cmin, vmin])
            vmax=max([cmax, vmax])
            frames.append(dict(data=[dict(z=x3.T[k],
                                     surfacecolor=pred.T[k])],
                               name='frame{}'.format(k)))

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
                            zaxis=dict(axis, **dict(range=[np.min(frame_zval), np.max(frame_zval)], autorange=False)),
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


        self.data = [base_trace]
        self.layout = layout
        self.frames = frames
        #plot(fig, validate=False, filename='soil_slice.html')
        return




graph = Slices()

import dash
import dash_core_components as dcc
import dash_html_components as html

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    html.Div(children='''
        Dash: A web application framework for Python.
    '''),

    dcc.Graph(
        id='example-graph',
        figure={
            'data': graph.data,
            'layout': graph.layout,
            'frames': graph.frames
        }
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
