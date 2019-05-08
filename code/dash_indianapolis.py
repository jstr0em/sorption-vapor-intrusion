# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

df = pd.read_csv('./data/indianapolis.csv')
df['Time'] = df['Time'].apply(pd.to_datetime)


col_options = ['IndoorConcentration','IndoorOutdoorPressure','OutdoorTemp']

layout = [
    html.Div(
        html.H1('Indianapolis Data Analysis App')
    ),
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='x-axis-col',
                options=[{'label': i, 'value': i} for i in col_options],
                value=col_options[1],
            ),
            dcc.RadioItems(
                id='x-axis-type',
                options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                value='Linear',
                labelStyle={'display': 'inline-block'}
            ),
        ],
        className='three columns'),
        html.Div([
            dcc.Dropdown(
                id='y-axis-col',
                options=[{'label': i, 'value': i} for i in col_options],
                value=col_options[0],
            ),
            dcc.RadioItems(
                id='y-axis-type',
                options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                value='Linear',
                labelStyle={'display': 'inline-block'}
            ),
        ],
        className='three columns')
    ],
    className='row'),
    html.Div([
        html.Div(
            dcc.Graph(id='time-series'),
        className='six columns'),
    ],
    className="row")
]


app.layout = html.Div(layout)

@app.callback(
    Output('time-series', 'figure'),
    [Input('x-axis-col', 'value'),
     Input('y-axis-col', 'value'),
     Input('x-axis-type', 'value'),
     Input('y-axis-type', 'value')])
def update_graph(xaxis_column_name, yaxis_column_name, xaxis_type, yaxis_type):

    return {
        'data': [go.Scatter(
            x=df[xaxis_column_name],
            y=df[yaxis_column_name],
            #text=df[df['Indicator Name'] == yaxis_column_name]['Country Name'],
            mode='markers',
            marker={
                'size': 15,
                'opacity': 0.5,
                'line': {'width': 0.5, 'color': 'white'}
            }
        )],
        'layout': go.Layout(
            xaxis={
                'title': xaxis_column_name,
                'type': 'linear' if xaxis_type == 'Linear' else 'log'
            },
            yaxis={
                'title': yaxis_column_name,
                'type': 'linear' if yaxis_type == 'Linear' else 'log'
            },
            margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
            hovermode='closest'
        )
    }


if __name__ == '__main__':
    app.run_server(debug=True)
