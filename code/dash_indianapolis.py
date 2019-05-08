# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

df = pd.read_csv('./data/indianapolis.csv')
df['Time'] = df['Time'].apply(pd.to_datetime)


layout = [
    html.Div(
        html.H1('Indianapolis Data Analysis App')
    ),

    html.Div([
        html.Div([
            html.P('Time-series'),
            dcc.Graph(
                id='time-series',
                figure={
                    'data': [
                        go.Scatter(
                            x=df[df['Contaminant'] == i]['Time'],
                            y=df[df['Contaminant'] == i]['IndoorConcentration'],
                            name=i,
                        ) for i in df['Contaminant'].unique()
                    ],
                    'layout': {
                        'title': 'Dash Data Visualization',
                    }
                }
            ),
        ]),
        html.Div([
            html.P('Box Plot'),
            dcc.Graph(
                id='box',
                figure={
                    'data': [
                        go.Box(
                            y=df[df['Contaminant'] == i]['IndoorConcentration'].dropna(),
                            name=i,
                        ) for i in df['Contaminant'].unique()
                    ],
                    'layout': 'Boxplot',
                }
            )
        ],
        className='four columns chart_div')
    ],
    className="row")
]

app.layout = html.Div(layout)

if __name__ == '__main__':
    app.run_server(debug=True)
