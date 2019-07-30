from plotly.offline import download_plotlyjs, init_notebook_mode,  iplot, plot
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from kriging import Kriging, Data



class Slices:
    def __init__(self, contaminant='Chloroform'):
        data_inst = Data()
        input_data = data_inst.get_data(contaminant=contaminant ,interpolate=True)
        max_time = pd.read_csv('./data/indianapolis.csv')['Time'].max()
        input_data = input_data[input_data['Date']<=max_time]
        self.gen_iterables(input_data)
        df = self.get_kriging_data(input_data)

        data = self.get_slice_data(df)
        layout = self.get_layout(data, contaminant)

        fig=dict(data=data, layout=layout)


        plot(fig,filename=contaminant.lower()+'.html')
        return

    def gen_iterables(self, input_data):
        times = np.sort(input_data['Date'].unique())
        depths = input_data['Depth'].unique()
        self.times = times
        self.depths = depths
        return

    def get_multi_index(self, iterables):
        index = pd.MultiIndex.from_product(iterables, names=['times', 'depth'])
        return index

    def get_iterables(self):
        return self.times, self.depths

    def get_kriging_data(self,input_data):
        times, depths = self.get_iterables()
        data_to_add = []
        for time in times:
            krig = Kriging(input_data[input_data['Date']==time])
            X, Y, Z, C = krig.x1, krig.x2, krig.x3, krig.pred
            for depth in depths:
                i = np.argmin(np.abs(Z[0][0] - depth))
                data_to_add.append(C.T[i])

        self.X, self.Y, self.Z = X, Y, Z
        index = self.get_multi_index([times, depths])
        df = pd.Series(data_to_add, index=index)
        self.df = df
        return df

    def get_grids(self):
        return self.X, self.Y, self.Z

    def get_dataframe(self):
        return self.df
    def get_slice_data(self,df):
        times, depths = self.get_iterables()
        X, Y, Z = self.get_grids()

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
        return data

    def get_slider_steps(self, data):
        times, depths = self.get_iterables()
        df = self.get_dataframe()
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
        return steps

    def get_sliders(self, data):
        steps = self.get_slider_steps(data)
        sliders = []
        sliders.append(
            dict(
                active=0,
                currentvalue={'prefix': 'Time: '},
                pad={'t': 50},
                steps=steps,
            )
        )
        return sliders

    def get_layout(self,data, contaminant):
        sliders = self.get_sliders(data)
        layout = dict(
            title='%s soil-gas concentration (ug/m^3)' % contaminant,
            sliders=sliders,
        )
        return layout

contaminants = pd.read_csv('./data/indianapolis.csv')['Contaminant'].unique()
for contaminant in contaminants:
    Slices(contaminant=contaminant)
