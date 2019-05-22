import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
import pandas as pd
import sqlite3
from math import isnan
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern
from utils import get_dropbox_path
from plotly.offline import download_plotlyjs, init_notebook_mode,  iplot, plot

class Data:
    # static variable
    m_in_ft = 0.3048
    def __init__(self):
        self.db = sqlite3.connect(get_dropbox_path() + '/var/Indianapolis.db')
        return

    @staticmethod
    def get_depths():
        depths = [3.5, 6.0, 9.0, 13.0, 16.5]
        return depths

    def get_data(self,depth=3.5, interpolate=False):
        depths = self.get_depths() # gets the unique probe depth values

        dfs = [] # list to store dataframes that will be concatenated
        for depth in depths:
            # query
            query = " \
                SELECT \
                    StopDate AS Date, \
                    Value AS Concentration, \
                    Location \
                FROM \
                    VOC_Data_SoilGas_and_Air \
                WHERE \
                    Variable = 'Chloroform' AND \
                    Depth_ft = %f AND \
                    (Location = 'SGP1' OR \
                    Location = 'SGP2' OR \
                    Location = 'SGP3' OR \
                    Location = 'SGP4' OR \
                    Location = 'SGP5' OR \
                    Location = 'SGP6' OR \
                    Location = 'SGP7') \
            ;" % depth
            # read data from database
            df = pd.read_sql_query(query, self.db)
            df['Date'] = df['Date'].apply(pd.to_datetime)
            # TODO: Make sure the pivoting doesn't misrepresent any values
            df = pd.pivot_table(
                df,
                index='Date',
                columns='Location',
                values='Concentration',
                aggfunc=np.max,
            ).reset_index()

            df['Depth'] = np.repeat(-depth*self.m_in_ft,len(df))

            if interpolate is True:
                df = df.interpolate()

            dfs.append(df)

        data = pd.concat(dfs).reset_index(drop=True)

        return data

class Kriging():
    def __init__(self,data):
        data_processing = Data()

        #data = data_processing.get_data(interpolate=True)



        #coords, vals = self.get_coords_vals(data[data['Date']=='2011-01-07'])
        coords, vals = self.get_coords_vals(data)
        x1, x2, x3, grid = self.get_meshgrid()
        pred = self.kriging(coords, vals, grid)
        predictions, grids = {}, {}


        self.x1, self.x2, self.x3, self.grid = x1, x2, x3, grid
        self.pred = pred
        """
        for time in data['Date'].unique():
        predictions = {}

        for i, time in enumerate(data.index):
            observations = data.iloc[i].values
            prediction = self.kriging(probe_locations, observations, grid)
            predictions[str(time.date())] = prediction


        print(grids)
        """
        #self.plot_results(x1, x2, predictions)
        return

    def get_results(self,log=False):
        if log is False:
            return self.x1, self.x2, self.x3, self.grid, self.pred
        elif log is True:
            self.pred = np.log10(self.pred)
            return self.x1, self.x2, self.x3, self.grid, self.pred
    def load_probes_coordinates(self):
        probe_coords = pd.read_csv('./data/indianapolis_probes.csv')
        return probe_coords

    def get_coords_vals(self, data):
        # loads the probe coordinates
        probe_coords = self.load_probes_coordinates()
        # list to store the active probe coordinates and associated values
        coords, vals = [], []

        depths = data['Depth'].unique()
        locations = list(data)[1:-1] # only loads the location names (first col is always date and last is always depth)

        for depth in depths:
            for loc in locations:
                probe_val = data.loc[data['Depth']==depth][loc].values[0] # current probe value
                # checks if the probe value is nan, if yes, skips those coords and vals
                if isnan(probe_val):
                    continue
                else:
                    xy = probe_coords.loc[probe_coords['Location']==loc][['x','y']].values[0].tolist()
                    z = [depth]
                    coords.append(xy + z)
                    vals.append(probe_val)

        coords = np.array(coords)
        vals = np.array(vals)
        return coords, vals

    def kriging(self, coords, vals, grid):
        # TODO: Lookup if specific values are use for geophysics or which kernel is best (might need to make a custom one)

        kernel = ConstantKernel()*RBF()
        #kernel = RBF()
        gpr = GaussianProcessRegressor(kernel=kernel) # regressor function
        gpr.fit(coords, vals)

        pred = gpr.predict(grid) # predicts values onto the grid
        pred = pred.reshape((self.res, self.res, 50)) # reshapes predicted values
        return pred

    def get_meshgrid(self):
        res = 200 # prediction resolution
        self.res = res
        x1, x2, x3 = np.meshgrid(np.linspace(0, 25, res), np.linspace(0, 25, res), np.linspace(0, -6, 50)) # grid to predict values onto
        grid = np.vstack([x1.reshape(x1.size), x2.reshape(x2.size), x3.reshape(x3.size)]).T # stacks gridpoints
        return x1, x2, x3, grid

        return
