import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
import pandas as pd
from utils import get_dropbox_path
import sqlite3


class Data():
    # TODO: Analyze to see if data is normally distributed, if not perform relevant transformation (see Pocket link)
    def __init__(self):
        query = " \
            SELECT \
                StopDate, \
                Value AS Concentration, \
                Location \
            FROM \
                VOC_Data_SoilGas_and_Air \
            WHERE \
                Variable = 'Chloroform' AND \
                Depth_ft = 3.5 AND \
                (Location = 'SGP1' OR \
                Location = 'SGP2' OR \
                Location = 'SGP3' OR \
                Location = 'SGP4' OR \
                Location = 'SGP5' OR \
                Location = 'SGP6' OR \
                Location = 'SGP7') \
        ;"
        db = sqlite3.connect(get_dropbox_path() + '/var/Indianapolis.db')
        data = pd.read_sql_query(query, db)
        data['StopDate'] = data['StopDate'].apply(pd.to_datetime)
        # TODO: Make sure the pivoting doesn't misrepresent any values
        data = pd.pivot_table(
            data,
            index='StopDate',
            columns='Location',
            values='Concentration',
            aggfunc=np.max,
        ).interpolate()


        self.data = data
        return


class Kriging():
    def __init__(self, data):

        probe_locations = self.assign_locations(data)
        #observations = self.assign_observations(data.iloc[1])
        observations = data.iloc[-1].values

        x1, x2, grid = self.get_meshgrid()

        prediction = self.kriging(probe_locations, observations, grid)

        self.plot_results(x1, x2, prediction)
        return

    def load_probes(self):
        probes = pd.read_csv('./data/indianapolis_probes.csv')
        return probes

    def assign_locations(self, data):

        probes = self.load_probes()
        probe_locations = []
        for loc in list(data):
            probe_locations.append(probes.loc[probes['Location']==loc][['x','y']].values[0])

        probe_locations = np.array(probe_locations)
        return probe_locations

    def assign_observations(self, data):
        obs = []
        for loc in list(data):
            obs.append(data[loc].values[0])

        obs = np.array(obs)
        return obs

    def kriging(self, probe_locations, observations, grid):

        # TODO: Lookup if specific values are use for geophysics
        gpr = GaussianProcessRegressor() # regressor function
        gpr.fit(probe_locations, observations)

        y_pred = gpr.predict(grid) # predicts values onto the grid
        y_pred = y_pred.reshape((self.res, self.res)) # reshapes predicted values

        return y_pred

    def get_meshgrid(self):
        res = 200 # prediction resolution
        self.res = res
        x1, x2 = np.meshgrid(np.linspace(0, 25, res), np.linspace(0, 25, res)) # grid to predict values onto
        grid = np.vstack([x1.reshape(x1.size), x2.reshape(x2.size)]).T # stacks gridpoints
        return x1, x2, grid

    def plot_results(self, x1, x2, y_pred):

        plt.contourf(x1, x2, y_pred)
        plt.show()


        return

Kriging(Data().data)
