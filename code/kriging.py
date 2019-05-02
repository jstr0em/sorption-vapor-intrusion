import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
import pandas as pd

import sqlite3


class Data:
    # static variable
    m_in_ft = 0.3048
    def __init__(self):
        from utils import get_dropbox_path
        self.db = sqlite3.connect(get_dropbox_path() + '/var/Indianapolis.db')
        return

    @staticmethod
    def get_depths():
        depths = [3.5, 6.0, 9.0, 13.0, 16.5]
        #depths['m'] = list(map(lambda x: x*m_in_ft, depths['ft']))

        return depths

    def get_data(self,depth=3.5, interpolate=False):
        depths = self.get_depths()

        print(self.m_in_ft)


        dfs = []

        for depth in depths:
            query = " \
                SELECT \
                    StopDate, \
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

            df = pd.read_sql_query(query, self.db)
            df['StopDate'] = df['StopDate'].apply(pd.to_datetime)
            # TODO: Make sure the pivoting doesn't misrepresent any values
            df = pd.pivot_table(
                df,
                index='StopDate',
                columns='Location',
                values='Concentration',
                aggfunc=np.max,
            ).reset_index()

            df['Depth'] = np.repeat(depth*self.m_in_ft,len(df))

            if interpolate is False:
                continue
            elif interpolate is True:
                df = df.interpolate()
            else:
                raise ValueError('Interpolate option must be boolean.')

            dfs.append(df)
        # todo: figure out why it doesnt concatenate
        data = pd.concat(dfs)

        return data

class Kriging(Data):
    def __init__(self):
        data_processing = Data()

        #depths = data_processing.get_depths()
        data = data_processing.get_data() # TODO: Dynamically detect NaNs and update probe locations based on that...

        print(data.iloc[3:5])

        #probe_locations = self.get_probe_locations(data.iloc[3])
        #observations = self.assign_observations(data.iloc[1])
        #print(probe_locations)

        """
        x1, x2, grid = self.get_meshgrid()

        predictions = {}

        for i, time in enumerate(data.index):
            observations = data.iloc[i].values
            prediction = self.kriging(probe_locations, observations, grid)
            predictions[str(time.date())] = prediction


        print(grids)
        """
        #self.plot_results(x1, x2, predictions)
        return


    def load_probes_coordinates(self):
        probes = pd.read_csv('./data/indianapolis_probes.csv')
        return probes

    def get_probe_locations(self, data):

        # loads the probe coordinates
        probes = self.load_probes_coordinates()
        # list to store the active probe locations
        probe_locations = []

        for loc in data.dropna().index:
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

        from sklearn.gaussian_process.kernels import Matern

        # TODO: Lookup if specific values are use for geophysics or which kernel is best (might need to make a custom one)
        gpr = GaussianProcessRegressor(kernel=Matern()) # regressor function
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

    def plot_results(self, x1, x2, predictions):
        from matplotlib import animation
        times, plot_data = [], []

        for key, val in predictions.items():
            times.append(key)
            plot_data.append(val)

        vmin, vmax = 0, 30

        print(vmax)
        kw = dict(levels=np.linspace(vmin, vmax, 10), cmap='jet', vmin=vmin, vmax=vmax, origin='lower')

        fig, ax = plt.subplots()
        #ax = plt.axes(xlim=(0, x1.max()), ylim=(0, x2.max()))
        cont = ax.contourf(x1, x2, plot_data[0], **kw)

        cbar = plt.colorbar(cont)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        # animation function
        def animate(i):
            ax.clear()
            cont = ax.contourf(x1, x2, plot_data[i], **kw)
            ax.set_title(times[i])

            return cont


        FFwriter = animation.FFMpegWriter()
        anim = animation.FuncAnimation(fig, animate, interval=500)
        print(FFwriter.bin_path())
        anim.save('animation.mp4', writer=FFwriter)

        #plt.contourf(x1, x2, y_pred)
        #plt.show()


        return

Kriging()
