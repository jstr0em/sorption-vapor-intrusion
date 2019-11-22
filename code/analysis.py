import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint, ode, solve_ivp
from scipy.optimize import curve_fit, leastsq
from scipy.interpolate import interp1d

from experiment import Experiment
from data import Data
from contaminant import Contaminant
from material import Material
from comsol import COMSOL
from kinetics import Kinetics
from indoor_source import IndoorSource

class Analysis:
    def get_kinetics_data(self):
        materials = Material('cinderblock').get_materials()
        k1s, k2s, Ks = [], [], []

        for material in materials:
            kinetics = Kinetics(
                '../data/adsorption_kinetics.csv', material=material)
            k1, k2, K = kinetics.get_reaction_constants()
            k1s.append(k1)
            k2s.append(k2)
            Ks.append(K)

        data = pd.DataFrame(index=materials, data={
                            'k1': k1s, 'k2': k2s, 'K': Ks})
        return data

    def generate_indoor_material_data(self, zero_entry_rate=False):
        materials = Material('cinderblock').get_materials()[0:-1]
        materials.append('none')
        dfs = []

        for material in materials:
            indoor = IndoorSource(
                '../data/simulation/CPM_cycle_final.csv', material=material, zero_entry_rate=zero_entry_rate)
            if material != 'none':
                rxn = Kinetics(
                    '../data/adsorption_kinetics.csv', material=material)
                k1, k2, K = rxn.get_reaction_constants()
                indoor.set_reaction_constants(k1, k2, K)
                df = indoor.get_dataframe()
            else:
                df = indoor.get_dataframe()
            df.sort_values(by='time', inplace=True)
            df.reset_index(inplace=True, drop=True)
            df['material'] = np.repeat(material, len(df))
            dfs.append(df)

        df = pd.concat(dfs, axis=0, sort=False)
        return df

    def get_indoor_material_data(self,file='../data/simulation/indoor_material.csv'):
        #try:
            #df = pd.read_csv(file)
        #except:
        df = self.generate_indoor_material_data()
        df.to_csv(file,index=False)
        return df.set_index(['material', 'time'])

    def get_soil_sorption_data(self):
        data = COMSOL('../data/simulation/transient_parametric_sweep.csv')
        df = data.get_data()
        return df.set_index(['K_ads', 'time'])

    def get_steady_state_data(self):
        data = COMSOL(file='../data/simulation/parametric_sweep_final.csv').get_data()
        data['soil'].replace([1,2], ['Sandy Loam', 'Sand'], inplace=True)

        return data.set_index(['soil','K_ads', 'p_in'])


    def get_indoor_zero_entry_material_data(self,file='../data/simulation/indoor_material_zero_entry_rate.csv'):
        #try:
            #df = pd.read_csv(file)
        #except:
        df = self.generate_indoor_material_data(zero_entry_rate=True)
        df['sorption_balance'] = df['rxn']/(df['c_in']*0.5)
        df.to_csv(file,index=False)
        return df.set_index(['material', 'time'])


    def get_time_to_equilibrium_data(self):

        data = COMSOL(file='../data/simulation/time_to_equilibrium_final.csv').get_data()
        data['soil'] = np.repeat('Sandy Loam', len(data))
        sand_data = COMSOL(file='../data/simulation/sand_time_to_equilibrium_final.csv').get_data()
        sand_data['soil'] = np.repeat('Sand', len(sand_data))

        df = pd.concat([data, sand_data], axis=0, sort=False)

        df['p_cycle'].replace([2, 3], ['Depressurization', 'Overpressurization'], inplace=True)
        return df.set_index(['soil','K_ads','p_cycle', 'time'])
