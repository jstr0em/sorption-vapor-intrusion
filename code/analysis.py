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
from material import get_indoor_materials
from mitigation import Mitigation

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

    def generate_indoor_material_data(self):
        materials = get_indoor_materials()
        dfs = []

        for material in materials:
            print('Processing '+material)
            indoor = IndoorSource(material=material)
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


def get_sorption_mitigation_data():
    materials = get_indoor_materials()
    times = np.linspace(0,24,100)
    ts, c_ins, c_sorbs, mats = [], [], [], []


    for material in materials:

        mit = Mitigation(material=material)
        for time in times:
            c = mit.get_solution(time)

            ts.append(time)
            c_ins.append(c[0][0])
            c_sorbs.append(c[1][0])
            mats.append(material)


    df = pd.DataFrame(data={'time': ts, 'c_in': c_ins, 'c_sorbs': c_sorbs, 'material': mats})
    return df.set_index(['material', 'time'])


def get_sorption_mitigation_reduction_table():
    materials = get_indoor_materials()
    reductions = [0.5, 0.1, 0.01, 0.001]
    taus = []
    reds = [] # for storage
    mats = []

    for material in materials:
        x = Mitigation(material=material)
        for reduction in reductions:
            tau = float(x.get_reduction_time(reduction=reduction))
            taus.append(tau)
            reds.append(reduction)
            mats.append(material.title())

    df = pd.DataFrame(
        data={
            'Material': mats,
            'Reduction time (hr)': taus,
            'Reduction factor': reds,
            }
        )


    df.sort_values(
        by=['Reduction factor','Reduction time (hr)'],
        ascending=[False, True],
        inplace=True,
    )
    return df
