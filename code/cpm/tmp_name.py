import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import curve_fit, leastsq

class Data:
    def __init__(self, file):
        self.set_path(file)
        return
    def set_path(self, path):
        self.path = path
        return
    def get_path(self):
        return self.path
    def get_data(self):
        return self.data
"""
Class to load, process, and return the COMSOL simulation data
"""
class COMSOL(Data):
    def __init__(self, file):
        Data.__init__(self,file)
        self.process_raw_data()
        return

    def get_raw_data(self):
        path = self.get_path()
        return pd.read_csv(path, header=4)

    def get_renaming_scheme(self):
        renaming = {'% Time (h)': 'time','p_in (Pa)': 'p','alpha (1)': 'alpha',
        'c_in (ug/m^3)': 'c','j_ck (ug/(m^2*s))': 'j_ck','m_ads (g)': 'm_ads','c_ads (mol/kg)': 'c_ads'}
        return renaming

    def process_raw_data(self):
        raw_df = self.get_raw_data()
        self.data = raw_df.rename(columns=self.get_renaming_scheme())
        return


class Experiment(Data):
    def __init__(self,file):
        Data.__init__(self,file)
        self.get_raw_data()
        return

    def get_raw_data(self):
        path = self.get_path()
        self.data = pd.read_csv(path)
        return

x = COMSOL('../../data/transient_sandy_loam.csv')
y = Experiment('../../data/adsorption_kinetics.csv')
print(y.get_data())
