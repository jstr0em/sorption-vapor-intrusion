import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint, ode, solve_ivp
from scipy.optimize import curve_fit, leastsq
from scipy.interpolate import interp1d

from comsol import COMSOL
from indoor_material import IndoorMaterial
from kinetics import Kinetics

class IndoorSource(COMSOL, IndoorMaterial, Kinetics):
    def __init__(self, file='../data/simulation/CPM_cycle_final.csv', material='cinderblock', contaminant='TCE'):
        COMSOL.__init__(self, file=file)
        IndoorMaterial.__init__(self,material=material)
        Kinetics.__init__(self, material=material, contaminant=contaminant)
        self.set_entry_rate()

        return

    def get_initial_concentration(self):
        """Returns the initial indoor air and sorbed concentrations."""
        Ae = self.get_air_exchange_rate()
        V = self.get_building_volume()
        M = self.get_molar_mass()
        k1, k2, K = self.get_reaction_constants()

        df = self.get_data()

        c0_in = df['c_in'].values[0]/M/1e6

        if K == 0:
            c0_sorb = 0
        else:
            c0_sorb = c0_in/K

        return [c0_in, c0_sorb]

    def set_entry_rate(self):
        """Sets an interpolation function for the contaminant entry rate."""
        # gets time and entry rate data
        t = self.get_time_data()

        try:
            n_ck = self.get_entry_rate_data() * 1e-6 / M
        except:
            j_ck = self.get_entry_flux_data()
            M = self.get_molar_mass()
            A_ck = self.get_crack_area()
            n_ck = j_ck * 1e-6 / M * 3600 * A_ck
        # interpolation function
        func = interp1d(t, n_ck, bounds_error=False, fill_value=n_ck[-1])
        self.n_ck = func
        return

    def get_entry_rate(self):
        return self.n_ck

    def reaction(self, c_in, c_star, k1, k2):
        return k1 * c_star - k2 * c_in

    def cstr(self, t, u):

        # gets parameters
        Ae = self.get_air_exchange_rate()
        V_bldg = self.get_building_volume()
        V_mat = self.get_material_volume()
        k1, k2, K = self.get_reaction_constants()

        # loads variables
        c_in = u[0]
        c_star = u[1]

        # assigns reaction and entry rate functions
        r = self.reaction(c_in, c_star, k1, k2)
        n = self.get_entry_rate()

        # odes
        dc_in_dt = n(t) / V - Ae * c_in + r * V_mat / V_bldg
        dc_star_dt = -r

        return [dc_in_dt, dc_star_dt]

    def solve_cstr(self):
        t = self.get_time_data()
        c0s = self.get_initial_concentrations()
        k1, k2, K = self.get_reaction_constants()

        r = solve_ivp(self.cstr, t_span=(t[0], t[-1]), y0=c0s, method='Radau',
                      t_eval=t, max_step=0.1)
        t, c, c_star = r['t'], r['y'][0], r['y'][1]
        rxn = self.reaction(c, c_star, k1, k2)
        return t, c, c_star, rxn

    def get_dataframe(self):
        df = self.get_data()
        t, c, c_star, rxn = self.solve_cstr()
        M = self.get_molar_mass()
        c_gw = self.get_groudwater_concentration()
        df['c_in'] = c * M * 1e6 # ug/m^3
        df['c_mat'] = c_star * M * 1e6 # ug/m^3
        df['alpha'] = df['c_in']/c_gw
        df['rxn'] = rxn * M * 1e6 # ug/hr
        df['V_bldg'] = np.repeat(self.get_building_volume() , len(df))
        df['V_mat'] = np.repeat(self.get_material_volume() , len(df))
        return df
