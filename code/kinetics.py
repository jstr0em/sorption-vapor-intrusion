from experiment import Experiment
from material import Material
from contaminant import Contaminant
import numpy as np
import pandas as pd
from scipy.integrate import odeint, ode, solve_ivp
from scipy.optimize import curve_fit, leastsq
#from scipy.interpolate import interp1d

class Kinetics(Experiment, Material, Contaminant):
    def __init__(self, file='../data/adsorption_kinetics.csv', material='cinderblock', contaminant='TCE', T=298, P=101325):
        Experiment.__init__(self, file)
        Material.__init__(self, material)
        Contaminant.__init__(self, contaminant)
        self.contaminant = contaminant
        self.T = T
        self.P = P
        return

    def get_thermo_states(self):
        """ Method that returns the temperature and pressure of the system.

        Return:
            tuple: temperature (K), absolute pressure (Pa)
        """
        return self.T, self.P

    def get_gas_const(self):
        """ Returns ideal gas constant, R.
        """
        return 8.31446261815324

    def get_gas_conc(self, part_by_part=1.12e-9):
        """ Method that converts the air concentration from part by part to mol/m^3
        Args:
            (optional): Part-by-part of the contaminant

        Return:
            Air concentration (mol/m^3)
        """
        # gas constant
        R = self.get_gas_const()  # (J/(mol*K))
        T, P = self.get_thermo_states()
        M = self.get_molar_mass()
        return P * part_by_part / (R * T)

    def get_adsorbed_conc(self):
        """ Return:
            Moles of contaminant sorbed unto material (mol/m^3)
        """
        mass_by_mass = self.get_adsorption_data()
        rho = self.get_material_density()
        M = self.get_molar_mass()
        return mass_by_mass * rho / 1e9 / M

    def adsorption_kinetics(self, c_star, t, k1, k2):
        c = self.get_gas_conc()
        r = k1 * c_star - k2 * c
        return -r

    def get_time_data(self):
        """
        Return:
            Adsorption time data (hr)
        """
        material = self.get_material()
        data = self.get_data()
        data = data.loc[data['material'] == material]
        return np.append(0, data['time'].values) / 60

    def get_adsorption_data(self):
        material = self.get_material()
        data = self.get_data()
        data = data.loc[data['material'] == material]

        mass_by_mass = np.append(0, data['mass'].values)
        return mass_by_mass

    def solve_reaction(self, t, k1, k2):
        c_star = odeint(self.adsorption_kinetics, t=t,
                        y0=0, args=(k1, k2), mxstep=5000)
        return c_star.flatten()

    def get_reaction_constants(self):
        """
        Returns the fitted reaction constants

        arg:
            None
        return:
            k1 : desorption rate (1/hr)
            k2 : adsorption rate (1/hr)
            K : equilibrium constant (1)
        """
        t_data = self.get_time_data()
        c_star_data = self.get_adsorbed_conc()

        popt, pcov = curve_fit(self.solve_reaction, t_data,
                               c_star_data, p0=[1e-2, 1e2])

        k1, k2 = popt
        K = k1 / k2
        return k1, k2, K

    def get_isotherm(self):
        """
        Returns the linear adsorption isotherm

        args:
            None
        return:
            K_iso : adsorption isotherm (m^3/kg)
        """
        k1, k2, K = self.get_reaction_constants()
        rho = self.get_material_density()
        rho *= 1e-3  # converts to kg/m^3
        return 1 / (K * rho)
