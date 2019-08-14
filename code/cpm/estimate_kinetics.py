import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import curve_fit, leastsq


class Kinetics:

    def __init__(self, material='concrete', contaminant='TCE', T=298, P=101325):
        self.material = material
        self.contaminant = contaminant
        self.T = T
        self.P = P
        self.path = '../../data/adsorption_kinetics.csv'
        return

    """
    Method that returns the temperature and pressure of the system.

    Return:
        tuple: temperature (K), absolute pressure (Pa)
    """
    def get_thermo_states(self):
        return self.T, self.P

    def get_material(self):
        return self.material

    def get_gas_const(self):
        return 8.31446261815324

    def get_contaminant(self):
        return self.contaminant
    """
    Method that converts the air concentration from part by part to mol/m^3

    Args:
        (optional): Part-by-part of the contaminant

    Return:
        Air concentration (mol/m^3)
    """
    def get_gas_conc(self, part_by_part = 1.12e-9):
        # gas constant
        R = self.get_gas_const()  # (J/(mol*K))
        T, P = self.get_thermo_states()
        M = self.get_molar_mass()
        return P * part_by_part / (R * T)

    """
    Return:
        Moles of contaminant sorbed unto material (mol/m^3)
    """
    def get_adsorbed_conc(self):
        mass_by_mass = self.get_adsorption_data()
        rho = self.get_material_density()
        M = self.get_molar_mass()
        return mass_by_mass * rho / 1e9 / M

    def get_molar_mass(self):
        contaminant = self.get_contaminant()
        # molar masses are in (g/mol)
        M = {'TCE': 131.38}
        return M[contaminant]

    def get_data_path(self):
        return self.path

    def get_material_density(self):
        material = self.get_material()
        # densities in g/m^3
        density = {'drywall': 0.6e6, 'concrete': 2.0e6,
                   'carpet': 1.314e6, 'wood': 0.86e6, 'paper': 0.8e6, 'soil': 1.46e6}
        # soil is based on sandy loam data
        return density[material]

    def rate_equation(self, c_star, t, k1, k2):
        c = self.get_gas_conc()
        r = k1 * c_star - k2 * c
        return -r

    def set_data_path(self, path):
        self.path = path
        return

    """
    Return:
        Adsorption time data (hr)
    """
    def get_time_data(self):
        path = self.get_data_path()
        material = self.get_material()

        data = pd.read_csv(path)
        data = data.loc[data['material'] == material]
        return np.append(0, data['time'].values)/60

    def get_adsorption_data(self):
        path = self.get_data_path()
        material = self.get_material()

        data = pd.read_csv(path)
        data = data.loc[data['material'] == material]

        mass_by_mass = np.append(0, data['mass'].values)
        return mass_by_mass

    def solve_reaction(self, t, k1, k2):
        c_star = odeint(self.rate_equation, t=t, y0=0, args=(k1, k2), mxstep=5000)
        return c_star.flatten()

    """
    Returns the fitted reaction constants

    arg:
        None
    return:
        k1 : desorption rate (mol/hr)
        k2 : adsorption rate (mol/hr)
        K : equilibrium constant (1)
    """
    def get_reaction_constants(self):
        t_data = self.get_time_data()
        c_star_data = self.get_adsorbed_conc()

        popt, pcov = curve_fit(self.solve_reaction, t_data, c_star_data, p0=[1e-2, 1e2])

        k1, k2 = popt
        K = k1 / k2
        return k1, k2, K

    """
    Returns the linear adsorption isotherm

    args:
        None
    return:
        K_iso : adsorption isotherm (m^3/kg)
    """
    def get_isotherm(self):
        k1, k2, K = self.get_reaction_constants()
        rho = self.get_material_density()
        rho *= 1e-3 # converts to kg/m^3
        print(rho)
        K_iso = 1/(K*rho)

        return K_iso


    def plot(self,save=False):

        t_data = self.get_time_data()
        c_star_data = self.get_adsorbed_conc()

        k1, k2, K = self.get_reaction_constants()
        M = self.get_molar_mass()
        rho = self.get_material_density()


        t = np.linspace(t_data[0], t_data[-1], 200)
        c_star = self.solve_reaction(t, k1, k2)

        fig, ax = plt.subplots(dpi=300)

        ax.plot(t_data*60, c_star_data/rho*M*1e9, 'o', label='Data')
        ax.plot(t*60, c_star/rho*M*1e9, label='Fit')

        ax.set_title('$k_1$ = %1.2e, $k_2$ = %1.2e, $K$ = %1.2e' % (k1, k2, K))
        plt.legend()
        plt.show()

        return


soil = Kinetics(material='soil')

M = soil.get_molar_mass()
print(soil.get_gas_conc()*1e6*M)
#soil.plot()
#print(soil.get_isotherm())
