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

    def set_organic_content(self):

        material = self.get_material()
        if material != 'soil':
            RuntimeError()

    def get_thermo_states(self):
        return self.T, self.P

    def get_material(self):
        return self.material

    def get_contaminant(self):
        return self.contaminant

    def get_gas_conc(self):
        # gas constant
        R = 8.31446261815324  # (J/(mol*K))
        T, P = self.get_thermo_states()
        part_by_part = 1.12e-9  # parts of TCE per part of air V_TCE/V_total
        c = P * part_by_part / (R * T)  # concentration of TCE (g/m^3)
        return c

    def get_adsorbed_conc(self):
        mass_by_mass = self.get_adsorption_data()
        rho = self.get_material_density()
        M = self.get_molar_mass()
        c_star = mass_by_mass * rho / 1e9 / M  # g/m3
        return c_star

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

    def get_time_data(self):
        path = self.get_data_path()
        material = self.get_material()

        data = pd.read_csv(path)
        data = data.loc[data['material'] == material]

        time = np.append(0, data['time'].values)
        return time

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

    def get_reaction_constants(self):
        t_data = self.get_time_data()
        c_star_data = self.get_adsorbed_conc()

        popt, pcov = curve_fit(self.solve_reaction, t_data, c_star_data, p0=[1e-2, 1e-4])

        k1, k2 = popt
        K = k1 / k2
        return k1, k2, K



    def plot(self,save=False):

        t_data = self.get_time_data()
        c_star_data = self.get_adsorbed_conc()

        k1, k2, K = self.get_reaction_constants()



        t = np.linspace(t_data[0], t_data[-1], 200)
        c_star = self.solve_reaction(t, k1, k2)

        fig, ax = plt.subplots(dpi=300)

        plt.plot(t_data, c_star_data, 'o', label='Data')
        plt.plot(t, c_star, label='Fit')

        plt.show()

        return

soil = Kinetics(material='soil')
#soil.plot()



k1, k2, K = soil.get_reaction_constants()
f = 0.007 # 0.7%  organic content
rho = soil.get_material_density() # g/m3
rho /= 1e3 # kg/m3

c = soil.get_gas_conc() # g/m3 in air
c_star = c*K
print(c, c_star)

K_d = 1/(K)*rho*1e3 # converts to L gas/kg solid

K_oc = K_d/f

print(K_d, K_oc)

K_d2 = c_star*rho/c*1e3
K_oc2 = K_d2/f

print(K_d2, K_oc2)
