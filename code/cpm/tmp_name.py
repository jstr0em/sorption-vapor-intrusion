import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint, ode
from scipy.optimize import curve_fit, leastsq
from scipy.interpolate import interp1d

class Data:
    def __init__(self, file):
        self.path = file
        return
    def get_path(self):
        return self.path
    def get_data(self):
        return self.data

class Contaminant:
    def __init__(self, contaminant):

        self.contaminant = contaminant
        self.set_molar_mass()
        return

    def set_molar_mass(self):
        contaminant = self.get_contaminant()
        # molar masses are in (g/mol)
        M = {'TCE': 131.38}
        self.M = M[contaminant]
        return

    def get_contaminant(self):
        return self.contaminant

    def get_molar_mass(self):
        return self.M

class Material:
    def __init__(self, material):
        self.material = material
        self.set_material_density()
        return

    def get_material(self):
        return self.material
    def set_material_density(self):
        material = self.get_material()
        # densities in g/m^3
        density = {'drywall': 0.6e6, 'concrete': 2.0e6,
                   'carpet': 1.314e6, 'wood': 0.86e6, 'paper': 0.8e6, 'soil': 1.46e6}
        # soil is based on sandy loam data
        self.rho = density[material]
        return

    def get_material_density(self):
        return self.rho

    def get_materials(self):
        return ['drywall', 'concrete', 'carpet', 'wood', 'paper', 'soil']

class COMSOL(Data):
    """
    Class to load, process, and return the COMSOL simulation data
    """
    def __init__(self, file):
        Data.__init__(self,file)
        self.process_raw_data()
        self.A_ck = 0.1*4
        return

    def get_crack_area(self):
        return self.A_ck

    def get_raw_data(self):
        path = self.get_path()
        return pd.read_csv(path, header=4)

    def get_renaming_scheme(self):
        renaming = {'% Time (h)': 'time','p_in (Pa)': 'p','alpha (1)': 'alpha',
        'c_in (ug/m^3)': 'c_in','j_ck (ug/(m^2*s))': 'j_ck','m_ads (g)': 'm_ads','c_ads (mol/kg)': 'c_ads'}
        return renaming

    def process_raw_data(self):
        raw_df = self.get_raw_data()
        self.data = raw_df.rename(columns=self.get_renaming_scheme())
        return

    def get_time_data(self):
        df = self.get_data()
        return df['time'].values

    def get_entry_flux_data(self):
        df = self.get_data()
        return df['j_ck'].values

    def get_concentration_data(self):
        df = self.get_data()
        return df['c_in'].values

class Experiment(Data):
    def __init__(self,file):
        Data.__init__(self,file)
        self.set_raw_data()
        return

    def set_raw_data(self):
        path = self.get_path()
        self.data = pd.read_csv(path)
        return

class Kinetics(Experiment, Material, Contaminant):
    def __init__(self, file, material='concrete', contaminant='TCE', T=298, P=101325):
        Experiment.__init__(self, file)
        Material.__init__(self, material)
        Contaminant.__init__(self, contaminant)
        self.contaminant = contaminant
        self.T = T
        self.P = P
        return

    def get_thermo_states(self):
        """
        Method that returns the temperature and pressure of the system.

        Return:
            tuple: temperature (K), absolute pressure (Pa)
        """
        return self.T, self.P
    def get_gas_const(self):
        return 8.31446261815324

    def get_gas_conc(self, part_by_part = 1.12e-9):
        """
        Method that converts the air concentration from part by part to mol/m^3

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
        """
        Return:
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
        return np.append(0, data['time'].values)/60

    def get_adsorption_data(self):
        material = self.get_material()
        data = self.get_data()
        data = data.loc[data['material'] == material]

        mass_by_mass = np.append(0, data['mass'].values)
        return mass_by_mass

    def solve_reaction(self, t, k1, k2):
        c_star = odeint(self.adsorption_kinetics, t=t, y0=0, args=(k1, k2), mxstep=5000)
        return c_star.flatten()

    def get_reaction_constants(self):
        """
        Returns the fitted reaction constants

        arg:
            None
        return:
            k1 : desorption rate (mol/hr)
            k2 : adsorption rate (mol/hr)
            K : equilibrium constant (1)
        """
        t_data = self.get_time_data()
        c_star_data = self.get_adsorbed_conc()

        popt, pcov = curve_fit(self.solve_reaction, t_data, c_star_data, p0=[1e-2, 1e2])

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
        rho *= 1e-3 # converts to kg/m^3
        return 1/(K*rho)


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


class IndoorSource(COMSOL, Material, Contaminant):
    def __init__(self, file, material='concrete', contaminant='TCE'):
        COMSOL.__init__(self, file)
        Contaminant.__init__(self, contaminant)
        self.set_building_param()
        self.set_entry_rate()

        if material != None:
            Material.__init__(self, material)
            self.set_material_volume()
        else:
            self.material = None

        return

    def get_air_exchange_rate(self):
        return self.Ae
    def get_building_volume(self):
        return self.V
    def get_room_area(self):
        return self.A_room

    def set_building_param(self, Ae=0.5, w_ck=1e-2, xyz=(10, 10, 3)):
        x, y, z = xyz  # x, y, z, dimensions

        A_floor = x * y  # area of the floor/ceilung
        A_wall_y = y * z  # area of one type of wall
        A_wall_x = x * z  # area of the other type of wall

        # assigns parameters to class
        self.V = x * y * z  # building/control volume
        # surface area of the room
        self.A_room = 2 * (A_floor + A_wall_y + A_wall_x)
        self.Ae = Ae
        return

    def get_initial_concentration(self):
        n = self.get_entry_rate()
        Ae = self.get_air_exchange_rate()
        V = self.get_building_volume()
        return n(0)/(Ae*V)

    def set_entry_rate(self):
        # gets time and entry rate data
        t = self.get_time_data()
        j_ck = self.get_entry_flux_data()
        M = self.get_molar_mass()
        A_ck = self.get_crack_area()
        n_ck = j_ck*1e-6/M*3600*A_ck
        # interpolation function
        func = interp1d(t, n_ck, bounds_error=False, fill_value=n_ck[-1])
        self.n_ck = func
        return

    def get_entry_rate(self):
        return self.n_ck

    def set_material_volume(self):
        material = self.get_material()
        A_room = self.get_room_area()
        penetration_depth = self.get_penetration_depth(material)
        self.V_mat = A_room * penetration_depth
        return

    def get_material_volume(self):
        return self.V_mat

    def get_penetration_depth(self, material):
        material = self.get_material()
        # depth to which contaminant has been adsorbed/penetrated into the material
        penetration_depth = {'concrete': 5e-3, 'wood': 1e-3,
                             'drywall': 1e-2, 'carpet': 1e-2, 'paper': 1e-4}
        return penetration_depth[material]

    def reaction(self, c_in, c_star, k1, k2):
        return k1 * c_star - k2 * c_in


    def set_reaction_constants(self, k1, k2, K):
        self.k1 = k1
        self.k2 = k2
        self.K = K
        return

    def get_reaction_constants(self):
        return self.k1, self.k2, self.K

    def cstr(self, t, u):

        # gets parameters
        Ae = self.get_air_exchange_rate()
        V = self.get_building_volume()
        V_mat = self.get_material_volume()
        k1, k2, K = self.get_reaction_constants()

        # loads variables
        c_in = u[0]
        c_star = u[1]

        # assigns reaction and entry rate functions
        r = self.reaction(c_in, c_star, k1, k2)
        n = self.get_entry_rate()

        # odes
        dc_in_dt = n(t) / V - Ae * c_in + r / V
        dc_star_dt = -r / V_mat

        return [dc_in_dt, dc_star_dt]

    def cstr_no_indoor_source(self, t, u):

        # gets parameters
        Ae = self.get_air_exchange_rate()
        V = self.get_building_volume()

        # loads variable
        c_in = u

        # assigns entry rate function
        n = self.get_entry_rate()

        # ode
        dc_in_dt = n(t) / V - Ae * c_in
        return dc_in_dt

    def solve_cstr(self):

        c0_in = self.get_initial_concentration()

        t_data = self.get_time_data()
        t0, t_end = t_data[0], t_data[-1]
        t = []
        c = []


        if self.get_material() != None:
            k1, k2, K = self.get_reaction_constants()
            c0 = [c0_in, c0_in / K]
            c_star = []
            r = ode(self.cstr).set_integrator('lsoda', method='bdf', nsteps=5000, max_step=0.2)
            r.set_initial_value(c0, t0)
            dt = 1
            while r.successful() and r.t < t_end:
                t.append(r.t)
                c.append(r.y[0])
                c_star.append(r.y[1])

                r.integrate(r.t+dt)
            return np.array(t), np.array(c), np.array(c_star)
        else:
            c0 = c0_in
            r = ode(self.cstr_no_indoor_source).set_integrator('lsoda', method='bdf', nsteps=5000, max_step=0.2)
            r.set_initial_value(c0, t0)
            dt = 1
            while r.successful() and r.t < t_end:
                t.append(r.t)
                c.append(r.y)

                r.integrate(r.t+dt)
            return np.array(t), np.array(c)



rxn = Kinetics('../../data/adsorption_kinetics.csv', material='soil')
k1, k2, K = rxn.get_reaction_constants()
indoor = IndoorSource('../../data/transient_results_shuai_isotherm_good_mesh.csv', material='concrete')
indoor.set_reaction_constants(k1, k2, K)
t, c, c_star = indoor.solve_cstr()


plt.plot(t, c)

plt.show()
