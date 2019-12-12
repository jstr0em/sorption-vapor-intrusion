import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint, ode, solve_ivp
from scipy.optimize import curve_fit, leastsq
from scipy.interpolate import interp1d

from comsol import COMSOL
from material import Material
from contaminant import Contaminant

class IndoorSource(COMSOL, Material, Contaminant):
    def __init__(self, file, material='cinderblock', contaminant='TCE', zero_entry_rate=False):
        COMSOL.__init__(self, file)
        Contaminant.__init__(self, contaminant)
        self.zero_entry_rate = zero_entry_rate
        self.set_building_param()
        self.set_entry_rate()
        self.set_groundwater_concentration()

        #if material is 'none':
            #self.material = None
        #else:
        Material.__init__(self, material)
        self.set_material_volume()
        return

    def get_groudwater_concentration(self):
        return self.c_gw
    def set_groundwater_concentration(self):
        df = self.get_data()
        self.c_gw = df['c_in'].values[0]/df['alpha'].values[0]
        return

    def get_zero_entry_rate(self):
        return self.zero_entry_rate
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

        Ae = self.get_air_exchange_rate()
        V = self.get_building_volume()
        M = self.get_molar_mass()

        df = self.get_data()

        return df['c_in'].values[0]/M/1e6

    def set_entry_rate(self):
        # gets time and entry rate data
        t = self.get_time_data()

        if self.get_zero_entry_rate():
            self.n_ck = interp1d(t, np.repeat(0, len(t)), bounds_error=False, fill_value=0)
            return

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
        penetration_depth = {'cinderblock': 5e-3, 'wood': 1e-3,
                             'drywall': 1e-2, 'carpet': 1e-2, 'paper': 1e-4, 'none': 0}
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
        dc_in_dt = n(t) / V - Ae * c_in + r * V_mat / V
        dc_star_dt = -r

        return [dc_in_dt, dc_star_dt]

    def cstr_no_rxn(self, t, u):

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
        t = self.get_time_data()
        c0_in = self.get_initial_concentration()

        if self.get_material() == None:
            r = solve_ivp(self.cstr_no_rxn, t_span=(
                t[0], t[-1]), y0=[c0_in], method='Radau', t_eval=t, max_step=0.1)
            t, c, c_star = r['t'], r['y'][0], np.repeat(0, len(r['y'][0]))

            return t, c, c_star, np.repeat(0, len(t))
        else:
            k1, k2, K = self.get_reaction_constants()
            c0 = [c0_in, c0_in / K]
            r = solve_ivp(self.cstr, t_span=(t[0], t[-1]), y0=c0, method='Radau',
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
