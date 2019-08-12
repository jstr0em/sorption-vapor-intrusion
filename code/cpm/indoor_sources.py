import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from estimate_kinetics import Kinetics


class IndoorSource(Kinetics):
    def __init__(self, df, material='concrete', contaminant='TCE', T=298, P=101325):
        super().__init__()
        self.df = df

        self.set_building_param()
        return

    def get_air_exchange_rate(self):
        return self.Ae

    def get_df(self):
        return self.df

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
        self.A_ck = 2 * (x + y) * w_ck  # crack area
        # surface area of the room
        self.A_room = 2 * (A_floor + A_wall_y + A_wall_x)
        self.Ae = Ae
        return

    def get_crack_area(self):
        return self.A_ck

    def get_t_data(self):
        df = self.get_df()
        return df['t'].values

    def get_entry_rate_data(self):
        df = self.get_df()
        if 'n_ck' in list(df):
            n_ck = df['n_ck'].values
        else:
            j_ck = df['j_ck'].values
            A_ck = self.get_crack_area()
            n_ck = j_ck * A_ck  # calculates molar entry rate mol/s
            n_ck *= 3600  # converts to mol/hr
        return n_ck

    def get_initial_concentration(self):
        df = self.get_df()
        return df['c_in'].values[0]

    def get_reaction_constants(self):
        M = self.get_molar_mass()
        k1, k2, K = super().get_reaction_constants()  # mol/hr

        # mol/hr -> g/hr
        k1 /= M
        k2 /= M

        # g/hr -> ug/hr
        k1 *= 1e6
        k2 *= 1e6
        return k1, k2, K

    def get_entry_rate(self):
        # gets time and entry rate data
        t = self.get_t_data()
        n_ck = self.get_entry_rate_data()
        # interpolation function
        func = interp1d(t, n_ck, bounds_error=False, fill_value=n_ck[0])
        return func

    def get_material_volume(self):
        material = self.get_material()

        A_room = self.get_room_area()
        penetration_depth = self.get_penetration_depth(material)
        return A_room * penetration_depth

    def get_penetration_depth(self, material):
        material = self.get_material()
        # depth to which contaminant has been adsorbed/penetrated into the material
        penetration_depth = {'concrete': 5e-3, 'wood': 1e-3,
                             'drywall': 1e-2, 'carpet': 1e-2, 'paper': 1e-4}
        return penetration_depth[material]

    def reaction(self, c_in, c_star, k1, k2):
        return k1 * c_star - k2 * c_in

    def cstr(self, u, t, Ae, V, V_mat, k1, k2):
        c_in = u[0]
        c_star = u[1]
        r = self.reaction(c_in, c_star, k1, k2)
        n = self.get_entry_rate()
        dc_in_dt = n(t) / V - Ae * c_in + r / V
        dc_star_dt = -r / V_mat

        return [dc_in_dt, dc_star_dt]

    def solve_cstr(self):

        k1, k2, K = self.get_reaction_constants()
        V_mat = self.get_material_volume()
        Ae = self.get_air_exchange_rate()
        V = self.get_building_volume()
        c0_in = self.get_initial_concentration()
        c0 = [c0_in, c0_in / K]

        t_data = self.get_t_data()

        t = np.linspace(0, t_data[-1], 200)

        result = odeint(func=self.cstr, y0=c0, t=t, args=(
            Ae, V, V_mat, k1, k2), mxstep=5000)
        c, c_star = result[:, 0], result[:, 1]
        return t, c, c_star


df = pd.read_csv('../../data/transient_sandy_loam.csv', header=4)
new_names2 = ('t', 'p_in', 'alpha', 'c_in', 'j_ck', 'm_ads', 'c_ads')

df.rename(columns=dict(zip(list(df), new_names2)), inplace=True)

x = IndoorSource(df)


fig, ax = plt.subplots(dpi=300)
t, c, c_star = x.solve_cstr()
ax.semilogy(t, c)
plt.show()

"""
# everything below here needs to be fixed
def reaction(c_in, c_star, k1, k2, K):
    k1 = K * k2
    r = k1 * c_star - k2 * c_in
    return r

def cstr(u, t, Ae, V, V_mat, k1, k2, K):
    c_in = u[0]
    c_star = u[1]
    r = reaction(c_in, c_star, k1, k2, K)
    dc_in_dt = n(t) / V - Ae * c_in + r / V
    dc_star_dt = -r / V_mat

    return [dc_in_dt, dc_star_dt]

# preprocess dfs here

transient = pd.read_csv('../../data/transient_results.csv', header=4)
p = pd.read_csv('../../data/p_cpm.csv', header=4)

p.rename(columns={'Pressure (Pa)': 'p_cpm'}, inplace=True)
new_names = ('t', 'c_in', 'c_ads', 'm_ads', 'alpha', 'j_sides', 'j_bottom', 'j_ck',
             'j_d_sides', 'j_d_bottom', 'j_d_ck', 'j_c_sides', 'j_c_bottom', 'j_c_ck', )

new_names2 = ('t', 'p_in', 'alpha', 'c_in', 'j_ck', 'm_ads', 'c_ads')

transient2 = pd.read_csv('../../data/transient_sandy_loam.csv', header=4)


soil_names = {'1': 'Sandy Clay', '2': 'Sand', '3': 'Loamy Sand', }

rename = dict(zip(list(transient), new_names))

transient.rename(columns=rename, inplace=True)
transient = pd.concat([transient, p['p_cpm']], axis=1)
transient['c_ads'] *= 131.38

transient2.rename(columns=dict(zip(list(transient2), new_names2)), inplace=True)

print(list(transient2))

Ae = 0.5
V = 300
M = 131.38
A_ck = 4*10*1e-2


transient2['j_ck'] *= 1e-6/M
transient2['c_in'] *= 1e-6/M




n = interp1d(transient2['t'].values, transient2['j_ck'].values *
             A_ck * 3600 * 1e2, fill_value=0, bounds_error=False)
c0_in = n(0) / V / Ae
transient2['c_in'] /= transient2['c_in'][0] / c0_in


# house properties
A_floor = 10 * 10  # 10x10 m surface
A_wall = 10 * 3  # 10 m wide, 3 m tall
A_room = A_floor * 2 + A_wall * 4
V_room = A_floor * 3
# ['Concrete', 'Drywall', 'Wood', 'Wool Carpet', 'Wall Paper']

materials = ['concrete', 'wood', 'drywall', 'carpet', 'paper']
penetration_depths = np.array([5e-3, 1e-3, 1e-2, 1e-2, 1e-4])
Vs = A_room * penetration_depths

rhos, k1s, k2s, Ks = [], [], [], []
for material in materials:
    kin = Kinetics(material=material)
    rho = kin.get_material_density()
    k1, k2, K = kin.get_reaction_constants()

    rhos.append(rho)
    k1s.append(k1)
    k2s.append(k2)
    Ks.append(K)

config = pd.DataFrame({'material': materials, 'volume': Vs,
                       'rho': rhos, 'k1': k1s, 'k2': k2s, 'K': Ks})


fig, (ax1, ax2, ax3) = plt.subplots(3, 1, dpi=300, sharex=True)
ax4 = ax3.twinx()

t = np.linspace(0, 72, 100)

for index, row in config.iterrows():
    material = row['material']
    V_mat = row['volume']
    k1 = row['k1']
    k2 = row['k2']
    K = row['K']

    c0 = [c0_in, c0_in / K]
    result = odeint(func=cstr, y0=c0, t=t, args=(Ae, V, V_mat, k1, k2, K))
    c_in = result[:, 0]
    c_star = result[:, 1]

    ax2.semilogy(t, c_in, label='%s' % material)
    ax3.semilogy(t, c_star * M * V_mat)
    ax4.semilogy(t, c_in * M * V, linestyle='--')

transient2.plot(x='t', y='c_in', ax=ax2, logy=True,
               legend=False, color='k', label='reference')

transient2.plot(x='t', y='m_ads', ax=ax3, color='k', legend=False)

table = config.set_index('material')
table['k1'] *= M
table['k2'] *= M
table = table.applymap("{0:1.2e}".format)

table.rename(columns={'volume': 'V ($\\mathrm{m^3}$)', 'k1': '$k_1$ ($\\mathrm{g/h}$)', 'k2': '$k_2$ ($\\mathrm{g/h}$)'}, inplace=True)
ax1.axis('off')
ax1.table(cellText=table.values, colWidths=[0.15] * len(table.columns),
          rowLabels=table.index,
          colLabels=table.columns,
          # cellLoc = 'center', rowLoc = 'center',
          loc='center')

ax2.set_ylabel('c_in (g/m^3)')
ax3.set_ylabel('m_star (g)')
ax4.set_ylabel('m_in (g)')
ax3.set_xlabel('time (h)')
ax2.legend()

plt.tight_layout()
plt.show()
"""
