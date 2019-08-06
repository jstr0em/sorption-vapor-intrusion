import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from estimate_kinetics import Kinetics

transient = pd.read_csv('../../data/transient_results.csv', header=4)
ss = pd.read_csv('../../data/steady_state_material_sweep.csv', header=4)

p = pd.read_csv('../../data/p_cpm.csv', header=4)

p.rename(columns={'Pressure (Pa)': 'p_cpm'}, inplace=True)
new_names = ('t', 'c_in', 'c_ads', 'm_ads', 'alpha', 'j_sides', 'j_bottom', 'j_ck',
             'j_d_sides', 'j_d_bottom', 'j_d_ck', 'j_c_sides', 'j_c_bottom', 'j_c_ck', )

new_names2 = ('soil', 'dp', 'c_in', 'c_ads', 'm_ads', 'alpha', 'j_sides', 'j_bottom',
              'j_ck', 'j_d_sides', 'j_d_bottom', 'j_d_ck', 'j_c_sides', 'j_c_bottom', 'j_c_ck', )

soil_names = {'1': 'Sandy Clay', '2': 'Sand', '3': 'Loamy Sand', }

rename = dict(zip(list(transient), new_names))

transient.rename(columns=rename, inplace=True)
ss.rename(columns=dict(zip(list(ss), new_names2)), inplace=True)
transient = pd.concat([transient, p['p_cpm']], axis=1)
transient['c_ads'] *= 131.38


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


# parameters
Ae = 0.5
V = 300
M = 131.38  # g/mol TCE
A_ck = 4 * 10 * 1e-2

n = interp1d(transient['t'].values, transient['j_ck'].values *
             A_ck * 3600 * 1e2, fill_value=0, bounds_error=False)
c0_in = n(0) / V / Ae
transient['c_in'] /= transient['c_in'][0] / c0_in


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

transient.plot(x='t', y='c_in', ax=ax2, logy=True,
               legend=False, color='k', label='reference')


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
