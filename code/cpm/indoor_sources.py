import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint
from scipy.interpolate import interp1d

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


def r(c_in, c_star, K, k2):
    k1 = K*k2
    r = k1*c_star-k2*c_in
    return r

def cstr(u, t, Ae, V, K, k2):
    c_in = u[0]
    c_star = u[1]
    dc_in_dt = n(t)/V - Ae*c_in + r(c_in, c_star, K, k2)/V
    dc_star_dt = -r(c_in, c_star, K, k2)/V

    return [dc_in_dt, dc_star_dt]

# parameters
Ae = 0.5
V = 300
K = 1e-2
k2 = 1e-4

A_ck = 4*10*1e-2

n = interp1d(transient['t'].values, transient['j_ck'].values*A_ck*3600*1e2, fill_value=0, bounds_error=False)
c0_in = n(0)/V/Ae
transient['c_in'] /= transient['c_in'][0]/c0_in

t = np.linspace(0, 72, 100)


for K in [1e-2, 1, 1e2]:

    fig, (ax1, ax2) = plt.subplots(2,1,dpi=300)
    for k2 in (1e-4, 1e-2, 1, 1e2, 1e4):
        k1 = K*k2
        c0 = [c0_in, c0_in/K]
        c = odeint(func=cstr, y0=c0, t=t, args=(Ae, V, K, k2,))
        c_in = c[:,0]
        c_star = c[:,1]

        ax1.semilogy(t,c_in,label='k1 = %1.0e, k2 = %1.0e' % (k1, k2))
        ax2.semilogy(t,c_star,label='k2 = %1.0e' % k2)

    transient.plot(x='t', y='c_in', ax=ax1, logy=True, legend=False, color='k', label='reference')
    ax1.set_title('K = %1.0e' % K)
    ax1.legend()
plt.show()
