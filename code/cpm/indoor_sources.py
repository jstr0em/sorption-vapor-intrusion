import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint

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

K = 10
k2 = 1e-2
k1 = K*k2

def r(c_in, c_star, K, k2):
    r = k1*c_star-k2*c_in
    return r

def cstr(u, t, Ae, V, K, k2):
    c_in = u[0]
    c_star = u[1]

    dc_in_dt = n(t)/V - Ae*c_in + r(c_in, c_star, K, k2)
    dc_star_dt = -r(c_in, c_star, K, k2)/V

    return [dc_in_dt, dc_star_dt]
import scipy.integrate

tmax = 10.0

def a(t):
    if t < tmax / 2.0:
        return ((tmax / 2.0) - t) / (tmax / 2.0)
    else:
        return 1.0

def func(x, t, a):
    return - (x - a(t))

x0 = 0.8
t = np.linspace(0.0, tmax, 1000)
args = (a,)
y = odeint(func, x0, t, args)
