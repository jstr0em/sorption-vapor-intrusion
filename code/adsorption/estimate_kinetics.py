import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import curve_fit

ppb = 1.12 # ppb
t_data = np.array([0, 310, 1080, 2614])
t_data = t_data/3600
mass_per_mass = np.array([0, 26.3947, 30.2191, 33.1549]) # ng/g

M = 131.38 # g/mol
rho = 1380e3 # g/m^3

# state variables
T = 298 # K
R = 8.31446261815324 # gas constant
P = 101325 # Pa/atm

c_star = mass_per_mass*rho
c_star /= 1e6 # mg/m^3
c_air = P*ppb/(R*T) # ng/m^3
#c_air *= 1e6 # mg/m^3

def ode(y, t, k1, k2):
    #print(k2*c_air, k1*y)
    return k2*c_air-k1*y

def solve_ode(t, k1, k2):
    y0 = 0

    #f = lambda y, t: ode(y, t, k1, k2)
    r = odeint(ode, t, y0)
    return r[0]

#t = t_data
#popt = curve_fit(solve_ode, xdata=t_data, ydata=c_star)


t = np.linspace(0,t_data[-1], 10)

fig, ax = plt.subplots(dpi=300)

ax.plot(t_data, c_star, 'o')
ax.plot(t, solve_ode(t, 1e-12, 1e4))

plt.show()


print(c_air, c_star)
