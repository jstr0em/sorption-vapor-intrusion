import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

df = pd.read_csv('~/Documents/COMSOL/temperature_study/collision_integral.csv', header=None)
I_D = interp1d(df[0],df[1])


# air
Vb2 = 29.9 # cm^3/mol
epskb1 = 97 # K
r2 = 3.617 # m
M2 = 28.97 # g/mol

rho1 = 1.46 # g/cm^3

M1 = 131.38 # g/mol
Vb1 = M1/rho1

Tc = 571 # K
epskb2 = 0.77*Tc # K
r1 = 1.18*Vb1**(1/3)

kB = 1.38e-6 # ergs/K


r12 = (r1+r2)/2

epskb12 = np.sqrt(epskb1*epskb2)
B = (10.85-2.50*np.sqrt(1/M1+1/M2))*1e-4
P = 1

T = np.linspace(298, 350)

D = B*T**(3/2)*np.sqrt(1/M1+1/M2)/(P*r12**2*I_D(T/epskb12))


T_amb = np.array([270.00,290.00, 310.00, 350.00])
c = np.array([9.0059E-5, 8.7346E-5, 8.4876E-5, 8.0480E-5])

plt.plot(T_amb/T_amb[0], c/c[0])

plt.show()
