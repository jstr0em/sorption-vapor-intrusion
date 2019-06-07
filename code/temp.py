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

X = 2.6
M_water = 18.01 # g/mol

# needs to return centipoise
def mu(T):
    mu = 1.3799566804-0.021224019151*T**1+1.3604562827E-4*T**2-4.6454090319E-7*T**3+8.9042735735E-10*T**4-9.0790692686E-13*T**5+3.8457331488E-16*T**6
    return mu*1e3

D_water = 7.4e-8*(X*M_water)**0.5/Vb1**0.6*T/mu(T)
D_water *= 1e-4

print(D_water)
