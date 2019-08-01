import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import leastsq

ppb = 1.12 # ppb
t_data = np.array([0, 310, 1080, 2614])
mass_per_mass = np.array([0, 26.3947, 30.2191, 33.1549]) # ng/g

M = 131.38 # g/mol
rho = 1680e3 # g/m^3

# state variables
T = 298 # K
R = 8.31446261815324 # gas constant
P = 101325 # 1 atm in Pa

c_star = mass_per_mass/rho
c_air = P*ppb/R/T # ng/m^3

def ode(y, t, k1, k2):
    return k1*c_air-k2*y

def solve_ode(t, args):
    y0 = 0

    f = lambda y, t: ode(y, t, args[0], args[1])
    r = odeint(f, t, y0)
    return r[0]

def f_resid(args):

    r = solve_ode(t, args)
    resid = c_star - r
    print(resid)
    return resid

t = t_data

guess = (1e2,1e-2) #initial guess for params
(c,kvg) = leastsq(f_resid, guess) #get para
print(c)
