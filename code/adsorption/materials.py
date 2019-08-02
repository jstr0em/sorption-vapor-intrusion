import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# state variables
T = 298 # K
R = 8.31446261815324 # gas constant
P = 101325 # 1 atm in Pa

# adsorption properties
c_ref = 1.12 # ppb
c_ref *= P/(R*T) # ng/m^3
c_ref *= 1e-9 # g/m^3
#c_star = mass_per_mass/rho

# house properties
A_floor = 10*10 # 10x10 m surface
A_wall = 10*3 # 10 m wide, 3 m tall
A_room = A_floor*2 + A_wall*4
V_room = A_floor*3

# ignoring dust for now
materials = ['Concrete', 'Drywall', 'Wood', 'Wool Carpet', 'Wall Paper']


penetration_depth = np.array([5e-3, 1e-2, 1e-3, 1e-2, 1e-4])

rho = np.array([2000, 7.62, 700, 1.3, 1.29]) # kg/m^3
rho *= 1e3 # g/m^3


capacity = np.array([115.8169e-9, 2e-9, 0.9874e-9, 2.43e-9, 15e-6]) # g adsorbate/g material
c_star = capacity*rho

K = c_ref/c_star

df = pd.DataFrame({'Material': materials, 'Capacity': capacity, 'rho': rho, 'K': K})

print(df)
