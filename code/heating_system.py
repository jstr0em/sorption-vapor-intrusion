import numpy as np
import matplotlib.pyplot as plt
import cantera as ct
import pandas as pd

class Furnace:
    def __init__(self,fuel='CH4',type='Forced-air'):

        self.fuel = fuel

        return

    def set_combustor(self):
        inlet = ct.Reservoir(gas)
        combustor = ct.IdealGasReactor(gas)

        return
    def mdot(self,t):
        return combustor.mass / residence_time


    def get_fuel_mix():
        gas = ct.Solution('gri30.xml')
        gas.equilibrate('HP')
        gas.TP = 300.0, ct.one_atm
        equiv_ratio = 0.5  # lean combustion
        gas.set_equivalence_ratio(equiv_ratio, 'CH4:1.0', 'O2:1.0, N2:3.76')

        return gas

    def get_inlet_outlet(self,gas):
        exhaust = ct.Reservoir(gas)
        inlet = ct.Reservoir(gas)

        inlet_mfc = ct.MassFlowController(inlet, combustor, mdot=mdot)
        outlet_mfc = ct.PressureController(combustor, exhaust, master=inlet_mfc, K=0.01)

        return inlet, exhaust

    def get_reactor_network(self):

        sim = ct.ReactorNet([combustor])
        return

    def heat(energy):

        return Q


df = pd.read_csv('../data/indianapolis.csv')
df = df.loc[df['Side']=='Heated']
df['Time'] = df['Time'].apply(pd.to_datetime)

df['dT'] = df['IndoorTemp'] - df['OutdoorTemp']
df['dT_heating'] = df['IndoorTemp'].diff()
df[df['dT_heating'] < 0] = 0
print(df['dT_heating'])

fig, ax = plt.subplots(dpi=300)

Cp = 1.006 # kJ/(kg*K)
V = 300 # m^3
rho = 1.276 # kg/m^3
M = 16.043 # g/mol of CH4
dH = -891.1 # kJ/mol
R = ct.gas_constant/1e3
print(R)
df['E'] = Cp*V*rho*df['dT_heating'] # kJ needed
df['n'] = df['E']/dH # mol of CH4 burnt
df['m'] = df['n']*M # g of CH4 burnt
df['dP_heating'] = 3*df['n']*R*df['IndoorTemp']/V


df.plot(
    x='Time',
    y=['IndoorOutdoorPressure','dP'],
    ax=ax,
)

plt.show()
