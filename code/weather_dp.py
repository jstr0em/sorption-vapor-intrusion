import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import cantera as ct

def dP_T(dT):

    alpha = 0.040 # Pa/(m*K)
    dz = -5 # basement floor depth from surface m
    return alpha*dz*dT


def dP_wind(u, dirs):
    neg_dirs = ['N', 'NE', 'E', 'SE', ] # wind directions that cause a negative pressure differential
    Cj = 0.3 # surface pressure coefficient
    rho = 1 # air density kg/m^3

    signs = []
    for dir in dirs:
        if dir in neg_dirs:
            signs.append(-1)
        else:
            signs.append(1)
    signs = np.array(signs)
    return 0.5*Cj*rho*signs*u**2


df = pd.read_csv('../data/indianapolis.csv')
df = df.loc[df['Side']=='Heated']
df['Time'] = df['Time'].apply(pd.to_datetime)

def get_wind_direction(degree):
    degs = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    dir = min(degs, key=lambda x:abs(x-degree))

    dirs = {
        '0': 'N',
        '45': 'NE',
        '90': 'E',
        '135': 'SE',
        '180': 'S',
        '225': 'SW',
        '270': 'W',
        '315': 'NW',
        '360': 'N',
    }

    return dirs[str(dir)]

df['Cardinal'] = df['WindDir'].apply(get_wind_direction)

df['dP_wind'] = dP_wind(df['WindSpeed'], np.repeat('S', len(df)))

df['dP_wind_corr'] = dP_wind(df['WindSpeed'], df['Cardinal'])

df['dT'] = df['IndoorTemp'] - df['OutdoorTemp']

df['dP_T'] = dP_T(df['dT'])


# heating stuff

df['dT_heating'] = df['IndoorTemp'].diff()
df[df['dT_heating'] < 0] = 0


Cp = 1.006 # kJ/(kg*K)
V = 300 # m^3
rho = 1.276 # kg/m^3
M = 16.043 # g/mol of CH4
dH = -891.1 # kJ/mol
R = ct.gas_constant/1e3
eff = 0.8 # efficiency of burner


df['E'] = Cp*V*rho*df['dT_heating']/eff # kJ needed
df['n'] = df['E']/dH # mol of CH4 burnt
df['m'] = df['n']*M # g of CH4 burnt
df['dP_heating'] = 3*df['n']*R*df['IndoorTemp']/V


df['dP_T_wind'] = df['dP_T'] + df['dP_wind']
df['dP_T_wind_corr'] = df['dP_T'] + df['dP_wind_corr']
df['dP_T_heating_wind'] = df['dP_T'] + df['dP_wind'] + df['dP_heating']
df['dP_T_heating_wind_corr'] = df['dP_T'] + df['dP_wind_corr'] + df['dP_heating']

to_plot = ('dP_T', 'dP_heating', 'dP_wind', 'dP_wind_corr', 'dP_T_wind', 'dP_T_wind_corr', 'dP_T_heating_wind', 'dP_T_heating_wind_corr')
fig, axes = plt.subplots(2,4, dpi=300, sharex=True, sharey=True)

for ax, col in zip(axes.flatten(), to_plot):
    df.plot(
        x='Time',
        y=['IndoorOutdoorPressure',col],
        ax=ax,
    )


"""
fig, (ax1, ax2) = plt.subplots(1,2, dpi=300)

sns.boxplot(
    x='Cardinal',
    y='IndoorOutdoorPressure',
    data=df,
    ax=ax1,
)

sns.lineplot(
    x='Cardinal',
    y='WindSpeed',
    data=df,
    ax=ax2,
)
"""
fig, axes = plt.subplots(2,4, dpi=300)

for ax, col in zip(axes.flatten(), to_plot):
    sns.kdeplot(
        df['IndoorOutdoorPressure'],
        ax=ax,
    )

    sns.kdeplot(
        df[col],
        ax=ax,
    )


# error plot

fig, ax = plt.subplots(dpi=300)

Linf, L1, L2 = [], [], []
for col in to_plot:
    x = df['IndoorOutdoorPressure'].values-df[col].values
    Linf.append(np.linalg.norm(x, ord=np.inf))
    L1.append(np.linalg.norm(x, ord=1))
    L2.append(np.linalg.norm(x, ord=2))


ax.plot(to_plot, Linf, label='Inf')
ax.plot(to_plot, L1, label='1')
ax.plot(to_plot, L2, label='2')


ax.legend()
plt.show()
