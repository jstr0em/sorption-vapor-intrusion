import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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

df['dP_wind'] = dP_wind(df['WindSpeed'], df['Cardinal'])

df['dT'] = df['IndoorTemp'].values - df['OutdoorTemp'].values

df['dP_T'] = dP_T(df['dT'])
df['dP_combo'] = df['dP_T'].values + df['dP_wind']




fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)

df.plot(
    x='Time',
    y=['IndoorOutdoorPressure','dP_T'],
    ax=ax1,
)

df.plot(
    x='Time',
    y=['IndoorOutdoorPressure','dP_wind'],
    ax=ax2,
)

df.plot(
    x='Time',
    y=['IndoorOutdoorPressure','dP_combo'],
    ax=ax3,
)

df.plot(
    x='Time',
    y=['IndoorOutdoorPressure','WindDir'],
    ax=ax4,
)

fig, (ax1, ax2) = plt.subplots(1,2)

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

plt.show()
