import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
plt.style.use('seaborn')
def dP_T(dT):

    alpha = 0.040 # Pa/(m*K)
    dz = -3 # basement floor depth from surface m
    return alpha*dz*dT


def dP_T2(Ti, To, dz):
    alpha = 3454 # Pa*K/m
    return alpha*(1/Ti-1/To)*dz

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


df = pd.read_csv('../../data/indianapolis.csv')
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

df['Ti'] = df['IndoorTemp']+273.15
df['To'] = df['OutdoorTemp']+273.15



#df['dP_T'] = dP_T(df['dT'])
df['dP_T'] = dP_T2(df['Ti'], df['To'], 2)

"""
# heating stuff

df['dT_heating'] = df['IndoorTemp'].diff()
df[df['dT_heating'] < 0] = 0


Cp = 1.006 # kJ/(kg*K)
V = 300 # m^3
rho = 1.276 # kg/m^3
M = 16.043 # g/mol of CH4
dH = -891.1 # kJ/mol
R = 8.314462618
eff = 0.8 # efficiency of burner


df['E'] = Cp*V*rho*df['dT_heating']/eff # kJ needed
df['n'] = df['E']/dH # mol of CH4 burnt
df['m'] = df['n']*M # g of CH4 burnt
df['dP_heating'] = 3*df['n']*R*df['IndoorTemp']/V
"""

df['dP_T_wind'] = df['dP_T'] + df['dP_wind']
df['dP_T_wind_corr'] = df['dP_T'] + df['dP_wind_corr']
df['Time'] = df['Time'].apply(pd.to_datetime)


colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
fig = plt.figure(dpi=300)

gs = GridSpec(3,6, figure=fig)


# top row
ax1 = fig.add_subplot(gs[0,0:3])
ax2 = fig.add_subplot(gs[0,3:],sharex=ax1)
# bottom row
ax6 = fig.add_subplot(gs[2,:],sharex=ax1)
# middle row
ax3 = fig.add_subplot(gs[1,0:2],sharex=ax1,sharey=ax6)
ax4 = fig.add_subplot(gs[1,2:4],sharex=ax1,sharey=ax6)
ax5 = fig.add_subplot(gs[1,4:],sharex=ax1,sharey=ax6)

fig.subplots_adjust(top=0.85)

# input data (temperature and wind speed)
df.plot( x='Time', y='dT', ax=ax1, color=colors[2], legend=False, )
df.plot( x='Time', y='WindSpeed', ax=ax2, legend=False, color=colors[3], sharex=ax1)

# cosmetic stuff
ax1.set(ylabel='$\\Delta T$ (C)', title='Indoor/outdoor temperature difference')
ax2.set(ylabel='u (m/s)',xlabel='', title='Wind speed')
ax2.tick_params(axis='x', labelbottom=False)

# middle row plots (individual dP contributions)
df.plot( x='Time', y='dP_T', ax=ax3, color=colors[1], legend=False)
df.plot( x='Time', y='dP_wind', ax=ax4, color=colors[1], legend=False)
df.plot( x='Time', y='dP_wind_corr', ax=ax5, color=colors[1], legend=False)

# cosmetic stuff
ax3.set(ylabel='$\\Delta p$ (Pa)', title='Temperature contribution\n$\\Delta p = \\alpha \\Delta z (1/T_{in} - 1/T_{out})$ \n $\\alpha = 3454\\; \\mathrm{(Pa\\cdot K/m))}, \\; \\Delta z=2\\; \\mathrm{(m)}$')
ax4.set(xlabel='', title='Wind contribution\n$\\Delta p = \\frac{1}{2}C_j\\rho u^2, \\; C_j = 0.3$')
ax5.set(xlabel='', title='Direction corrected\nwind contribution\nN, NE, E, SE => negative ')
ax4.tick_params(axis='x', labelbottom=False)
ax5.tick_params(axis='x', labelbottom=False)

# comparing the actual and predicted values
df.plot( x='Time', y='IndoorOutdoorPressure', ax=ax6, )
df.plot( x='Time', y='dP_T_wind_corr', ax=ax6, alpha=0.8, )

# cosmetic stuff
ax6.set(ylabel='$\\Delta p$ (Pa)',title='Temperature and (corrected) wind contributions vs. recorded pressure difference')
ax6.legend(labels=['Data', 'Estimation'],frameon=True, loc='upper center')

fig.suptitle(
    'Estimation of temperatue and wind induced indoor/outdoor pressure difference at the Indianapolis site',
    #y=1.01
)

fig.tight_layout(rect=[0, 0.03, 1, 0.95])

#plt.tight_layout()
plt.show()
