import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


df = pd.read_csv('../data/transient_results.csv', header=4)
p = pd.read_csv('../data/p_cpm.csv', header=4)

p.rename(columns={'Pressure (Pa)': 'p_cpm'}, inplace=True)
new_names = (
    't',
    'c_in',
    'c_ads',
    'm_ads',
    'alpha',
    'j_sides',
    'j_bottom',
    'j_ck',
    'j_d_sides',
    'j_d_bottom',
    'j_d_ck',
    'j_c_sides',
    'j_c_bottom',
    'j_c_ck',
)

soil_names = {'1': , '2': , '3': ,}

rename = dict(zip(list(df), new_names))

df.rename(columns=rename, inplace=True)

df = pd.concat([df, p['p_cpm']], axis=1)
df['d_alpha'] = df['c_in']/df['c_in'][0]
df['c_ads'] *= 131.38

#
fig = plt.figure(dpi=300)

gs1 = gridspec.GridSpec(3, 2, figure=fig)
ax1 = plt.subplot(gs1[0,:])
ax2 = ax1.twinx()

ax3 = plt.subplot(gs1[1,0])
ax4 = plt.subplot(gs1[1,1])

ax5 = plt.subplot(gs1[2,0])
ax6 = plt.subplot(gs1[2,1])


cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

df.plot(x='t',y='p_cpm',ax=ax1,legend=False)
df.plot(x='t',y='d_alpha',ax=ax2,logy=True,legend=False,color=cycle[1])

plt.show()
