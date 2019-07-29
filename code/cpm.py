import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


df = pd.read_csv('../data/transient_results.csv', header=4)
ss = pd.read_csv('../data/steady_state_material_sweep.csv', header=4)

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

new_names2 = (
    'soil',
    'dp',
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

soil_names = {'1': 'Sandy Clay', '2': 'Sand', '3': 'Loamy Sand',}

rename = dict(zip(list(df), new_names))

df.rename(columns=rename, inplace=True)
ss.rename(columns=dict(zip(list(ss), new_names2)), inplace=True)
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


# sharing axes
#ax3.get_shared_y_axes().join(ax3, ax4)
#ax5.get_shared_y_axes().join(ax5, ax6)

# joins x-axes of the middle and bottom plots
ax3.get_shared_x_axes().join(ax3, ax5)
ax4.get_shared_x_axes().join(ax4, ax6)


cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

#
ss_sandy_clay = ss[ss['soil']==1]

# top plot
df.plot(x='t',y='p_cpm',ax=ax1,legend=False)
df.plot(x='t',y='d_alpha',ax=ax2,logy=True,legend=False,color=cycle[1])

# middle left
df.plot(x='t',y='alpha',ax=ax3,logy=True,legend=False)

# middle right
#df.plot(x='p_cpm',y='alpha',ax=ax4,logy=True,legend=False)
ss_sandy_clay.plot(x='dp',y='alpha',ax=ax4,logy=True,legend=False)

# bottom left
df.plot(x='t',y='m_ads',ax=ax5,legend=False)
#ax5.plot(ss['m_ads'].min())


# bottom right
#df.plot(x='p_cpm',y='m_ads',ax=ax6,legend=False)
ss_sandy_clay.plot(x='dp',y='m_ads',ax=ax6,legend=False)


# axis labels
ax1.set_xlabel('Time (h)')
ax1.set_ylabel('$\\Delta p$ (Pa)')
ax2.set_ylabel('$\\alpha/\\alpha_{ss}$')

ax5.set_xlabel('Time (h)')
ax3.set_ylabel('$\\alpha$')
ax5.set_ylabel('$m_{ads}$ (g)')

ax6.set_xlabel('$\\Delta p$ (Pa)')

# titles
ax1.set_title('CPM cycle and change in $\\alpha$')
ax3.set_title('Transient $\\alpha$')
ax4.set_title('Steady-state $\\alpha$')
ax5.set_title('Transient $m_{ads}$')
ax6.set_title('Steady-state $m_{ads}$')

plt.tight_layout()
plt.show()
