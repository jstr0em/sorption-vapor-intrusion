import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



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

rename = dict(zip(list(df), new_names))

df.rename(columns=rename, inplace=True)

df = pd.concat([df, p['p_cpm']], axis=1)
df['d_alpha'] = df['c_in']/df['c_in'][0]
df['c_ads'] *= 131.38
fig, (ax1, ax2) = plt.subplots(1,2,dpi=300)

df.plot(x='t',y=['p_cpm','m_ads'],ax=ax1, label=['p_in (Pa)','m_ads (g)'])
df.plot(x='t',y=['alpha','d_alpha','c_ads'],ax=ax2, logy=True, label=['alpha','d_alpha', 'c_ads (g/kg)'])

plt.show()
