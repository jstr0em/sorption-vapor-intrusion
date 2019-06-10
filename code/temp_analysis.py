import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

df = pd.read_csv('./data/temp_sim.csv')



piv = df.pivot_table(
    index='T',
    columns=['Soil','Sub-base'],
    values='C',
)

for soil in df['Soil'].unique():
    for sub_base in df['Sub-base'].unique():
        piv[(soil, sub_base)] = piv[(soil, sub_base)]/piv[(soil, sub_base)].values[0]
print(piv)
fig, ax = plt.subplots(dpi=300)


"""
sns.lineplot(
    data=df,
    x='T',
    y='C',
    hue='Soil',
    style='Sub-base',
    ax=ax,
)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%2.1e'))

plt.show()
"""


piv.plot(ax=ax)
plt.show()
