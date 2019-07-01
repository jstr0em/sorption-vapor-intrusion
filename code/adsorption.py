import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



df = pd.read_csv('../data/simulation_adsorption_p.csv', header=4, index_col=0)


print(df)

M = 131.39 # g/mol
df['c_ads (mol/kg)'] *= M # converts to g adsorbate/kg soil

fig, axes = plt.subplots(2,2,dpi=300,sharex=True)


titles = (
    'Indoor concentration (mol/m^3)',
    'Adsorbed concentration (g/kg)',
    'Adsorbed mass (g)',
    'Attenuation from source',
)

for col, title, ax in zip(list(df), titles, axes.flatten()):
    df.plot(
        ax=ax,
        y=col,
        title=title,
        legend=False,
    )
    ax.set(
        yscale='log',
    )

axes = axes.flatten()



plt.tight_layout()
plt.show()
