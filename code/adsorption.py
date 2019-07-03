import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



df = pd.read_csv('../data/simulation_adsorption_p.csv', header=4, index_col=0)


M = 131.39 # g/mol
df['c_ads (mol/kg)'] *= M*1e3 # converts to mg adsorbate/kg soil
df['c_indoor (mol/m^3)'] *= M*1e6 # ug/m3
fig, axes = plt.subplots(2,3,dpi=300,sharex=True)


titles = (
    'Indoor concentration (ug/m^3)',
    'Adsorbed concentration (mg/kg)',
    'Adsorbed mass (g)',
    'Attenuation from source',
    'Flux from below (mol/(m^2*s))',
    'Flux from sides (mol/(m^2*s))',
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
        xlabel='$\\Delta P_\\mathrm{in/out}$ (Pa)'
    )

axes = axes.flatten()

axes[1].set(
    yscale='linear'
)
axes[2].set(
    yscale='linear'
)

axes[-2].set(
    yscale='linear'
)
axes[-1].set(
    yscale='linear'
)

#plt.tight_layout()
plt.show()
