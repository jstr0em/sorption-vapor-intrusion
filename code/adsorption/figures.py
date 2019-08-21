import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from analysis import IndoorSource, Kinetics, Material


# indoor adsorption figure

materials = Material('concrete').get_materials()[0:-1] # list of all materials except soil

fig, ax = plt.subplots(dpi=300)
for material in materials:
    indoor = IndoorSource('../../data/no_soil_adsorption.csv', material=material)
    rxn = Kinetics('../../data/adsorption_kinetics.csv', material=material)
    k1, k2, K = rxn.get_reaction_constants()
    indoor.set_reaction_constants(k1, k2, K)
    df = indoor.get_dataframe()

    df.plot(x='time', y='c', ax=ax, logy=True, label=material)


ref = IndoorSource('../../data/no_soil_adsorption.csv', material=None)
df_ref = ref.get_dataframe()
df_ref.plot(x='time', y='c', ax=ax, logy=True, label='Ref')

ax.legend()
plt.show()
