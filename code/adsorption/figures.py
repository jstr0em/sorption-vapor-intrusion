import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from analysis import IndoorSource, Kinetics, Material


# indoor adsorption figure
"""
materials = Material('concrete').get_materials()[0:-1] # list of all materials except soil


for material in materials:
    indoor = IndoorSource('../../data/transient_results_shuai_isotherm_good_mesh.csv', material=material)
    rxn = Kinetics('../../data/adsorption_kinetics.csv', material=material)
    k1, k2, K = rxn.get_reaction_constants()
    indoor.set_reaction_constants(k1, k2, K)
    t, c, c_star = indoor.solve_cstr()

    ax.plot(t, c, label=material)

ax.legend()
plt.show()
"""
material = 'concrete'
rxn = Kinetics('../../data/adsorption_kinetics.csv', material=material)
k1, k2, K = rxn.get_reaction_constants()
indoor = IndoorSource('../../data/transient_results_shuai_isotherm_good_mesh.csv', material=material)
indoor.set_reaction_constants(k1, k2, K)


print(indoor.get_dataframe())
#t, c, c_star = indoor.solve_cstr2()
#fig, ax = plt.subplots(dpi=300)


ax.plot(t, c)

plt.show()
