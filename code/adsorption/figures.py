import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from analysis import IndoorSource, Kinetics, Material, Analysis



def indoor_adsorption():
    df, kinetics = Analysis().get_indoor_material_data()

    # indoor material analysis
    # combo plot
    materials = list(df.index.levels[0])
    fig, ax = plt.subplots(dpi=300)
    for material in materials:
        df.loc[material].plot(y='c',ax=ax, label=material.title(), logy=True, secondary_y='p')

    ax.legend()
    plt.tight_layout()

    # separate plots
    fig, (ax1, ax2, ax3) = plt.subplots(1,3,dpi=300)
    for material in materials:
        df.loc[material].plot(y='c',ax=ax1, label=material.title(), logy=False)
        df.loc[material].plot(y='c',ax=ax2, legend=False, label=material.title(), logy=True)
        df.loc[material].plot(y='c',ax=ax3, legend=False, label=material.title(), logy=True)

    ax1.set(xlim=[0,25])
    ax2.set(xlim=[25,48])
    ax3.set(xlim=[48,72])
    ax1.legend()
    plt.tight_layout()
    plt.show()
    return

def soil_adsorption():
    df = Analysis().get_soil_data()
    return

indoor_adsorption()
