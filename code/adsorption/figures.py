import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from analysis import IndoorSource, Kinetics, Material, Analysis


def indoor_adsorption():
    analysis = Analysis()
    df = analysis.get_indoor_material_data()
    kin = analysis.get_kinetics_data()
    print(kin)
    # indoor material analysis
    # combo plot
    materials = list(df.index.levels[0])
    fig, ax = plt.subplots(dpi=300)
    for material in materials:
        df.loc[material].plot(y='c',ax=ax, label=material.title(), logy=True, secondary_y='p')

    #ax.table(cellText=df.values, rowLabels=df.index, colLabels=df.columns,cellLoc = 'center', rowLoc = 'center', loc='bottom')

    ax.legend()
    ax.set(xlabel='Time (hr)', ylabel='c_in (ug/m3)')
    plt.tight_layout()

    # separate plots
    fig, (ax1, ax2, ax3) = plt.subplots(1,3,dpi=300)
    for material in materials:
        df.loc[material].plot(y='c',ax=ax1, label=material.title(), logy=False)
        df.loc[material].plot(y='c',ax=ax2, legend=False, label=material.title(), logy=True)
        df.loc[material].plot(y='c',ax=ax3, legend=False, label=material.title(), logy=True)

    ax1.set(xlim=[0,25], ylabel='c_in (ug/m3)')
    ax2.set(xlim=[25,48])
    ax3.set(xlim=[48,72])
    ax1.legend()
    plt.tight_layout()
    return

def soil_adsorption():
    df = Analysis().get_soil_data()

    cases = list(df.index.levels[0])

    fig, ax = plt.subplots(dpi=300)

    for case in cases:
        df.loc[case].plot(y='c', ax=ax, label=case.title(), logy=True)


    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1,dpi=300,sharex=True)
    for case in cases:
        df_now = df.loc[case]
        df_now.plot(y='Pe', ax=ax1, label=case.title()+', c_ads/c_soil = %1.1e' % df_now['c_ads/c_soil'].mean(), legend=False)
        df_now.plot(y='u_ck', ax=ax2, label=case.title(), legend=False)
        df_now.plot(y='c_soil', ax=ax3, label=case.title(), legend=False)
        df_now.plot(y='c_ads/c_soil', ax=ax4, label=case.title(), legend=False)


    ax1.set(ylabel='Pe')
    ax2.set(ylabel='u_ck (cm/hr)')
    ax3.set(xlabel='time (hr)', ylabel='c_soil (ug/m^3)', title='Average soil concentration in radius of influence')
    ax1.legend()
    plt.tight_layout()


    fig, ax = plt.subplots(dpi=300)
    for case in cases:
        df_now = df.loc[case]
        df_now.plot(x='p_in',y='c_soil',ax=ax,label=case.title())
    return


def transport_analysis():
    analysis = Analysis()
    df = analysis.get_steady_state_data()

    cases = list(df.index.levels[0])

    fig, ax = plt.subplots(dpi=300)

    for case in cases:
        df_now = df.loc[case]

        df_now.plot(x='p', y='Pe', ax=ax)

    return

indoor_adsorption()
#soil_adsorption()


#Kinetics(file='../../data/adsorption_kinetics.csv',material='drywall').plot()
plt.show()
