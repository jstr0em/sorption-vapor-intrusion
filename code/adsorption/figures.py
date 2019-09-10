import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from analysis import IndoorSource, Kinetics, Material, Analysis
import matplotlib.ticker as mtick

def indoor_adsorption():
    analysis = Analysis()
    df = analysis.get_indoor_material_data()
    kin = analysis.get_kinetics_data()

    # indoor material analysis
    # combo plot
    materials = list(df.index.levels[0])
    fig, ax = plt.subplots(dpi=300)
    for material in materials:
        df_now = df.loc[material]
        df_now.plot(y='c_in',ax=ax, label=material.title())

    return

def soil_adsorption():
    analysis = Analysis()
    ss = analysis.get_steady_state_data()
    df = analysis.get_soil_sorption_data()
    cases = list(df.index.levels[0])

    print(cases, df)
    fig, ax = plt.subplots(dpi=300)
    for case in cases:
        df_now = df.loc[case]
        df_now.plot(y='c_in', label='K_ads = %1.2e' % case, ax=ax)
    return

def transport_analysis():
    analysis = Analysis()
    ss = analysis.get_steady_state_data()
    t = analysis.get_soil_data()
    cases = list(ss.index.levels[0])


    t_ads = t.loc[cases[1]]
    t_ref = t.loc[cases[0]]
    ss_ads = ss.loc[cases[1]]
    ss_ref = ss.loc[cases[0]]

    figure_scheme = {'cols': ['Pe', 'u_ck', 'c_soil'],
    'ylabels': ['Peclet number through crack','Average velocity into crack (cm/hr)','Average concentration in zone of influence (ug/m3)']}

    # soil concentration plots

    for col, ylabel in zip(figure_scheme['cols'], figure_scheme['ylabels']):
        fig, (ax1, ax2) = plt.subplots(2,1,dpi=300, sharex=True, sharey=True)

        t_ads.plot(x='p_in', y=col, ax=ax1, label='Transient solution')
        ss_ads.plot(y=col, ax=ax1, label='Steady-state solution')

        t_ref.plot(x='p_in', y=col, ax=ax2, legend=False)
        ss_ref.plot(y=col, ax=ax2, legend=False)

        fig.text(0.025, 0.5, ylabel, va='center', rotation='vertical')

        ax1.set(title='Soil sorption')
        ax2.set(title='No soil sorption')

    return

def indoor_adsorption_zero_entry():
    analysis = Analysis()
    df = analysis.get_indoor_zero_entry_material_data()
    kin = analysis.get_kinetics_data()

    # indoor material analysis
    # combo plot
    materials = list(df.index.levels[0])
    fig, ax = plt.subplots(dpi=300)
    for material in materials:
        df_now = df.loc[material]
        c_gw = df_now['c'].values[0]/df_now['alpha'].values[0]
        df_now['alpha2'] = df_now['c']/c_gw
        df_now.plot(y='alpha2',ax=ax, label=material.title(), logy=True, secondary_y='p')

    #ax.table(cellText=df.values, rowLabels=df.index, colLabels=df.columns,cellLoc = 'center', rowLoc = 'center', loc='bottom')

    ax.legend()
    ax.set(xlabel='Time (hr)', ylabel='Attenuation from groundwater', title='Attenuation factor from groundwater\nfollowing elimination of contaminant entry')
    plt.tight_layout()
    return


#Kinetics(file='../../data/adsorption_kinetics.csv',material='drywall').plot()
soil_adsorption()
#transport_analysis()
#indoor_adsorption()
#indoor_adsorption_zero_entry()
plt.show()
