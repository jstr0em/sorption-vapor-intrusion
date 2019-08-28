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
        c_gw = df_now['c'].values[0]/df_now['alpha'].values[0]
        df_now['alpha2'] = df_now['c']/c_gw
        df_now.plot(y='alpha2',ax=ax, label=material.title(), logy=True, secondary_y='p')

    #ax.table(cellText=df.values, rowLabels=df.index, colLabels=df.columns,cellLoc = 'center', rowLoc = 'center', loc='bottom')

    ax.legend()
    ax.set(xlabel='Time (hr)', ylabel='Attenuation from groundwater')
    plt.tight_layout()

    # separate plots
    fig, (ax1, ax2, ax3) = plt.subplots(1,3,dpi=300)
    for material in materials:
        df_now = df.loc[material]
        c_gw = df_now['c'].values[0]/df_now['alpha'].values[0]
        df_now['alpha2'] = df_now['c']/c_gw
        df_now.plot(y='alpha2',ax=ax1, label=material.title(), logy=False)
        df_now.plot(y='alpha2',ax=ax2, legend=False, label=material.title(), logy=True)
        df_now.plot(y='alpha2',ax=ax3, legend=False, label=material.title(), logy=True)

    ax1.set(xlim=[0,25], ylabel='Attenuation from groundwater')
    ax2.set(xlim=[25,48])
    ax3.set(xlim=[48,72])
    ax1.legend()
    plt.tight_layout()
    return

def soil_adsorption():
    analysis = Analysis()
    ss = analysis.get_steady_state_data()
    t = analysis.get_soil_data()
    cases = list(ss.index.levels[0])

    t_ads = t.loc[cases[1]]
    t_ref = t.loc[cases[0]]
    ss_ads = ss.loc[cases[1]]
    ss_ref = ss.loc[cases[0]]


    fig, ax = plt.subplots(dpi=300)
    t_ads.plot(y='alpha', ax=ax, logy=True, label='Soil sorption')
    t_ref.plot(y='alpha', ax=ax, logy=True, label='No soil sorption')

    ax.set(xlabel='Time (hr)',  ylabel='Attenuation from groundwater', title='Effect of soil sorption\nm_ads/m_soil = %1.2e' % t_ads['c_ads/c_soil'].mean())


    # change in adsorbed mass
    fig, ax = plt.subplots(dpi=300)


    t_ads['m_ads_change'] = t_ads['m_ads'].values/t_ads['m_ads'].values[0]*100
    t_ads.plot(y='m_ads_change', ax=ax)

    ax.set(ylabel='%-change', xlabel='Time (hr)', title='Change in adsorbed mass')
    #fmt = '%.2f%%' # Format you want the ticks, e.g. '40%'
    #yticks = mtick.FormatStrFormatter(fmt)
    #ax.yaxis.set_major_formatter(yticks)
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

#Kinetics(file='../../data/adsorption_kinetics.csv',material='drywall').plot()

soil_adsorption()
#transport_analysis()
#indoor_adsorption()
plt.show()
