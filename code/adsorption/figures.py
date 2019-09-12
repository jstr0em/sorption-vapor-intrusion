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
    cases = np.array(df.index.levels[0])

    depress_change = []
    overpress_change = []

    fig, ax = plt.subplots(dpi=300)
    for case in cases:
        df_now = df.loc[case]
        df_now['d_alpha'] = df_now['alpha'].values / df.loc[cases[0]]['alpha'].values
        depress_change.append(df_now['d_alpha'].max())
        overpress_change.append(df_now['d_alpha'].min())
        df_now.plot(y='alpha', label='K_ads = %1.2e' % case, ax=ax)
    ax.set(title='Attenuation factor during during over-/depressurization cycle',
    xlabel='Time (hr)', ylabel='$\\alpha$')
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))


    fig, ax = plt.subplots(dpi=300)
    change_in_response = pd.DataFrame({'K_ads': cases.astype(str), 'dP_plus': depress_change, 'dP_neg': overpress_change})
    change_in_response.plot(x='K_ads', y=['dP_plus', 'dP_neg'], ax=ax, style='-o', label=['Overpressurization', 'Depressurization'])
    ax.set(title='Change in attenuation factor relative to no soil sorption\nduring over-/depressurization cycle\nas a function of soil sorption isotherm',
    xticks=np.arange(0,len(cases)), xticklabels=cases.astype(str),
    xlabel='$K_\\mathrm{ads} \; \\mathrm{(m^3/kg)}$', ylabel='$\\frac{\\alpha}{\\alpha_{no-sorption}}$')

    return

def transport_analysis():
    analysis = Analysis()
    ss = analysis.get_steady_state_data()
    t = analysis.get_soil_sorption_data()
    cases = np.array(ss.index.levels[0])

    figure_scheme = {'cols': ['Pe', 'u_ck', 'c_gas'],
    'ylabels': ['Peclet number through crack','Average velocity into crack (cm/hr)','Average concentration in zone of influence (ug/m3)']}


    ss_ref = ss.loc[cases[0]]
    # soil concentration plots

    for col, ylabel in zip(figure_scheme['cols'], figure_scheme['ylabels']):
        fig, ax = plt.subplots(dpi=300)

        # steady-state plot

        ss_ref.plot(y=col, ax=ax, label='Steady-state')
        for case in cases:
            t_now = t.loc[case]
            t_now.plot(x='p_in',y=col, ax=ax, label='K_ads = %1.2e' % case)
            ax.set(ylabel=ylabel)

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
        df_now.plot(y='alpha',ax=ax, label=material.title(), logy=True, secondary_y='p')

    #ax.table(cellText=df.values, rowLabels=df.index, colLabels=df.columns,cellLoc = 'center', rowLoc = 'center', loc='bottom')

    ax.legend()
    ax.set(xlabel='Time (hr)', ylabel='Attenuation from groundwater', title='Attenuation factor from groundwater\nfollowing elimination of contaminant entry')
    plt.tight_layout()
    return


#Kinetics(file='../../data/adsorption_kinetics.csv',material='drywall').plot()
#soil_adsorption()
#transport_analysis()
#indoor_adsorption()
indoor_adsorption_zero_entry()
plt.show()
