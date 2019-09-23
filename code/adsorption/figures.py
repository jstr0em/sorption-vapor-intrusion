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
    ads_ratio = []

    fig, ax = plt.subplots(dpi=300)
    for case in cases:
        df_now = df.loc[case]
        df_now['d_alpha'] = df_now['alpha'].values / df.loc[cases[0]]['alpha'].values
        depress_change.append(df_now['d_alpha'].max())
        overpress_change.append(df_now['d_alpha'].min())
        ads_ratio.append(df_now['c_ads/c_gas'].values[0])
        df_now.plot(y='alpha', label='K_ads = %1.2e' % case, ax=ax)
    ax.set(title='Attenuation factor during during over-/depressurization cycle',
    xlabel='Time (hr)', ylabel='$\\alpha$')
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))


    fig, (ax1, ax2) = plt.subplots(2,1,dpi=300,sharex=True)
    change_in_response = pd.DataFrame({'K_ads': cases.astype(str), 'dP_plus': depress_change, 'dP_neg': overpress_change, 'c_ads/c_gas': ads_ratio})
    change_in_response.plot(x='K_ads', y=['dP_plus', 'dP_neg'], ax=ax1, style='-o', label=['Overpressurization', 'Depressurization'])
    change_in_response.plot(x='K_ads', y='c_ads/c_gas', ax=ax2, style='-o', legend=False,logy=True)

    ax1.set(title='Change in attenuation factor relative to no soil sorption\nduring over-/depressurization cycle\nas a function of soil sorption isotherm',
     ylabel='$\\frac{\\alpha}{\\alpha_{no-sorption}}$')
    ax2.set(xticks=np.arange(0,len(cases)), xticklabels=cases.astype(str),
    xlabel='$K_\\mathrm{ads} \; \\mathrm{(m^3/kg)}$', ylabel='$c_{ads}/c_{gas}$')

    fig, ax = plt.subplots(dpi=300)
    change_in_response.plot(x='c_ads/c_gas', y=['dP_plus', 'dP_neg'], ax=ax, style='-o', label=['Overpressurization', 'Depressurization'])
    ax.set(xlabel='$K_{ads} \; \\mathrm{(m^3/kg)}$', ylabel='$c_{ads}/c_gas$')
    return

def transport_analysis():
    analysis = Analysis()
    ss = analysis.get_steady_state_data()
    t = analysis.get_soil_sorption_data()
    cases = np.array(ss.index.levels[0])
    cases2 = np.array(t.index.levels[0])

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

    #print(t.unstack())
    fig, ax = plt.subplots(dpi=300)
    t.unstack().plot(y='c_ads/c_gas',ax=ax, legend=False, logx=True, logy=True, marker='o') # TODO: fix so that only 1 line is produced instead of 72...
    ax.set(ylabel='$c_{ads}/c_{gas}$', xlabel='$K_{ads} \; \\mathrm{(m^3/kg)}$')
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


def time_to_equilibrium():
    analysis = Analysis()
    df = analysis.get_time_to_equilibrium_data()
    ss = analysis.get_steady_state_data()
    ss = ss.loc[5.28]

    fig, (ax1, ax2) = plt.subplots(2,1,sharex=True,dpi=300)
    ax3 = ax2.twinx()

    isotherms = list(df.index.levels[0])
    cycles = list(df.index.levels[1])
    labels = {15: 'Overpressurization, -5 -> 15 Pa', -15: 'Depressurization, -5 -> -15 Pa'}

    for isotherm in isotherms:
        for cycle in cycles:
            df_now = df.loc[(isotherm, cycle)]
            p_in = df_now['p_in'].values[-1]

            alpha0 = df_now['alpha'].values[0]
            alpha_eq = ss.loc[p_in]['alpha']

            u0 = df_now['u_ck'].values[0]
            u_eq = ss.loc[p_in]['u_ck']

            c0 = df_now['c_gas'].values[0]
            c_eq = ss.loc[p_in]['c_gas']

            df_now['alpha_from_eq'] = np.abs(df_now['alpha']-alpha0)/np.abs(alpha_eq-alpha0)
            df_now['u_ck_from_eq'] = np.abs(df_now['u_ck']-u0)/np.abs(u_eq-u0)
            df_now['c_gas_from_eq'] = np.abs(df_now['c_gas']-c0)/np.abs(c_eq-c0)

            # plotting
            df_now.plot(y='alpha_from_eq', ax=ax1, label=labels[p_in])
            #df_now.plot(y='p_in', ax=ax2, legend=False,linestyle='--')
            df_now.plot(y='u_ck_from_eq', ax=ax2, legend=False)
            df_now.plot(y='c_gas_from_eq', ax=ax3, legend=False, linestyle='--', logy=True)

            # formatting
            ax1.set(ylabel='$\\frac{|\\alpha-\\alpha_0|}{|\\alpha_{eq}-\\alpha_0|}$', title='Distance from new equilibrium state following\nindoor pressurization change')
            ax2.set(ylabel='$\\frac{u_{ck}-u_{ck,0}|}{|u_{ck,eq}-u_{ck,0}|}$',
            xlabel='Time (hr)')
            ax3.set(ylabel='$\\frac{|c_{gas}-c_{gas,0}|}{|c_{gas,eq}-c_{gas,0}|}$', ylim=[0,1])
            ax1.legend(loc='center right')
            plt.tight_layout()
    return

#Kinetics(file='../../data/adsorption_kinetics.csv',material='drywall').plot()
#soil_adsorption()
#transport_analysis()
#indoor_adsorption()
#indoor_adsorption_zero_entry()
time_to_equilibrium()
plt.show()
