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
    def equilibrium_distance(df_t, df_ss):
        # end pressure
        p_in = df_t['p_in'].values[-1]
        df_ss = df_ss.loc[p_in]

        alpha0 = df_t['alpha'].values[0]
        alpha_eq = df_ss['alpha']

        u0 = df_t['u_ck'].values[0]
        u_eq = df_ss['u_ck']

        c0 = df_t['c_gas'].values[0]
        c_eq = df_ss['c_gas']

        df_t['alpha_from_eq'] = np.abs(df_t['alpha']-alpha0)/np.abs(alpha_eq-alpha0)
        df_t['u_ck_from_eq'] = np.abs(df_t['u_ck']-u0)/np.abs(u_eq-u0)
        df_t['c_gas_from_eq'] = np.abs(df_t['c_gas']-c0)/np.abs(c_eq-c0)
        return df_t

    analysis = Analysis()
    df = analysis.get_time_to_equilibrium_data()
    ss = analysis.get_steady_state_data()
    #ss = ss.loc[5.28]

    fig, (ax1, ax2) = plt.subplots(2,1,sharex=True,dpi=300)
    ax3 = ax2.twinx()

    soils = list(df.index.levels[0])
    isotherms = list(df.index.levels[1])
    cycles = list(df.index.levels[2])
    labels = {15: '-5 -> 15 Pa', -15: '-5 -> -15 Pa'}
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    handles = []
    labels = []

    for i, isotherm in enumerate(isotherms):
        ss_now = ss.loc[('Sandy Loam', 5.28)] # keep
        color = colors[i]
        handles.append(plt.Line2D((0,1),(0,0), color=color,linestyle='-'))
        labels.append('K_ads = %1.2e' % isotherm)

        for cycle in cycles:

            df_now = df.loc[('Sandy Loam',isotherm, cycle)] # keep

            df_now = equilibrium_distance(df_now, ss_now)
            # plotting
            p_in = df_now['p_in'].values[-1]
            if p_in == -15:
                df_now.plot(y='alpha_from_eq', style='-', ax=ax1, legend=False, color=color)
            elif p_in == 15:
                df_now.plot(y='alpha_from_eq', style='--', ax=ax1, legend=False, color=color)

            #df_now.plot(y='p_in', ax=ax2, legend=False,linestyle='--')
            df_now.plot(y='u_ck_from_eq', ax=ax2, color=color, legend=False)
            df_now.plot(y='c_gas_from_eq', ax=ax3, color=color, legend=False, linestyle='--', logy=True)

    # formatting
    ax1.set(ylabel='$\\frac{|\\alpha-\\alpha_0|}{|\\alpha_{eq}-\\alpha_0|}$', title='Distance from new equilibrium state following\nindoor pressurization change')
    ax2.set(ylabel='$\\frac{u_{ck}-u_{ck,0}|}{|u_{ck,eq}-u_{ck,0}|}$',
    xlabel='Time (hr)')
    ax3.set(ylabel='$\\frac{|c_{gas}-c_{gas,0}|}{|c_{gas,eq}-c_{gas,0}|}$', ylim=[0,1])


    # custom legend
    handles.append(plt.Line2D((0,1),(0,0), color='k',linestyle='-'))
    handles.append(plt.Line2D((0,1),(0,0), color='k',linestyle='--'))
    labels.append('-5 -> -15 Pa')
    labels.append('-5 -> 15 Pa')
    ax1.legend(handles=handles, labels=labels,loc='center right')



    # sand analysis
    fig, (ax1, ax2, ax3) = plt.subplots(3,1, dpi=300, sharex=True)
    handles = []
    labels = []


    for i, soil in enumerate(soils):
        color = colors[i]
        handles.append(plt.Line2D((0,1),(0,0), color=color,linestyle='-'))
        labels.append(soil)
        for cycle in cycles:
            df_now = df.loc[(soil, 0, cycle)]
            ss_now = ss.loc[(soil, 0)]
            df_now = equilibrium_distance(df_now, ss_now)
            # plotting
            p_in = df_now['p_in'].values[-1]
            if p_in == -15:
                linestyle='-'
            elif p_in == 15:
                linestyle='--'

            df_now.plot(y='alpha_from_eq', linestyle=linestyle, ax=ax1, legend=False, color=color)
            df_now.plot(y='u_ck_from_eq', ax=ax2, linestyle=linestyle, color=color, legend=False)
            df_now.plot(y='c_gas_from_eq', ax=ax3, color=color, legend=False, linestyle=linestyle)

            #df_now.plot(y='p_in', ax=ax2, legend=False,linestyle='--')
            #df_now.plot(y='u_ck_from_eq', ax=ax2, color=color, legend=False)
            #df_now.plot(y='c_gas_from_eq', ax=ax3, color=color, legend=False, linestyle='--', logy=True)

    handles.append(plt.Line2D((0,1),(0,0), color='k',linestyle='-'))
    handles.append(plt.Line2D((0,1),(0,0), color='k',linestyle='--'))
    labels.append('-5 -> -15 Pa')
    labels.append('-5 -> 15 Pa')
    ax1.legend(handles=handles, labels=labels,loc='center right')

    plt.tight_layout()
    return

#Kinetics(file='../../data/adsorption_kinetics.csv',material='drywall').plot()
#soil_adsorption()
#transport_analysis()
#indoor_adsorption()
#indoor_adsorption_zero_entry()
time_to_equilibrium()
plt.show()
