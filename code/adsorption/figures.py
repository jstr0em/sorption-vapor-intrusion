import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from analysis import IndoorSource, Kinetics, Material, Analysis
import matplotlib.ticker as mtick
from matplotlib.gridspec import GridSpec
plt.style.use('seaborn')
def indoor_adsorption():
    # loading all the data
    analysis = Analysis()
    df = analysis.get_indoor_material_data()
    kin = analysis.get_kinetics_data()

    materials = list(kin.sort_values(by='K',ascending=False).index)
    materials.insert(0, 'none')
    materials.remove('soil')

    # setting up figure
    fig = plt.figure(dpi=300, constrained_layout=True)
    gs = GridSpec(2,2, figure=fig)

    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[1,:])


    # pressure plot
    df.loc[materials[0]].plot(y='p_in', ax=ax1, color='k', legend=False)

    for material in materials:
        df_now = df.loc[material]
        df_now.plot(y='rxn', ax=ax2, legend=False)
        df_now.plot(y='c_in',ax=ax3, label=material.title(), legend=False)

    xlabel = 'Time (h)'

    ax1.set(title='Building pressurization cycle', ylabel='$\\Delta p \\; \\mathrm{(Pa)}$', xlabel=xlabel)
    ax2.set(title='Sorption rate', ylabel='$r_{sorb} \\; \\mathrm{(\\mu g/h)}$', xlabel=xlabel)
    ax3.set(title='Indoor air contaminant concentration', ylabel='$c_{in} \\; \\mathrm{(\\mu g/m^3)}$', xlabel=xlabel)


    handles, labels = ax3.get_legend_handles_labels()

    # TODO: See if you can place the figure legend where it currently is without this hack...
    ax3.legend([],[],loc='center left',bbox_to_anchor=(1.15,1))
    fig.legend(handles, labels,loc='center left', title='Material', bbox_to_anchor=(0.85,0.5))

    return

def sorption_fit():
    """
    Figure showing some of Shuai's data points and my kinetic model fit to them
    """

    materials = ['wood', 'concrete']
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    handles = []
    labels = []
    fig, ax = plt.subplots(dpi=300)

    # loops over the two chosen materials and plots their sorption data with
    # fitted curves
    for i, material in enumerate(materials):
        color = colors[i] # current color

        # creates instance
        kin = Kinetics(material=material)
        # gets data kinetic and contaminant data
        k1, k2, K = kin.get_reaction_constants()
        M = kin.get_molar_mass()
        rho = kin.get_material_density()

        t_data = kin.get_time_data()
        c_star_data = kin.get_adsorbed_conc()

        t = np.linspace(t_data[0], t_data[-1], 200)
        c_star = kin.solve_reaction(t, k1, k2)

        ax.semilogy(t_data * 60, c_star_data / rho * M * 1e9, 'o', color=color)
        ax.semilogy(t * 60, c_star / rho * M * 1e9, color=color)

        # legend entry
        handles.append(plt.Line2D((0,1),(0,1),color=color,linestyle='-'))
        #labels.append('%s, $k_1$ = %1.1e, $k_2$ = %1.1e, K = %1.1e' % (material.title(), k1, k2, K))
        labels.append('%s' % material.title())

    # custom handles and labels
    handles.append(plt.Line2D((0,1),(0,1),color='k'))
    handles.append(plt.Line2D((0,1),(0,1),marker='o',linestyle='None',color='k'))
    labels.append('Fitted curve')
    labels.append('Experimental data')

    ax.legend(handles, labels, title='Material')

    ax.set(
        title='Sorption of 1.12 $\\mathrm{ppb_v}$ of %s on %s and %s with fitted curves' % (kin.get_contaminant().upper(), materials[0], materials[1]),
        xlabel='Time (min)',
        ylabel='Adsorbed mass (ng/g)'
        )

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




    materials = list(kin.sort_values(by='K',ascending=False).index)
    materials.insert(0, 'none')
    materials.remove('soil')

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # indoor material analysis
    # combo plot
    #materials = list(df.index.levels[0])
    fig, (ax1, ax2) = plt.subplots(2,1,sharex=True,dpi=300)
    for i, material in enumerate(materials):

        color = colors[i]
        df_now = df.loc[material]
        df_now.plot(y='alpha',ax=ax1, label=material.title(), logy=True, color=color)

        if material != 'none':
            print(material, df_now['c_mat'].max()/df_now['c_in'].max())
            df_now['ConcentrationDifference'] = df_now['alpha']/df.loc['none']['alpha']
            df_now.plot(y='ConcentrationDifference', ax=ax2, label=material.title(), logy=True, color=color, legend=False)

    #ax.table(cellText=df.values, rowLabels=df.index, colLabels=df.columns,cellLoc = 'center', rowLoc = 'center', loc='bottom')

    ax1.legend(title='Material')
    ax1.set(ylabel='$\\alpha$', title='Attenuation factor from groundwater following elimination of contaminant entry')
    ax2.set(xlabel='Time (hr)', ylabel='$\\alpha/\\alpha_{ref}$', title='Elevated indoor concentration due to indoor material sorption')

    plt.tight_layout()
    return


def time_to_equilibrium_old():
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

    soils = ('Sand',)
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

    # formatting
    ax1.set(ylabel='$\\frac{|\\alpha-\\alpha_0|}{|\\alpha_{eq}-\\alpha_0|}$', title='Distance from new equilibrium state following\nindoor pressurization change')
    ax2.set(ylabel='$\\frac{u_{ck}-u_{ck,0}|}{|u_{ck,eq}-u_{ck,0}|}$',
    xlabel='Time (hr)')
    ax3.set(ylabel='$\\frac{|c_{gas}-c_{gas,0}|}{|c_{gas,eq}-c_{gas,0}|}$')

    plt.tight_layout()
    return

def time_to_equilibrium():
    analysis = Analysis()
    df = analysis.get_time_to_equilibrium_data()
    ss = analysis.get_steady_state_data()

    soils = df.index.levels[0]
    Ks = df.index.levels[1]
    cycles = df.index.levels[2]

    vars = ['alpha', 'u_ck', 'c_gas']

    for cycle in cycles:
        # setting up figure
        fig = plt.figure(dpi=300, constrained_layout=True)
        gs = GridSpec(2,2, figure=fig)

        ax1 = fig.add_subplot(gs[0,:])
        ax2 = fig.add_subplot(gs[1,0])
        ax3 = fig.add_subplot(gs[1,1])
        axes = (ax1, ax2, ax3)
        #soils = [soils[-1],] # temporary thing
        for soil in soils:
            for K in Ks:
                try:
                    df_now = df.loc[(soil, K, cycle)]
                    p_end = df_now['p_in'].values[-1]
                    ss_now = ss.loc[(soil, 5.28, p_end)]

                    for ax, var in zip(axes, vars):
                        var_0 = df_now[var].values[0] # intial value
                        var_eq = ss_now[var] # equilibrium values
                        df_now[var+'_distance'] = (df_now[var]-var_0)/(var_eq-var_0)
                        df_now.plot(y=var+'_distance', ax=ax, legend=False, label='%s, K = %1.2e' % (soil, K))
                        ax.set(xlabel='Time (hr)')
                except:
                    continue
        ax1.legend(loc='lower right', title='Soil & sorptivity', frameon=True)
        ax1.set(
            ylim=[0,1],
            title='Distance from new indoor air concentration equilibrium',
            ylabel='$\\frac{\\alpha - \\alpha_0}{\\alpha_{eq} - \\alpha_0}$',
        )
        ax2.set(
            ylim=[0,1],
            title='... flow velocity through crack equilibrium',
            ylabel='$\\frac{u - u_0}{u_{eq} - u_0}$',
        )
        ax3.set(
            ylim=[1e-6, 1],
            title='... soil-gas equilibrium near crack',
            ylabel='$\\frac{c - c_0}{c_{eq} - c_0}$',
        )
        ax3.set(yscale='log')


    return

#Kinetics(file='../../data/adsorption_kinetics.csv',material='drywall').plot()
#soil_adsorption()
#transport_analysis()
#indoor_adsorption()
#indoor_adsorption_zero_entry()
time_to_equilibrium()
#sorption_fit()
plt.show()
