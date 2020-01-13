import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from analysis import Analysis, get_sorption_mitigation_data, get_sorption_mitigation_reduction_table
import matplotlib.ticker as mtick
from matplotlib.gridspec import GridSpec

from material import get_all_materials, get_indoor_materials
from mitigation import Mitigation
plt.style.use('seaborn')
def indoor_adsorption():
    # loading all the data
    analysis = Analysis()
    df = analysis.get_indoor_material_data()
    kin = analysis.get_kinetics_data()
    print(kin.sort_values(by='K',ascending=False))
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

    ax1.set(title='Building pressurization', ylabel='$\\Delta p \\; \\mathrm{(Pa)}$', xlabel=xlabel)
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

    materials = ['wood', 'soil', 'cinderblock']
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
        title='Sorption of 1.12 $\\mathrm{ppb_v}$ of %s on cinderblock, soil, and wood with fitted curves' % kin.get_contaminant().upper(),
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

def indoor_adsorption_zero_entry():
    analysis = Analysis()
    df = get_sorption_mitigation_data()
    df2 = get_sorption_mitigation_reduction_table()
    materials = df2['Material'].unique()


    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # indoor material analysis
    # combo plot
    # setting up figure
    fig, (ax1, ax2) = plt.subplots(2,1,dpi=300)
    t0, tau = 0, 24
    c0 = df['c_in'].values[0]
    reduction = 0.5
    c0_red = c0*reduction
    #ax1.plot([t0,tau], [c0_red, c0_red], 'k-')

    for i, material in enumerate(materials):
        color = colors[i]
        df_now = df.loc[material.lower()]
        df_now.plot(y='c_in',ax=ax1, label=material.title(), logy=True, color=color)


    sns.barplot(
        data=df2,
        x='Reduction factor',
        y='Reduction time (hr)',
        hue='Material',
        ax=ax2,
    )
    for p in ax2.patches:
        height = p.get_height()
        ax2.text(
            p.get_x()+p.get_width()/2.,
            height,
            '{:1.1f}'.format(height),
            ha="center",
        )

    ax1.legend(title='Material', frameon=True)
    ax1.set(
        xlabel='Time (hr)',
        ylabel='$c_\\mathrm{in} \\; \\mathrm{(\\mu g/m^3)}$',
        title='Delay in indoor containant concentration reduction, due to contaminant \n desorption from indoor materials, after contaminant entry ceases ',
        ylim=[1e-2, 1e1]
    )

    ax2.set(
        title='Increase in time for indoor contaminant concentration to decrease by a certain factor due to desorption',
        yscale='log',
    )



    plt.tight_layout()
    return

def time_to_equilibrium():
    analysis = Analysis()
    df = analysis.get_time_to_equilibrium_data()
    ss = analysis.get_steady_state_data()

    soils = df.index.levels[0]
    Ks = df.index.levels[1]
    cycles = df.index.levels[2]

    vars = ['alpha', 'c_gas',]
    titles = {
        'Overpressurization': 'Effect of overpressurizing building from -5 to 15 Pa',
        'Depressurization': 'Effect of depressurizing building from -5 to -15 Pa',
    }


    for cycle in cycles:
        # setting up figure
        fig = plt.figure(dpi=300, constrained_layout=True, figsize=(10,6))
        gs = GridSpec(2,2, figure=fig)

        ax1 = fig.add_subplot(gs[0,:])
        ax2 = fig.add_subplot(gs[1,0])
        ax3 = fig.add_subplot(gs[1,1])
        axes = (ax2, ax3)
        #soils = [soils[-1],] # temporary thing
        for soil in soils:
            for K in Ks:
                try:
                    df_now = df.loc[(soil, K, cycle)]
                    p_end = df_now['p_in'].values[-1]
                    ss_now = ss.loc[(soil, 5.28, p_end)]
                    df_now.plot(y='alpha', ax=ax1, legend=False)
                    for ax, var in zip(axes, vars):
                        var_0 = df_now[var].values[0] # intial value
                        var_eq = ss_now[var] # equilibrium values
                        df_now[var+'_distance'] = np.abs(df_now[var]-var_0)/np.abs(var_eq-var_0)
                        df_now.plot(y=var+'_distance', ax=ax, legend=False, label='%s, K = %1.2e' % (soil, K))
                        ax.set(xlabel='Time (hr)')
                except:
                    continue

        ax1.set(
            #ylim=[0,1],
            title='Indoor air concentration over time (as attenuation factor)',
            ylabel='$\\alpha$',
            #yscale='log',
        )
        ax2.set(
            #ylim=[0,1],
            title='Attenuation factor distance from new equilibrium',
            ylabel='$\\frac{\\alpha - \\alpha_0}{\\alpha_{eq} - \\alpha_0}$',
        )
        ax3.set(
            #ylim=[1e-6, 1],
            title='Soil-gas concentration near crack ...',
            ylabel='$\\frac{c_{g} - c_{0,g}}{c_{eq,g} - c_{0,g}}$',
            yscale='log',
        )

        ax1.ticklabel_format(axis='y', style='sci') # TODO: Figure out why this doesn't work
        handles, labels = ax2.get_legend_handles_labels()
        ax1.legend(handles, labels, title='Soil & sorptivity', loc='upper right', frameon=True)
        #ax1.legend([],[],loc='center left',bbox_to_anchor=(1.60,1))
        #fig.legend(handles, labels, title='Soil & sorptivity',loc='center left', bbox_to_anchor=(0.72,0.5))

        fig.suptitle(titles[cycle])

        plt.savefig('../figures/equilibrium_retardation_%s.pdf' % cycle.lower())

    return


def mitigation_time_to_reduction():
    materials = get_indoor_materials()
    reductions = [0.5, 0.1, 0.01, 0.001]
    taus = []
    reds = [] # for storage
    mats = []

    for material in materials:
        x = Mitigation(material=material)
        for reduction in reductions:
            tau = float(x.get_reduction_time(reduction=reduction))
            taus.append(tau)
            reds.append(reduction)
            mats.append(material.title())

    df = pd.DataFrame(
        data={
            'Material': mats,
            'Reduction time (hr)': taus,
            'Reduction factor': reds,
            }
        )


    df.sort_values(
        by=['Reduction factor','Reduction time (hr)'],
        ascending=[False, True],
        inplace=True,
    )
    fig, ax = plt.subplots(dpi=300)
    sns.barplot(
        data=df,
        x='Reduction factor',
        y='Reduction time (hr)',
        hue='Material',
        ax=ax,
    )
    total = len(df)
    for p in ax.patches:
        height = p.get_height()
        ax.text(
            p.get_x()+p.get_width()/2.,
            height,
            '{:1.1f}'.format(height),
            ha="center",
        )

    ax.set(
        title='How contaminant desorption affects the time for indoor air concentration to reduce by a given factor\nafter contaminant entry has ceased.',
        yscale='log',
    )

    print(df.set_index(['Reduction factor', 'Material']))

    ax.legend(loc='upper left', title='Material')
    plt.tight_layout()
    return
# may not be included
#Kinetics(file='../../data/adsorption_kinetics.csv',material='drywall').plot()
#soil_adsorption()

# story
path = '../figures/'
#sorption_fit()
#plt.savefig(path+'sorption_fit.pdf')
#time_to_equilibrium()



#indoor_adsorption()
#plt.savefig(path+'sorption_indoor_cycle.pdf')
indoor_adsorption_zero_entry()
plt.savefig(path+'sorption_mitigation.pdf')
#mitigation_time_to_reduction()
#plt.savefig(path+'sorption_reduction_time.pdf')

plt.show()
