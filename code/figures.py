import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from utils import get_log_ticks
import matplotlib.gridspec as gridspec

class Meeting:
    def __init__(self):

        self.data = pd.read_csv('./data/indianapolis.csv')
        self.data['Time'] = self.data['Time'].apply(pd.to_datetime)

        return

    def heated_vs_unheated(self):

        df = self.data
        df = df[df['Contaminant']=='Chloroform']

        fig = plt.figure()

        gs = gridspec.GridSpec(2,2)

        ax1 = plt.subplot(gs[0,:])
        ax2 = plt.subplot(gs[1,0])
        ax3 = plt.subplot(gs[1,1])

        # time series figure
        sns.lineplot(
            x='Time',
            y='IndoorConcentration',
            hue='Side',
            data=df,
            ax=ax1,
        )

        ax1.set(
            title='Chloroform concentration in the basement of two duplex sides',
            ylabel='$\\mathrm{c_{in}} \; \\mathrm{(\\mu g/L)}$',
            yscale='log',
        )

        # concentrations boxplots
        sns.boxplot(
            x='Season',
            y='logIndoorConcentration',
            hue='Side',
            data=df,
            ax=ax2,
            whis=10,
        )

        ticks, labels = get_log_ticks(-1,0.5, 'f')

        ax2.set(
            yticks=ticks,
            yticklabels=labels,
            title='Seasonal concentration\ndistributions',
            ylabel='$\\mathrm{c_{in}} \; \\mathrm{(\\mu g/L)}$',
        )
        ax2.legend().remove()

        sns.boxplot(
            x='Season',
            y='SubslabPressure',
            hue='Side',
            data=df,
            ax=ax3,
            whis=1000,
        )

        ax3.set(
            ylabel='$\\mathrm{\\Delta p_{subslab}} \; (Pa)$',
            title='Seasonal subslab pressure\ndifference distributions',
            ylim=[-2,5],
        )
        ax3.legend().remove()

        plt.tight_layout()
        dpi = 500
        plt.savefig('./figures/heated_vs_unheated.pdf', dpi=dpi)
        plt.savefig('./figures/heated_vs_unheated.png', dpi=dpi)


        return

    def correlations(self):

        cols = [
            'IndoorConcentration',
            'SubslabPressure',
            'OutdoorTemp',
            'Rain',
            'WindSpeed',
            'IndoorHumidity',
            'OutdoorHumidity',
            'SnowDepth',
            'BarometricPressure',
        ]


        df = self.data
        df = df[df['Contaminant']=='Chloroform']

        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        fig, (ax1, ax2) = plt.subplots(1,2,sharey=True,figsize=(10,5))


        sns.heatmap(
            df.loc[df['Side']=='Unheated'][cols].corr(),
            cmap=cmap,
            cbar=False,
            ax=ax1,
            annot=True,
            fmt='1.1f',
        )

        ax1.set(
            title='Unheated',
        )
        ax1.tick_params(axis='x', rotation=45)

        sns.heatmap(
            df.loc[df['Side']=='Heated'][cols].corr(),
            cmap=cmap,
            ax=ax2,
            annot=True,
            fmt='1.1f',
        )

        ax2.set(
            title='Heated',
        )
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        dpi = 500
        plt.savefig('./figures/correlation_heatmap.pdf', dpi=dpi)
        plt.savefig('./figures/correlation_heatmap.png', dpi=dpi)

        return

    def pressure(self):
        df = self.data
        df = df[(df['Contaminant']=='Chloroform') & (df['Side']=='Heated')]

        g = sns.PairGrid(
            df,
            x_vars=['BarometricPressure','WindSpeed','OutdoorTemp','OutdoorHumidity'],
            y_vars=['IndoorConcentration','IndoorOutdoorPressure','IndoorTemp','IndoorHumidity'],
            diag_sharey=False,
        )

        g.map(sns.regplot, x_bins=10)

        plt.tight_layout()
        dpi = 500
        plt.savefig('./figures/outdoor_factor_pairgrid.pdf', dpi=dpi)
        plt.savefig('./figures/outdoor_factor_pairgrid.png', dpi=dpi)

        return

meeting_plots = Meeting()
#meeting_plots.heated_vs_unheated()
#meeting_plots.correlations()
meeting_plots.pressure()
