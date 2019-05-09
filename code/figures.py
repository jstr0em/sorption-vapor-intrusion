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
        fig, axes = plt.subplots(1,2,sharey=True)

        for ax, side in zip(axes, df['Side'].unique()):

            sns.heatmap(
                df.loc[df['Side']==side][cols].corr(),
                cmap=cmap,
                ax=ax,
                annot=True,
                fmt='1.1f',
            )

            ax.set(
                title=side + ' side',
            )

        plt.show()

        return

meeting_plots = Meeting()
#meeting_plots.heated_vs_unheated()
meeting_plots.correlations()
