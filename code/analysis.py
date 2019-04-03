import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_log_ticks(start, stop, style='e'):

    ticks = np.array([])
    ints = np.arange(np.floor(start),np.ceil(stop)+1)
    for int_now in ints:
        ticks = np.append(ticks, np.arange(0.1,1.1,0.1)*10.0**int_now)
        ticks = np.unique(ticks)


    if style=='e':
        labels = ['%1.1e' % tick for tick in ticks]
        ticks_to_keep = ['%1.1e' % 10**int for int in ints]
    elif style=='f':
        labels = ['%1.12f' % tick for tick in ticks]
        ticks_to_keep = ['%1.12f' % 10**int for int in ints]

    ticks = np.log10(ticks)

    labels = list(map(lambda x: x.rstrip('0'), labels))
    ticks_to_keep = list(map(lambda x: x.rstrip('0'), ticks_to_keep))

    for i, label in enumerate(labels):

        if label in ticks_to_keep:
            #print('Not removing label')
            continue
        else:
            #print('Removing label')
            labels[i] = ' '
    # TODO: add more logic to keep first zero after . i.e. 1. -> 1.0
    return ticks, labels

class Season:
    def __init__(self):

        data = pd.read_csv('./data/indianapolis.csv')

        g = sns.boxplot(
            x="Season",
            y="logIndoorConcentration",
            hue="Specie",
            data=data,
        )

        ticks, labels = get_log_ticks(-1,1,'f')
        g.axes.set(
            yticks=ticks,
            yticklabels=labels,
        )

        plt.savefig('./figures/analysis/season.png', dpi=300)
        plt.savefig('./figures/analysis/season.pdf', dpi=300)

        #plt.show()
        return

class Snow:
    def __init__(self):

        data = pd.read_csv('./data/indianapolis.csv')
        data = data.loc[(data['Specie']=='Chloroform')]

        g = sns.boxplot(
            x="SnowDepth",
            y="logIndoorConcentration",
            hue="Season",
            data=data,
        )

        ticks, labels = get_log_ticks(-1,1,'f')
        g.axes.set(
            yticks=ticks,
            yticklabels=labels,
        )
        plt.savefig('./figures/analysis/snowdepth.png', dpi=300)
        plt.savefig('./figures/analysis/snowdepth.pdf', dpi=300)

        #plt.show()
        return

class AC:
    def __init__(self):

        data = pd.read_csv('./data/indianapolis.csv')
        data = data.loc[ (data['Specie']=='Chloroform')]
        g = sns.boxplot(
            x="AC",
            y="logIndoorConcentration",
            hue="Season",
            data=data,
        )

        ticks, labels = get_log_ticks(-1,1,'f')
        g.axes.set(
            yticks=ticks,
            yticklabels=labels,
        )
        plt.savefig('./figures/analysis/ac.png', dpi=300)
        plt.savefig('./figures/analysis/ac.pdf', dpi=300)

        #plt.show()
        return

class Heating:
    def __init__(self):

        data = pd.read_csv('./data/indianapolis.csv')
        data = data.loc[(data['Specie']=='Chloroform')]

        g = sns.boxplot(
            x="Heating",
            y="logIndoorConcentration",
            hue="Season",
            data=data,
        )

        ticks, labels = get_log_ticks(-1,1,'f')
        g.axes.set(
            yticks=ticks,
            yticklabels=labels,
        )
        plt.savefig('./figures/analysis/heating.png', dpi=300)
        plt.savefig('./figures/analysis/heating.pdf', dpi=300)

        #plt.show()
        return

class OutdoorTemp:
    def __init__(self):

        data = pd.read_csv('./data/indianapolis.csv')
        data = data.loc[(data['Specie']=='Chloroform')]


        g = sns.regplot(
            x="OutdoorTemp",
            y="logIndoorConcentration",
            #hue="Specie",
            data=data,
        )

        ticks, labels = get_log_ticks(-1,1,'f')
        g.axes.set(
            yticks=ticks,
            yticklabels=labels,
        )
        plt.savefig('./figures/analysis/outdoor_temp.png', dpi=300)
        plt.savefig('./figures/analysis/outdoor_temp.pdf', dpi=300)

        #plt.show()
        return

class Correlations:
    def __init__(self):

        data = pd.read_csv('./data/indianapolis.csv')
        data = data.loc[(data['Specie']=='Chloroform')]

        print(list(data))

        cols = [
            'logIndoorConcentration',
            'IndoorOutdoorPressure',
            'OutdoorTemp',
            'Rain',
            'WindSpeed',
            'IndoorHumidity',
            'OutdoorHumidity',
            'SnowDepth',
            'BarometricPressure',
        ]


        corr = data[cols].corr()

        fig, ax = plt.subplots(dpi=300)
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(
            corr,
            cmap=cmap,
            ax=ax,
            annot=True,
            fmt='1.1f',
        )
        plt.tight_layout()
        plt.show()
        return

class DiurnalTemp:
    def __init__(self):
        data = pd.read_csv('./data/indianapolis.csv')
        #data = data.loc[(data['Specie']=='Chloroform')]
        data['Time'] = data['Time'].apply(pd.to_datetime)

        fig, ax1 = plt.subplots(dpi=300)

        ax2 = ax1.twinx()

        for season in data['Season'].unique():

            diurnal = data.loc[data['Season']==season][['Time','OutdoorTemp','logIndoorConcentration']].groupby(data['Time'].dt.hour).median()
            diurnal.plot(y='OutdoorTemp',label=season, ax=ax1)
            diurnal.plot(y='logIndoorConcentration', ax=ax2, style='--', legend=False)


        ax1.legend()
        plt.show()
        return

class TempCorrelation:

    def __init__(self):
        data = pd.read_csv('./data/indianapolis.csv').dropna()
        g = sns.PairGrid(
            data[['IndoorConcentration','OutdoorTemp','OutdoorHumidity','IndoorHumidity','Season']],
            hue='Season',
            diag_sharey=False,
        )

        g.map_diag(sns.kdeplot, shade=True)

        plt.show()
        return



#Season()
#Snow()
#AC()
#Heating()
#OutdoorTemp()
#Correlations()
#DiurnalTemp()
TempCorrelation()
