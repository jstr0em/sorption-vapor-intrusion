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

        plt.show()
        return

class Snow:
    def __init__(self):

        data = pd.read_csv('./data/indianapolis.csv')
        data = data.loc[data['Season']=='Winter']

        g = sns.boxplot(
            x="SnowDepth",
            y="logIndoorConcentration",
            #hue="Specie",
            data=data,
        )

        ticks, labels = get_log_ticks(-1,1,'f')
        g.axes.set(
            yticks=ticks,
            yticklabels=labels,
        )

        plt.show()
        return

class AC:
    def __init__(self):

        data = pd.read_csv('./data/indianapolis.csv')

        g = sns.boxplot(
            x="AC",
            y="logIndoorConcentration",
            #hue="Specie",
            data=data,
        )

        ticks, labels = get_log_ticks(-1,1,'f')
        g.axes.set(
            yticks=ticks,
            yticklabels=labels,
        )

        plt.show()
        return

class Heating:
    def __init__(self):

        data = pd.read_csv('./data/indianapolis.csv')

        g = sns.boxplot(
            x="Heating",
            y="logIndoorConcentration",
            #hue="Specie",
            data=data,
        )

        ticks, labels = get_log_ticks(-1,1,'f')
        g.axes.set(
            yticks=ticks,
            yticklabels=labels,
        )

        plt.show()
        return

class OutdoorTemp:
    def __init__(self):

        data = pd.read_csv('./data/indianapolis.csv')

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

        plt.show()
        return
#Season()
#Snow()
#AC()
#Heating()
OutdoorTemp()
