import numpy as np
import pandas as pd
import sqlite3
import seaborn as sns
import os
import matplotlib.pyplot as plt

dropbox_folder = None

def get_dropbox_path():
    for dirname, dirnames, filenames in os.walk(os.path.expanduser('~')):
        for subdirname in dirnames:
            if(subdirname == 'Dropbox'):
                dropbox_folder = os.path.join(dirname, subdirname)
                break
        if dropbox_folder:
            break
    return dropbox_folder

# TODO: create this function to intelligently find infs
def replace_inf(df):
    df.replace([-np.inf,np.inf], np.nan)
    return

def get_season(x):
    seasons = {
        'Winter': (12, 2),
        'Spring': (3, 5),
        'Summer': (6, 8),
        'Fall': (9, 11),
    }
    if (x == 12) or (x == 1) or (x == 2):
        return 'Winter'
    elif (x == 3) or (x == 4) or (x == 5):
        return 'Spring'
    elif (x == 6) or (x == 7) or (x == 8):
        return 'Summer'
    elif (x == 9) or (x == 10) or (x == 11):
        return 'Fall'
    else:
        return 'Error'

class Indianapolis:
    def __init__(self):

        self.db = sqlite3.connect(get_dropbox_path() + '/var/Indianapolis.db')
        #df = self.get_data()
        #df.to_csv('./data/indianapolis.csv')
        self.get_meteorological_data()
        return
    def process_time(self,df):
        df = df.assign(Time=lambda x: x['StopDate']+' '+x['StopTime'])
        df.drop(columns=['StopDate','StopTime'],inplace=True)

        df['Time'] = df['Time'].apply(pd.to_datetime)
        df.sort_values(by=['Time'],inplace=True)
        #df.set_index('Time',inplace=True)

        return df
    def get_meteorological_data(self):

        query = " \
            SELECT \
                StopTime, StopDate, Variable, Value \
            FROM \
                Meteorological_Data \
            WHERE \
                Variable = 'Rain' OR Variable = 'Wind.Speed' OR Variable = 'Temp.Out' OR Variable = 'Bar..' \
         ;"

        df = pd.read_sql_query(
            query,
            self.db,
        )

        df = self.process_time(df)
        df = df.pivot(index='Time', columns='Variable', values='Value')

        df['Temp.Out'] = df['Temp.Out'].apply(lambda x: (x-32)/1.8)
        df['Bar..'] *= 3386.389
        df['Rain'] *= 2.54

        df.plot()
        plt.show()
        return df


Indianapolis()
