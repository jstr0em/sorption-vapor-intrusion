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
    def __init__(self, side='422'):
        self.side = side
        self.db = sqlite3.connect(get_dropbox_path() + '/var/Indianapolis.db')


        df = self.get_data()
        df.to_csv('./data/indianapolis_%s.csv' % side)

        return

    def get_data(self):
        # loads individual dataframes

        indoor = self.get_indoor_air() # this is the basis for all df merging (i.e. we wish to fit all time stamps to these)
        ssd = self.get_ssd_status()
        subslab = self.get_subslab() # special because we need to match species

        df = pd.merge_asof(indoor, subslab, left_index=True, right_index=True, by='Contaminant')
        df = df.loc[(df.index.date>=ssd.index.date.min()) & (df.index.date<=ssd.index.date.max())]

        # tuples of dataframes/data to fit to our indoor concentration data
        dfs_to_merge = (
            self.get_meteorological_data(),
            self.get_pressure(),
            self.get_obs_status(),
            self.get_soil_temp(),
        )

        for _ in dfs_to_merge:
            df = pd.merge_asof(df, _, left_index=True, right_index=True,)


        # new columns/calculations
        df['AttenuationSubslab'] = df['IndoorConcentration']/df['SubslabConcentration']
        df['AttenuationSubslab'] = df['AttenuationSubslab'].replace([-np.inf,np.inf], np.nan)
        df['logIndoorConcentration'] = df['IndoorConcentration'].apply(np.log10).replace([-np.inf,np.inf], np.nan)
        df['logAttenuationSubslab'] = df['AttenuationSubslab'].apply(np.log10).replace([-np.inf,np.inf], np.nan)
        df['Season'] = df.index.month.map(get_season)

        return df

    def process_time(self, df, reset_index=True):
        df = df.assign(Time=lambda x: x['StopDate']+' '+x['StopTime'])
        df.drop(columns=['StopDate','StopTime'],inplace=True)

        df['Time'] = df['Time'].apply(pd.to_datetime)
        df.sort_values(by=['Time'],inplace=True)

        if reset_index is True:
            df.set_index('Time',inplace=True)

        return df
    def get_meteorological_data(self):

        query = " \
            SELECT \
                StopTime, StopDate, Variable, Value \
            FROM \
                Meteorological_Data \
            WHERE \
                Variable = 'Rain' OR \
                Variable = 'Wind.Speed' OR \
                Variable = 'Temp.Out' OR \
                Variable = 'In.Temp' OR \
                Variable = 'Bar..' OR \
                Variable = 'Out.Hum' OR \
                Variable = 'In.Hum' \
         ;"

        df = pd.read_sql_query(
            query,
            self.db,
        )

        df = self.process_time(df)
        df = df.pivot( columns='Variable', values='Value')

        f_to_c = lambda x: (x-32)/1.8

        df['Temp.Out'] = df['Temp.Out'].apply(f_to_c)
        df['In.Temp'] = df['In.Temp'].apply(f_to_c)
        df['Bar..'] *= 3386.389
        df['Rain'] *= 2.54

        df.rename(
            columns={
                'Temp.Out': 'OutdoorTemp',
                'In.Temp': 'IndoorTemp',
                'Bar..': 'BarometricPressure',
                'Wind.Speed': 'WindSpeed',
                'Out.Hum': 'OutdoorHumidity',
                'In.Hum': 'IndoorHumidity',
            },
            inplace=True,
        )

        return df
    def get_pressure(self):
        query = " \
            SELECT \
                StopDate, \
                StopTime, \
                Variable, \
                Value \
            FROM \
                Differential_Pressure_Data \
            WHERE \
                Location = '%s' AND \
                (Variable = 'Basement.Vs.Exterior' OR Variable = 'SubSlab.Vs.Basement') \
        ;" % self.side

        df = pd.read_sql_query(query, self.db)
        df = self.process_time(df, reset_index=False)

        df = df.pivot_table(index='Time', columns='Variable', values='Value')

        rename = {
            'Basement.Vs.Exterior': 'IndoorOutdoorPressure',
            'SubSlab.Vs.Basement': 'SubslabPressure',
        }

        df.rename(columns=rename, inplace=True)

        return df
    def get_ssd_status(self):
        query = "\
            SELECT \
                StopDate, \
                StopTime, \
                Value AS Mitigation \
            FROM \
                Observation_Status_Data \
            WHERE \
                Mitigation = 'not yet installed' \
        ;"

        ssd = pd.read_sql_query(query, self.db)

        ssd = self.process_time(ssd)
        return ssd
    def get_indoor_air(self):
        query = "\
            SELECT \
                StopDate, \
                StopTime, \
                Variable AS Contaminant, \
                Value AS IndoorConcentration \
            FROM \
                VOC_Data_SRI_8610_Onsite_GC \
            WHERE \
                Location = '%sBaseS' AND \
                Depth_ft = 'Basement' \
        ;" % self.side

        indoor = pd.read_sql_query(query, self.db)
        indoor = self.process_time(indoor)

        return indoor
    def get_subslab(self):
        # port selection based on building side
        ports = {
            '420': 'SSP-7',
            '422': 'SSP-4',
        }

        query = "\
            SELECT \
                StopDate, \
                StopTime, \
                Variable AS Contaminant, \
                Value AS SubslabConcentration \
            FROM \
                VOC_Data_SRI_8610_Onsite_GC \
            WHERE \
                Location = '%s' \
        ;" % ports[self.side]

        subslab = pd.read_sql_query(query, self.db)
        subslab = self.process_time(subslab)
        return subslab
    def get_obs_status(self):
        query = " \
            SELECT \
                StopDate, StopTime, Variable, Value \
            FROM \
                Observation_Status_Data \
            WHERE \
                (Variable = 'AC' AND Location = '422') OR \
                Variable = 'SnowDepth' OR \
                (Variable = 'Heating' AND Location = '422') \
        ;"

        df = pd.read_sql_query(
            query,
            self.db,
        )

        df = self.process_time(df)

        df = df.pivot(columns='Variable', values='Value')

        return df

    def get_soil_temp(self):
        query = "\
            SELECT \
                StopDate, \
                StopTime, \
                Value AS SoilTemp, \
                Depth_ft AS Depth \
            FROM \
                Soil_Temperature_Data \
            WHERE \
                Location = 'MW3'\
        ;"

        df = pd.read_sql_query(query, self.db)
        df = self.process_time(df, reset_index=False)
        df['Depth'] *= 0.3048

        df = df.pivot_table(index='Time', columns='Depth', values='SoilTemp')
        rename = {}
        for _ in list(df):
            rename[_] = 'SoilTempDepth%1.1f' % _

        df.rename(columns=rename, inplace=True)

        return df
Indianapolis(side='422')
Indianapolis(side='420')
