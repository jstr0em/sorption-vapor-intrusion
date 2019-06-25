import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
from utils import get_dropbox_path


db = sqlite3.connect(get_dropbox_path() + '/var/Indianapolis.db')


query = " \
    SELECT \
        StopDate, StopTime, Value, Location, Depth_ft \
    FROM \
        Soil_Temperature_Data; \
"

df = pd.read_sql_query( query, db, )
df = df.assign(Time=lambda x: x['StopDate']+' '+x['StopTime'])
df['Depth'] = df['Depth_ft']*0.334
df.drop(columns=['StopDate','StopTime','Depth_ft'],inplace=True)


locations = df['Location'].unique()
depths = df['Depth'].unique()

df = df.pivot_table(index='Time',columns=['Location', 'Depth'], values='Value')
df = df.interpolate()



query2 = " \
    SELECT \
        StopDate, StopTime, Value AS Temperature \
    FROM \
        Meteorological_Data \
    WHERE \
        Variable = 'Temp.Out' \
"

df2 = pd.read_sql_query( query2, db, )
df2 = df2.assign(Time=lambda x: x['StopDate']+' '+x['StopTime'])
df2.drop(columns=['StopDate','StopTime'],inplace=True)
df2['Temperature'] = df2['Temperature'].apply(lambda x: (x-32)/1.8)

fig, ax = plt.subplots(dpi=300)

df2.plot(label='Outdoor temp', ax=ax)

for depth in depths:
    try:
        df[(locations[1], depth)].plot(label=depth,ax=ax)
    except:
        continue


plt.xticks(rotation=45)

ax.legend()
plt.tight_layout()
plt.show()
