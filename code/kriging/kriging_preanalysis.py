import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from kriging import Data
from utils import get_dropbox_path
import sqlite3
import statsmodels.api as sm

#data = Data().get_data(contaminant='Chloroform')


db = sqlite3.connect(get_dropbox_path() + '/var/Indianapolis.db')
# query
query = " \
    SELECT \
        Value AS Concentration \
    FROM \
        VOC_Data_SoilGas_and_Air \
    WHERE \
        Variable = 'Chloroform'\
;"
# read data from database
df = pd.read_sql_query(query, db)

print(list(df))

df['Concentration'] = df['Concentration'].apply(np.log10)
import numpy as np
import matplotlib.pyplot as plt


sm.qqplot(df['Concentration'], line='45')
plt.show()

"""
df.plot(
kind='hist'
)

plt.title('Skew = %1.3f, Kurtosis = %1.3f' % (df.skew(), df.kurtosis()))
plt.show()
"""
