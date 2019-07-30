import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./data/indianapolis_probes_raw.csv')


y_shift = 26.451 # shift to reverse y-locations
df['y'] = y_shift - df['y']


df.to_csv('./data/indianapolis_probes.csv', index=False)
