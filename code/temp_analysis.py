import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('./data/temp_sim.csv')


fig, ax = plt.subplots()



sns.lineplot(
    data=df,
    x='T',
    y='C',
    hue='Soil',
    style='Sub-base',
    ax=ax,
)

ax.set(
    yscale='log',
)

plt.show()
