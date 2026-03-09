import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



x = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']
labels = ['1e-05', '0.0001', '0.001', '0.01', '0.1']

df = pd.read_csv('Grid_Search_probability/learning_rate.csv')

values = [1,6,11,16,21]
experiments = [val for val in range(10)]

for ind_i, i in enumerate(values):
    y = []
    for row in experiments:

        y.append(np.average(df.iloc[row, i:i+5].values))

    plt.plot(x, y, label=labels[ind_i])


plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.savefig('lr_probability.png')