import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

df = pd.read_csv('estimated_realistic_distribution.csv')
x = df['day_number'].values
h = df['probability'].values
y = np.cumsum(h)
_, ax = plt.subplots(1, 1, figsize=(7,7))
ax.plot(x, y)
ax.set_title('CDF of the estimated real distribution')
ax.set_xlabel('Day of the year [d]')
ax.set_ylabel('P(X<d)')
plt.show()
