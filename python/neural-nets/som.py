# Self Organizing Map

# Importing Libraries
import numpy as np
import pandas as pd

# Importing the dats set
dataset = pd.read_csv('Credit_Card_Applications.csv')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0, 1))
x = sc.fit_transform(x)

# Training the SOM
from minisom import MiniSom

som = MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.5)
som.random_weights_init(x)
som.train_random(x, num_iteration=100)

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show

bone()
pcolor(som.distance_map().T)  # returns the mean interneuron distances for all winning nodes
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, z in enumerate(x):
    w = som.winner(z)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor=colors[y[i]],
         markerfacecolor='None',
         markersize=10,
         markeredgewidth=2)
show()

# Finding frauds
mappings = som.win_map(x)
frauds = np.concatenate((mappings[(1,7)], mappings[(2,8)]), axis=0)  # Find mapping coords from SOM - this will
frauds = sc.inverse_transform(frauds)                                # change every time you run it

print('Potential Fraudulent Customer IDs')
for i in frauds[:, 0]:
    print(int(i))
