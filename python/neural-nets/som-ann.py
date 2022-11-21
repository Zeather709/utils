# Mega Case Study - Making a hybrid Deep Learning Model

# Part 1 - SOM

# Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf

# Importing the data set
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
frauds = np.concatenate((mappings[(3, 8)], mappings[(2, 8)]), axis=0)  # Find mapping coords from SOM - this will
frauds = sc.inverse_transform(frauds)  # change every time you run it

print('Potential Fraudulent Customer IDs')
for i in frauds[:, 0]:
    print(int(i))

# Part 2 - ANN

# Create Matrix of Features
customers = dataset.iloc[:, 1:].values

# Create Dependent Variable
is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
    if dataset.iloc[i, 0] in frauds:
        is_fraud[i] = 1

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
customers = sc.fit_transform(customers)

# Building the ANN
# Initializing the ANN
ann = tf.keras.models.Sequential()
# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=2, activation='relu', input_dim=15))
# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
# Compiling the ANN
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the ANN
# Training the ANN on the Training set
ann.fit(customers, is_fraud, batch_size=1, epochs=5)

# Making the predictions and evaluating the model
# Predict probabilities of fraud
y_pred = ann.predict(customers)
y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis=1)
y_pred = y_pred[y_pred[:,1].argsort()]