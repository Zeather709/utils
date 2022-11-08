# Recurrent Neural Net - Long Short-Term Memory

# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training data
train = pd.read_csv('Google_Stock_Price_Train.csv')
train_np = train.iloc[:, 1:2].values

# Feature Scaling - normalization works best with sigmoid function
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1), copy=True)  # scaled stock prices will be between 0 and 1
train_scaled = sc.fit_transform(train_np)  # fit calculated max & min, transform scales data

# Create data structure with 60 timesteps and 1 output (3 months of previous stock prices)
x_train = []
y_train = []

for i in range(60, 1258):
    x_train.append(train_scaled[i-60:i, 0])
    y_train.append(train_scaled[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Building the RNN

# Importing the keras libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initializing the RNN & adding neuron layers
rnn = Sequential()
rnn.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))   # Add LSTM layer, return_sequences=True if adding more LSTM layers
rnn.add(Dropout(rate=0.2))      # Dropout layer avoids overfitting, drop 20% of neurons (10) during each iteration
rnn.add(LSTM(units=50, return_sequences=True))  # Second LSTM model, don't need to specify input layer shape
rnn.add(Dropout(rate=0.2))
rnn.add(LSTM(units=50, return_sequences=True))  # Third LSTM layer
rnn.add(Dropout(rate=0.2))
rnn.add(LSTM(units=50))      # return_sequences=False is default
rnn.add(Dropout(rate=0.2))
rnn.add(Dense(units=1))  # units corresponds to dimension of output layer (1 value - stock price)

# Compiling the RNN
rnn.compile(optimizer='adam', loss='mean_squared_error') # use RMSprop or Adam optimizer for RNNs, MSE for regression
# can use loss=tf.keras.losses.MeanAbsoluteError() to calculate relative error
# Fitting the RNN to the training data
rnn.fit(x=x_train, y=y_train, epochs=100, batch_size=32)

# Making predictions

# Getting the stock price
test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = test.iloc[:, 1:2].values

dataset_all = pd.concat((train['Open'], test['Open']), axis=0)
inputs = dataset_all[len(dataset_all) - len(test) - 60:].values  # test data set plus 60 previous days
inputs = inputs.reshape(-1,1)

test_scaled = sc.transform(inputs)      # Must be same scaling as training data set - only need transform, not fit

# Create data structure with 60 previous days
x_test = []
for i in range(60, 80):
    x_test.append(test_scaled[i-60:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_stock_price = rnn.predict(x_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualizing the model performance
plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
plt.plot(predicted_stock_price, color='purple', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time (Days)')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()