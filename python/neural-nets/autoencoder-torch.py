# AutoEncoder

# Import Libraries
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

#%%
# Import the data set
movies = pd.read_csv('ml-1m/movies.dat', sep='::', header=None, engine='python', encoding='latin-1')
users = pd.read_csv('ml-1m/users.dat', sep='::', header=None, engine='python', encoding='latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', header=None, engine='python', encoding='latin-1')

# Prepare training and testing data sets
train = pd.read_csv('ml-100k/u1.base', delimiter='\t')
train = np.array(train, dtype='int')
test = pd.read_csv('ml-100k/u1.test', delimiter='\t')
test = np.array(test, dtype='int')

# Get total number of observations (rows) and features (columns)
n_users = int(max(max(train[:, 0]), max(test[:, 0])))  # number of users
n_movies = int(max(max(train[:, 1]), max(test[:, 1])))  # number of movies
#%%
# Convert data from long format to array-like list of lists (to be converted to tensors)
def df2matrix(data):
    new_data = []
    for i in range(1, n_users + 1):
        id_movies = data[:,1][data[:,0]==i]
        id_ratings = data[:,2][data[:,0]==i]
        ratings = np.zeros(n_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data
#%%
train = df2matrix(train)
test = df2matrix(test)
#%%
# Convert data to torch tensors
train = torch.FloatTensor(train)
test = torch.FloatTensor(test)
#%%
# Creating the architecture of the neural net
class SAE(nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__() # Get inherited classes and methods of parent class (nn.Module)
        self.fc1 = nn.Linear(n_movies, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, n_movies)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x

#%%
# Create object of stacked autoencoder class
sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)
#%%
# Training the SAE
n_epochs = 200
for e in range(1, n_epochs + 1):
    train_loss = 0
    s = 0.
    for i in range(n_users):
        input = Variable(train[i]).unsqueeze(0)
        target = input.clone()
        if torch.sum(target.data > 0) > 0:
            output = sae.forward(input)
            target.require_grad = False
            output[target == 0] = 0  # Values without ratings in input will not be considered in calculating loss
            loss = criterion(output, target)
            mean_corrector = n_movies/float(torch.sum(target.data > 0) + 1e-10)  # Make sure denominator is never 0
            loss.backward()
            train_loss += np.sqrt(loss.data*mean_corrector)
            s += 1.
            optimizer.step()
    print('Epoch: ' + str(e) + ' Loss: ' + str(train_loss/s))
#%%
# Testing the SAE
test_loss = 0
s = 0.
for i in range(n_users):
    input = Variable(train[i]).unsqueeze(0)  # Has to be training set because movies without ratings are predicted based on existing ratings
    target = Variable(test[i]).unsqueeze(0)
    if torch.sum(target.data > 0) > 0:
        output = sae(input)
        target.require_grad = False
        output[target == 0] = 0
        loss = criterion(output, target)
        mean_corrector = n_movies/float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.data*mean_corrector)
        s += 1.
print('Test Loss: ' + str(test_loss/s))

#%%
