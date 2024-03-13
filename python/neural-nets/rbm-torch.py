import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.parallel
import torch.utils.data
import os

print(os.getcwd())
# sys.path.append('/home/zeather/repos/utils/python/neural-nets/')

# !wget "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
# !unzip ml-1m.zip
# !ls
# !wget "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
# !unzip ml-100k.zip
# !ls

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


# Convert data from long format to array-like list of lists (to be converted to tensors)
def df2matrix(data):  # converts long data to list of lists with obs in rows and features in columns
    data_out = []
    for id_users in range(1, n_users + 1):
        id_movies = data[:, 1][data[:, 0] == id_users]
        id_ratings = data[:, 2][data[:, 0] == id_users]
        ratings = np.zeros(n_movies)
        ratings[id_movies - 1] = id_ratings  # subtract 1 b/c index for id_movies starts at 1 & ratings starts at 0
        data_out.append(list(ratings))  # torch expects list of lists, not numpy array
    return data_out


train = df2matrix(train)
test = df2matrix(test)

# Convert data into Torch tensors from list of lists
train = torch.FloatTensor(train)
test = torch.FloatTensor(test)

# Convert 5-star ratings into binary rating (1 - liked, 0 - not liked)
train[train == 0] = -1
train[train == 1] = 0
train[train == 2] = 0
train[train >= 3] = 1

test[test == 0] = -1
test[test == 1] = 0
test[test == 2] = 0
test[test >= 3] = 1


# Create architecture of the neural net
# RBM is a probabilistic graphical model

class RBM:
    def __init__(self, n_vis, n_hidden):  # visible & hidden nodes
        self.W = torch.randn(n_hidden, n_vis)  # Initialize weights - random values normal distribution
        self.a = torch.randn(1, n_hidden)  # a is bias for visible nodes, pytorch expects 2D array, p_h given v
        self.b = torch.randn(1, n_vis)  # bias for hidden nodes, p_v given h

    def sample_hidden(self, x):  # x corresponds to visible neurons v in the probabilities p_h given v
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)  # Apply bias to each line of the batch (weights + bias), probability
        # that hidden node will be activated given the value of the visible node
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)

    def sample_vis(self, y):  # Calculate p of v given h, predict which values in vis nodes equal 1
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)

    def train(self, v0, vk, ph0, phk):  # Train model, contrastive divergence - approx log likelihood gradient
        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()  # Update weights based on input v0 and
                                                                       # probability of v after k iterations
        self.b += torch.sum((v0 - vk), 0)  # Keep tensor in 2 dimensions
        self.a += torch.sum((ph0 - phk), 0)


# Setting up the RBM
nv = len(train[0])
nh = 100
batch_size = 100
rbm = RBM(nv, nh)

# Training the RBM
n_epoch = 10
for epoch in range(1, n_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(0, n_users - batch_size, batch_size):  # Batch training
        vk = train[id_user:id_user + batch_size]
        v0 = train[id_user:id_user + batch_size]
        ph0, _ = rbm.sample_hidden(v0)
        for k in range(10):  # Contrastive divergence over k steps
            _, hk = rbm.sample_hidden(vk)  # Update vk, not v0 (target, will be used to calculate loss/error)
            _, vk = rbm.sample_vis(hk)
            vk[v0 < 0] = v0[v0 < 0]  # All -1 ratings stay as -1 (no training on movies missing ratings)
        phk, _ = rbm.sample_hidden(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0 >= 0] - vk[v0 >= 0]))  # average error
        # train_loss += np.sqrt(torch.mean((v0[v0>=0] - vk[v0>=0])**2)) # RMSE
        s += 1.
    print('epoch: ' + str(epoch) + ' loss: ' + str(train_loss / s))

# Testing RBM performance
test_loss = 0
s = 0.
for id_user in range(n_users):  # MCMC technique (blind walk - like random walk but probs not the same)
    v = train[id_user:id_user + 1]  # v is input, keep training set - this will be used to activate neurons of RBM...
    vt = test[id_user:id_user + 1]  # vt is target, ...to predict ratings of movies in test set
    if len(vt[vt >= 0]) > 0:  # Only make predictions for movies which have ratings in target
        _, h = rbm.sample_hidden(v)  # Update vk, not v0 (target, will be used to calculate loss/error)
        _, v = rbm.sample_vis(h)
        test_loss += torch.mean(torch.abs(vt[vt >= 0] - v[vt >= 0]))  # Only compare to existing ratings in vt
        # test_loss += np.sqrt(torch.mean((vt[vt>=0] - v[vt>=0])**2)) # RMSE
        s += 1.  # Normalize test loss
print('test loss: ' + str(test_loss / s))
