'''
author Manjiang Cao 
e-mail <mcao999@connect.hkust-gz.edu.cn>
this file include functions related to using neural network and GMM to predict distance based on RTT
'''

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.mixture import GaussianMixture


class RegressionNet(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(RegressionNet, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.elu = nn.ELU()
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.elu = nn.ELU()
            self.fc3 = nn.Linear(hidden_size, output_size)



        def forward(self, x): 
            out = self.fc1(x)
            out = self.elu(out)
            out = self.fc2(out)
            out = self.elu(out)
            out = self.fc3(out)
            return out
        
def neural_network_with_GMM_train(X,Y,sample_times):

    X = torch.from_numpy(X[:,:]).float()

    Y = torch.from_numpy(Y.reshape(11*sample_times,1)).float()
    #test_x = torch.from_numpy(test_x[:,:]).float()
    #test_y = torch.from_numpy(test_y.reshape((15,1))).float()

    input_size = 10
    hidden_size = 256
    output_size = 1

    GMM_model = RegressionNet(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(GMM_model.parameters(), lr=0.001)

    num_epochs = 1000
    train_loss = []
    test_loss = []
    for epoch in range(num_epochs):
        # 前向传播
        outputs = GMM_model(X)
        loss = criterion(outputs, Y) #+ 0.0001*data_loss(X,Y,outputs)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 打印训练过程中的损失
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        train_loss.append(loss.item())

        #predictions = model(test_x)
        #tt_loss = criterion(predictions,test_y)
        #test_loss.append(tt_loss.item())

    return GMM_model

def neural_network_predicting(model,test_x):
    # 在测试集上进行预测
    with torch.no_grad():
        predictions = model(test_x)

    predictions = predictions.numpy()
    return predictions

def neural_network_without_GMM_train(X,Y,sample_times):
    X = torch.from_numpy(X[:,6:]).float()

    Y = torch.from_numpy(Y.reshape(11*sample_times,1)).float()
    #test_x = torch.from_numpy(test_x[:,6:]).float()
    #test_y = torch.from_numpy(test_y.reshape((15,1))).float()

    input_size = 4
    hidden_size = 256
    output_size = 1

    model = RegressionNet(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 1000
    train_loss = []
    test_loss = []
    for epoch in range(num_epochs):
        # 前向传播
        outputs = model(X)
        loss = criterion(outputs, Y) #+ 0.0001*data_loss(X,Y,outputs)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 打印训练过程中的损失
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        train_loss.append(loss.item())

        #predictions = model(test_x)
        #tt_loss = criterion(predictions,test_y)
        #test_loss.append(tt_loss.item())

    return model
    
def neural_network_predicting(model,test_x):
    # 在测试集上进行预测
    with torch.no_grad():
        predictions = model(test_x)

    predictions = predictions.numpy()
    return predictions
