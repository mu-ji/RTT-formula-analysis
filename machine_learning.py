'''
author Manjiang Cao 
e-mail <mcao999@connect.hkust-gz.edu.cn>
this file include functions related to using machine learning solution to predict distance based on RTT
'''

import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def ridge_training(X,Y):
    ridge_model = Ridge()
    ridge_model.fit(X,Y)
    return ridge_model

def lasso_training(X,Y):
    lasso_model = Lasso()
    lasso_model.fit(X,Y)
    return lasso_model

def ridge_predicting(ridge_model,test_x):
    ridge_predictions = ridge_model.predict(test_x)
    return ridge_predictions

def lasso_predicting(lasso_model,test_x):
    lasso_predictions = lasso_model.predict(test_x)
    return lasso_predictions

def data_process_ML(train_set_file):
    with open(train_set_file, 'r') as train_file:
        lines = train_file.readlines()

    X = np.array([0,0,0,0])
    for i in range(len(lines)):
        line = lines[i].strip().split(' ')
        line = [float(num) for num in line]
        
        rtt_mean = np.mean(line[:200])
        rtt_var = np.var(line[:200])
        rssi_mean = np.mean(line[200:])
        rssi_var = np.var(line[200:])
        X = np.vstack((X, np.array([rtt_mean,rtt_var,rssi_mean,rssi_var])))

    X = X[1:,:]  #去掉第一行的0
    return X

def generate_train_test_y():
    distance_list = [i for i in range(1,12)]
    train_y = np.array([0])
    for distance in distance_list:
        for j in range(int(len(train_x)/len(distance_list))):
            train_y = np.vstack((train_y,distance))
    train_y = train_y[1:,:]

    test_y = np.array([0])
    for distance in distance_list:
        for j in range(int(len(test_x)/len(distance_list))):
            test_y = np.vstack((test_y,distance))
    test_y = test_y[1:,:]
    return train_y, test_y


train_x = data_process_ML('train_set/indoor_with_people_walking_train_set.txt')
test_x = data_process_ML('test_set/indoor_with_people_walking_test_set.txt')
train_y,test_y = generate_train_test_y()

ridge_model = ridge_training(train_x,train_y)
ridge_predictions = ridge_predicting(ridge_model,test_x)

lasso_model = lasso_training(train_x,train_y)
lasso_predictions = lasso_predicting(lasso_model,test_x)

ridge_error = ridge_predictions.reshape(20, 11)
lasso_error = lasso_predictions.reshape(20, 11)

boxprops = dict(facecolor='lightblue', color='blue')
plt.boxplot(ridge_error,positions=[i for i in range(1,23,2)],patch_artist=True, boxprops=boxprops)
boxprops = dict(facecolor='red', color='maroon')
plt.boxplot(lasso_error,positions=[i for i in range(2,23,2)],patch_artist=True, boxprops=boxprops)

rect_ridge = plt.Rectangle((0, 0), 1, 1, facecolor='lightblue', edgecolor='blue')
rect_lasso = plt.Rectangle((0, 0), 1, 1, facecolor='red', edgecolor='maroon')
plt.legend([rect_ridge, rect_lasso], ['ridge error', 'lasso error'])

labels = (['{} meters'.format(i) for i in range(1,12)])
plt.xticks([i+0.5 for i in range(1,23,2)], labels)
plt.title('Boxplot of errors in different distance')
plt.ylabel('error(meters)')
plt.show()