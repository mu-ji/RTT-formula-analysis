'''
author Manjiang Cao 
e-mail <mcao999@connect.hkust-gz.edu.cn>
this script using to estimate parameters in no multi path situation formula
'''

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

wired_signal_speed = 222222222.222       #4.5ns/meter
adapter_length = 0.1

distance_list = [i for i in range(1,12)]

def read_data(distance):
    f = open ('wiredexperiment\distance{}.txt'.format(distance), 'r')
    time_list = []
    data = f.readlines()
    for i in range(len(data)):
        time_list.append(int(data[i].split(' ')[0]))
    time = np.array(time_list)
    return time

def negative_log_likelihood(params, data):
    m, Q, T_los, T_waiting = params
    log_likelihood = 0.0
    for x in data:
        log_likelihood += -0.5 * np.log(2 * np.pi * Q**2) - ((x - T_los - T_waiting - m)**2) / (2 * Q**2)
    return -log_likelihood

def maximum_likelihood_estimation_m_Q(distance,data):
    T_los = (distance + adapter_length)*2/wired_signal_speed*16000000
    T_waiting = 16000

    initial_guess = [np.mean(data)-T_los-T_waiting-10, np.var(data)*0.5-1, T_los, T_waiting]

    result = minimize(negative_log_likelihood, initial_guess, args=(data,), method='L-BFGS-B')

    estimated_m = result.x[0]
    estimated_Q = result.x[1]

    return estimated_m,estimated_Q

data1 = read_data(1)

estimate_m, estimate_Q = maximum_likelihood_estimation_m_Q(1,data1)
print(estimate_m,estimate_Q)

m_list = []
my_m_list = []
Q_list = []
my_Q_list = []

for i in distance_list:
    data = read_data(i)
    estimate_m, estimate_Q = maximum_likelihood_estimation_m_Q(i,data)
    data_mean = np.mean(data)
    data_var = np.var(data)
    T_los = (i + adapter_length)*2/wired_signal_speed*16000000
    T_waiting = 16000

    m_list.append(estimate_m)
    my_m_list.append(data_mean - T_los - T_waiting)
    Q_list.append(estimate_Q)
    my_Q_list.append(data_var**0.5)

plt.figure()
ax = plt.subplot(211)
ax.plot(distance_list, m_list, c = 'r', label='m estimate value')
ax.plot(distance_list, my_m_list, c = 'b', label='simple m estimate value')

ax = plt.subplot(212)
ax.plot(distance_list, Q_list, c = 'r', label='Q estimate value')
ax.plot(distance_list, my_Q_list, c = 'b', label='simple Q estimate value')

plt.legend()
plt.show()
