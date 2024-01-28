'''
author Manjiang Cao 
e-mail <mcao999@connect.hkust-gz.edu.cn>
this script using to estimate parameters in no multi path situation formula
'''

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.stats


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

m_list = []
Q_list = []
means = []
varance = []
for i in distance_list:
    data = read_data(i)
    data_mean = np.mean(data)
    data_var = np.var(data)
    T_los = (i + adapter_length)*2/wired_signal_speed*16000000
    T_waiting = 16000
    means.append(data_mean)
    varance.append(data_var)
    m_list.append(data_mean - T_los - T_waiting)
    Q_list.append(data_var)

def pdf_distance(params):
    """计算新正态分布与观测数据的 KL 散度之和"""
    mu, sigma = params
    kl_divergences = []

    for i in range(len(m_list)):
        mean_i = m_list[i]
        variance_i = Q_list[i]
        
        distance_between_pdf = (mu-mean_i)**2 + (sigma - np.sqrt(variance_i))**2

        print(distance_between_pdf)
        kl_divergences.append(distance_between_pdf)

    return np.sum(kl_divergences)

# 初始化新正态分布的参数 
mu_init, sigma_init = 4000, 1

# 最小化 KL 散度之和
result = minimize(pdf_distance, [mu_init, sigma_init])

# 获取优化后的参数
mu_opt, sigma_opt = result.x
print(mu_opt, sigma_opt)

traditional_pre = []
new_model_pre = []
for i in means:
    traditional_pre.append(((i-20074.659)/2)/16000000*222222222.222)
    new_model_pre.append(((i - T_waiting - mu_opt)/2)/16000000*wired_signal_speed)

def compute_err(distance_list, pre_list):
    error_list = []
    for i in range(len(distance_list)):
        error = np.abs(distance_list[i] - pre_list[i])
        error_list.append(error)
    
    return np.mean(error_list)

plt.figure()
ax = plt.subplot(111)
ax.plot(distance_list, traditional_pre, c = 'b', label = 'traditional error={}'.format(compute_err(distance_list,traditional_pre)))
ax.plot(distance_list, new_model_pre, c = 'y', label = 'new_model error={}'.format(compute_err(distance_list,new_model_pre)))
ax.plot(distance_list, distance_list, c = 'r', label = 'true distance')
ax.plot(distance_list, [i+1 for i in distance_list], c = 'r', linestyle = '--', label = '+1 error boundray')
ax.plot(distance_list, [i-1 for i in distance_list], c = 'r', linestyle = '--', label = '-1 error boundray')
ax.set_xlabel('true distance')
ax.set_ylabel('predict distance')
ax.set_title('new model in wired experiment with error based time')


plt.legend()
plt.show()
