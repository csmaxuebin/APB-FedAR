import copy
import random

import numpy as np
import torch
from torch import nn

# since the number of samples in all the users is same, simple averaging works 所有用户中的样本数量相同，进行简单平均
def FedAvg(w):

    '''

    Function to average the updated weights of client models to update the global model (when the number of samples is same for each client)
    对客户端本地模型的更新权重进行平均以更新全局模型（当每个客户端的样本数量相同时）

    Parameters:

        w (list) : The list of state_dicts of each client 每个客户端的状态字典列表

    Returns:

        w_avg (state_dict) : The updated state_dict for global model 全局模型的更新状态字典

    '''

    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))  # torch.div()：张量和标量做逐元素除法
    return w_avg

def DR_FedAvg(args, w, pk, Fk):
    e = [0 for i in range(int(args.frac * args.num_users))]
    for i in range(int(args.frac * args.num_users)):
        sumpkFk = sum([pk[i] * (Fk[i] ** (args.q + 1)) for i in range(int(args.frac * args.num_users))])  # 求DRFL的分母
    w_avg = copy.deepcopy(w[0])
    e[0] = pk[0] * (Fk[0] ** (args.q + 1)) / sumpkFk
    for k in w_avg.keys():
        w_avg[k] = w_avg[k] * e[0]
    for k in w_avg.keys():
        for i in range(1, len(w)):
            e[i] = pk[i] * (Fk[i] ** (args.q + 1)) / sumpkFk
            w_avg[k] += w[i][k] * e[i]
    return w_avg, e

def AFL(args, w, e):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = e[0] * w_avg[k]
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * e[i]
    return w_avg

def project(y):
    ''' algorithm comes from:
    https://arxiv.org/pdf/1309.1541.pdf
    '''
    u = sorted(y, reverse=True)
    x = []
    rho = 0
    for i in range(len(y)):
        if (u[i] + (1.0/(i+1)) * (1-np.sum(np.asarray(u)[:i]))) > 0:
            rho = i + 1
    lambda_ = (1.0/rho) * (1-np.sum(np.asarray(u)[:rho]))
    for i in range(len(y)):
        x.append(max(y[i]+lambda_, 0))
    return x

def W_FedAvg(w, e):

    '''

    Function to average the updated weights of client models to update the global model (when the number of samples is same for each client)
    对客户端本地模型的更新权重进行平均以更新全局模型（当每个客户端的样本数量相同时）

    Parameters:

        w (list) : The list of state_dicts of each client 每个客户端的状态字典列表

    Returns:

        w_avg (state_dict) : The updated state_dict for global model 全局模型的更新状态字典

    '''
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = e[0] * w_avg[k]
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * e[i]
    return w_avg

def FedAvg_delta(w, w_glob):

    '''

    Function to average the updated weights of client models to update the global model (when the number of samples is same for each client)
    对客户端本地模型的更新权重进行平均以更新全局模型（当每个客户端的样本数量相同时）

    Parameters:

        w (list) : The list of state_dicts of each client 每个客户端的状态字典列表

    Returns:

        w_avg (state_dict) : The updated state_dict for global model 全局模型的更新状态字典

    '''

    w_avg_delta = copy.deepcopy(w[0])
    for k in w_avg_delta.keys():  # 计算本地更新量的平均值并更新全局模型
        for i in range(1, len(w)):
            w_avg_delta[k] += w[i][k]
        w_avg_delta[k] = torch.div(w_avg_delta[k], len(w))  # torch.div()：张量和标量做逐元素除法
        w_glob[k] = w_glob[k] - w_avg_delta[k]
    return w_glob  # 返回更新后的全局模型

def customFedAvg(w,weight=1):

    '''

    Function to average the updated weights of client models to update the global model (when the number of samples is same for each client)
    对客户端本地模型的更新权重进行平均以更新全局模型（当每个客户端的样本数量相同时）

    Parameters:

        w (list) : The list of state_dicts of each client 每个客户端的状态字典列表

    Returns:

        w_avg (state_dict) : The updated state_dict for global model 全局模型的更新状态字典

    '''

    print(len(w))
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

# if number of samples are different, use this function in personal_fed.py
def FedAvgRefined(w,count):

    '''

    Function to average the updated weights of client models to update the global model (clients can have different number of samples)

    Parameters:

        w (list) : The list of state_dicts of each client

        count (list) : The list of number of samples each client has

    Returns:

        w_updated (state_dict) : The updated state_dict for global model after doing the weighted average of local models

    '''

    
    w_mul = []

    for j in range(len(w)):
        w_avg = copy.deepcopy(w[j])

        for i in w_avg.keys():
            w_avg[i] = torch.mul(w_avg[i],count[0])

        w_mul.append(w_avg)

    w_updated = copy.deepcopy(w_mul[0])

    for k in w_updated.keys():
        for i in range(1, len(w_mul)):
            w_updated[k] += w_mul[i][k]
        w_updated[k] = torch.div(w_updated[k], sum(count))
    return w_updated



# if you want to add random noise for some users with probability p, use this update function  带有DP的FedAvg
def DiffPrivFedAvg(w):

    '''
    通过加入差分隐私（向客户端的模型添加噪声，使其数据无法根据模型重建）来聚合本地模型
    Update global model by incorporating Differential Privacy (Adding noise to the weights of the clients so that their data cannot be reconstructed from model weights)
    此函数是针对每个客户端拥有相同数量的数据样本
    Current implementation is for same number of data samples per client

    Parameters:

        w (list) : The list of state_dicts of local models  本地模型的字典列表

    Returns:

        w_avg (state_dict) : Updated state_dict for global model  聚合后的全局模型字典

    Working:
        p（客户端选择原始模型的概率）：（0,1）间的一个概率值
        p (probability of selecting original weights for a particular client) : Set this value from (0,1) 

        Generate noise:

            Mean : 0  均值为0
            标准差：所有模型参数的平方和除以模型张量的元素总数
            Standard Deviation : Sum of squares of all the weights divided by total number of elements of the weight tensor
            形状：与模型参数张量相同
            Shape : Same as that of weight tensor
        将此生成的噪声添加到模型参数张量的副本中，并使用该值进行聚合
        Add this generated noise to a copy of weight tensor and use that value for aggregation

    '''


    
    w_new = []  # 保存加噪后的本地模型参数
    
    for i in range(len(w)):
        
        a = random.uniform(0,1)
        
        #probability of selecting the original weights  选择原始模型参数的概率
        p = 0.8

        if(a<=p):
            w_new.append(copy.deepcopy(w[i]))
        else:
            w_temp = copy.deepcopy(w[i])  # 本地模型副本
            
            for keys in w_temp.keys():  # 循环遍历模型的每一层
                
                # copy original model weights  保存原始第key层参数

                beta = copy.deepcopy(w_temp[keys])
                
                # convert it to numpy to find sum of squares of its elements  将第key层参数转换为numpy，以计算其元素的平方和

                alpha = w_temp[keys].cpu().numpy()
                
                epsilon = 10**(-8)  # 10的-8次方

                # set very small elements to zero
                # 将alpha中绝对值小于epsilon的元素置为0
                alpha[np.abs(alpha) < epsilon] = 0
                
                alpha = alpha + 0.000005
                # 将alpha中的每个元素平方
                ele_square = np.power(alpha,2)
                # 平方后求和
                ele_sum = np.sum(ele_square)
                
                # Divide sum of squares value by size of tensor to get standard deviation
                # 计算标准差
                ele_val = ele_sum.item()/alpha.size

                # Generate gaussian noise of same shape as that of model weights
                # 生成与模型参数形状相同的高斯噪声
                # numpy.random.normal(loc=0,scale=1.0,size=shape)：生成高斯分布的随机数
                # loc(float)：正态分布的均值；scale(float)：正态分布的标准差；size(int或者整数元组)：输出的值赋在shape里，默认为None
                w_temp[keys] = np.random.normal(0,ele_val,np.shape(w_temp[keys]))
                # 把数组转换成张量，且二者共享内存，对张量进行修改，原始数组也会相应发生改变
                w_temp[keys] = torch.from_numpy(w_temp[keys])
                
                w_temp[keys] = w_temp[keys].type(torch.cuda.FloatTensor)
                
                # Add noise to the weights  添加噪声

                w_temp[keys] = beta + w_temp[keys]
            
            w_new.append(copy.deepcopy(w_temp))
            
    
    w_avg = copy.deepcopy(w_new[0])
    
    for k in w_avg.keys():
        for i in range(1, len(w_new)):
            
            w_avg[k] = w_avg[k].type(torch.cuda.FloatTensor)
            w_new[i][k] = w_new[i][k].type(torch.cuda.FloatTensor)
        
            w_avg[k] += w_new[i][k]
        
        w_avg[k] = torch.div(w_avg[k], len(w_new))
        w_avg[k] = w_avg[k].type(torch.cuda.FloatTensor)
        
    return w_avg   # 返回加噪后的聚合模型

