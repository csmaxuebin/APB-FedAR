import copy
import math
import sys

import torch
from dill import extend
from torch import nn, autograd, special
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
import random
import matplotlib.pyplot as plt
import torch.nn.functional as F
from typing import List
from decimal import *
from scipy.special import comb
from scipy import special
from torch.utils.data import TensorDataset

class DatasetSplit(Dataset):

    """
    Class DatasetSplit - To get datasamples corresponding to the indices of samples a particular client has from the actual complete dataset
    DatasetSplit类：从实际完整的数据集中获取与指定客户端拥有的样本索引相对应的数据样本

    """

    def __init__(self, dataset, idxs):

        """

        Constructor Function 构造函数

        Parameters:

            dataset: The complete dataset 完整的数据集

            idxs : List of indices of complete dataset that is there in a particular client 指定客户端中完整数据集的索引列表

        """
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):

        """

        returns length of local dataset 返回本地数据集的数量

        """

        return len(self.idxs)

    def __getitem__(self, item):

        """
        Gets individual samples from complete dataset 从完整的数据集中获取单个样本

        returns image and its label 返回图像及其标签

        """
        image, label = self.dataset[self.idxs[item]]
        return image, label
    

# function to train a client 客户端本地训练

def train_client(args,dataset,train_idx,net,eps_user):

    '''

    Train individual client models 训练本地模型

    Parameters:

        net (state_dict) : Client Model 客户端本地模型

        datatest (dataset) : Complete dataset loaded by the Dataloader 数据加载器加载的完整数据集

        args (dictionary) : The list of arguments defined by the user 用户定义的参数列表

        train_idx (list) : List of indices of those samples from the actual complete dataset that are there in the local training dataset of this client
        此客户端的本地训练数据集中实际完整数据集中的样本索引列表

    Returns:

        net.state_dict() (state_dict) : The updated weights of the client model 客户端本地模型的更新权重

        train_loss (float) : Cumulative loss while training 训练时的累积损失

    '''

    # loss_func = nn.CrossEntropyLoss()
    # train_idx = list(train_idx)
    # ldr_train = DataLoader(DatasetSplit(dataset, train_idx), batch_size=args.local_bs, shuffle=True)
    # net.train()
    #
    # # train and update
    # optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    # epoch_loss = []
    #
    # for iter in range(args.local_ep):
    #     batch_loss = []
    #     for batch_idx, (images, labels) in enumerate(ldr_train):
    #         images, labels = images.to(args.device), labels.to(args.device)
    #         optimizer.zero_grad()
    #         log_probs = net(images)
    #         loss = loss_func(log_probs, labels)
    #         loss.backward()
    #         optimizer.step()
    #         batch_loss.append(loss.item())
    #     epoch_loss.append(sum(batch_loss) / len(batch_loss))
    #
    # return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

def finetune_client(args,dataset,train_idx,net):

    '''

    Train individual client models

    Parameters:

        net (state_dict) : Client Model

        datatest (dataset) : Complete dataset loaded by the Dataloader

        args (dictionary) : The list of arguments defined by the user

        train_idx (list) : List of indices of those samples from the actual complete dataset that are there in the local training dataset of this client

    Returns:

        net.state_dict() (state_dict) : The updated weights of the client model

        train_loss (float) : Cumulative loss while training

    '''

    loss_func = nn.CrossEntropyLoss()
    train_idx = list(train_idx)
    ldr_train = DataLoader(DatasetSplit(dataset, train_idx), batch_size=args.local_bs, shuffle=True)
    net.train()
    
    # train and update
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    epoch_loss = []
    
    for iter in range(1):   
        batch_loss = []
        
        for batch_idx, (images, labels) in enumerate(ldr_train):
            
            images, labels = images.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            log_probs = net(images)
            loss = loss_func(log_probs, labels)
            loss.backward()
            optimizer.step()
            
            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
    return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


# function to test a client 测试客户端本地模型
def test_client(args,dataset,test_idx,net):

    '''

    Test the performance of the client models on their datasets 在客户端的本地数据集上测试本地模型的性能

    Parameters:

        net (state_dict) : Client Model 客户端本地模型

        datatest (dataset) : The data on which we want the performance of the model to be evaluated 希望的评估模型性能的数据

        args (dictionary) : The list of arguments defined by the user 用户定义的参数列表

        test_idx (list) : List of indices of those samples from the actual complete dataset that are there in the local dataset of this client
        此客户端本地数据集中实际完整数据集中这些样本的索引列表

    Returns:

        accuracy (float) : Percentage accuracy on test set of the model 模型测试集的准确率百分比

        test_loss (float) : Cumulative loss on the data 样本的累积损失

    '''
    
    data_loader = DataLoader(DatasetSplit(dataset, test_idx), batch_size=args.local_bs)  
    net.eval()
    #print (test_data)
    test_loss = 0
    correct = 0
    
    l = len(data_loader)
    
    with torch.no_grad():
                
        for idx, (data, target) in enumerate(data_loader):
            if args.gpu != -1:
                data, target = data.cuda(), target.cuda()
            log_probs = net(data)
            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            
            correct += y_pred.eq(target.data.view_as(y_pred)).float().cpu().sum()

        test_loss /= len(data_loader.dataset)
        accuracy = 100.00 * correct / len(data_loader.dataset)

        return accuracy, test_loss

def cal_clip(w):
    norm = 0.0
    for name in w.keys():
        norm += pow(w[name].float().norm(2), 2)
    total_norm = np.sqrt(norm.cpu().numpy()).reshape(1)
    return total_norm[0]
