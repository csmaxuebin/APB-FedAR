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

###################################################原始更新#################################################
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


###################################################Prox#################################################
    # loss_func = nn.CrossEntropyLoss()
    # train_idx = list(train_idx)
    # ldr_train = DataLoader(DatasetSplit(dataset, train_idx), batch_size=args.local_bs, shuffle=True)
    # net.train()
    # mu = 0.01
    # # train and update
    #
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
    #         # 计算近端项损失
    #         loss_proximal = 0
    #         for pm, ps in zip(net.parameters(), netg.parameters()):
    #             loss_proximal += torch.sum(torch.pow(pm - ps, 2))
    #         loss = loss + 0.5 * mu * loss_proximal
    #         loss.backward()
    #         optimizer.step()


#########################################梯度裁剪########################################
    # loss_func = nn.CrossEntropyLoss()
    # train_idx = list(train_idx)
    # # ldr_train：batch数量，train_idx：本地样本数量
    # ldr_train = DataLoader(DatasetSplit(dataset, train_idx), batch_size=args.local_bs, shuffle=True)
    # net.train()
    #
    # # train and update
    # optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    # epoch_loss = []
    # eps = args.eps0 / args.epochs / args.local_ep
    # sensitivity = cal_sensitivity(args.lr, args.norm_clip, args.local_bs)
    # noise_scale = np.sqrt(2 * np.log(1.25 / args.delta)) * sensitivity / eps
    # for iter in range(args.local_ep):
    #     batch_loss = []
    #
    #     for batch_idx, (images, labels) in enumerate(ldr_train):
    #         images, labels = images.to(args.device), labels.to(args.device)
    #         optimizer.zero_grad()
    #         log_probs = net(images)
    #         loss = loss_func(log_probs, labels)
    #         loss.backward()
    #         clip_gradients(args, net, noise_scale)
    #         optimizer.step()
    #         batch_loss.append(loss.item())
    #     epoch_loss.append(sum(batch_loss) / len(batch_loss))
    #
    # return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

#########################################增量裁剪########################################
    print("用户的预算：{:.3f}".format(eps_user))
    loss_func = nn.CrossEntropyLoss()
    train_idx = list(train_idx)
    ldr_train = DataLoader(DatasetSplit(dataset, train_idx), batch_size=args.local_bs, shuffle=True)
    net.train()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.995)
    epoch_loss = []
    eps = eps_user / args.epochs
    sensitivity = args.norm_clip
    noise_scale = np.sqrt(2 * np.log(1.25 / args.delta)) * sensitivity / eps
    start_net = copy.deepcopy(net)
    for iter in range(args.local_ep):
        batch_loss = []
        for batch_idx, (images, labels) in enumerate(ldr_train):
            images, labels = images.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            log_probs = net(images)
            loss = loss_func(log_probs, labels)
            loss.backward()
            optimizer.step()
            # scheduler.step()
            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss) / len(batch_loss))
    delta_net = copy.deepcopy(net)
    w_start = start_net.state_dict()
    w_delta = delta_net.state_dict()
    for i in w_delta.keys():
        w_delta[i] -= w_start[i]
    delta_net.load_state_dict(w_delta)

    # # 打印增量的L2范数
    # l2_norm = 0.0
    # for param in delta_net.parameters():
    #     l2_norm += torch.norm(param)
    # print(f"L2 Norm of the model: {l2_norm}")

    # # 打印增量的L2范数
    # norm = 0.0
    # for name in w_delta.keys():
    #     if (
    #             "running_mean" not in name
    #             and "running_var" not in name
    #             and "num_batches_tracked" not in name
    #     ):
    #         norm += pow(w_delta[name].float().norm(2), 2)
    # total_norm = np.sqrt(norm.cpu().numpy()).reshape(1)
    # print(f"L2 Norm of the model: {total_norm[0]}")

    with torch.no_grad():
        delta_net = clip_parameters(args, delta_net)
    delta_net = add_noise(args, delta_net)
    w_delta = delta_net.state_dict()
    for i in w_start.keys():
        w_start[i] += w_delta[i].to(w_start[i].dtype)

    # self.add_noise(net_noise)
    # w_delta = clip_and_add_noise_our(w_delta,noise_multiplier,args)
    # for i in w_start.keys():
    #     w_start[i] += w_delta[i].to(w_start[i].dtype)

    return w_start, sum(epoch_loss) / len(epoch_loss)


#########################################逐样本裁剪########################################
    # # train and update
    # net.train()
    # loss_func = nn.CrossEntropyLoss(reduction='none')
    # optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    # train_idx = list(train_idx)
    # # ldr_train：batch数量，train_idx：本地样本数量
    # ldr_train = DataLoader(DatasetSplit(dataset, train_idx), batch_size=args.local_bs, shuffle=True)
    #
    # epoch_loss = []
    # sigma = calibrating_sampled_gaussian(args.q, args.eps0, args.delta, iters=args.local_ep * args.epochs, err=1e-3)
    # for iter in range(args.local_ep):
    #     # randomly select q fraction samples from data
    #     # according to the privacy analysis of moments accountant
    #     # training "Lots" are sampled by poisson sampling
    #     batch_loss = []
    #     optimizer.zero_grad()
    #
    #     clipped_grads = {name: torch.zeros_like(param) for name, param in net.named_parameters()}
    #     for batch_x, batch_y in ldr_train:
    #         simple_loss = []
    #         batch_x, batch_y = batch_x.to(args.device), batch_y.to(args.device)
    #         pred_y = net(batch_x.float())
    #         loss = loss_func(pred_y, batch_y.long())
    #
    #         # bound l2 sensitivity (gradient clipping)
    #         # clip each of the gradient in the "Lot"
    #         for i in range(loss.size()[0]):
    #             loss[i].backward(retain_graph=True)
    #             simple_loss.append(loss[i])
    #             torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=args.norm_clip)
    #             for name, param in net.named_parameters():
    #                 clipped_grads[name] += param.grad
    #             net.zero_grad()
    #         batch_loss.append(sum(simple_loss) / len(simple_loss))
    #     # add Gaussian noise
    #     for name, param in net.named_parameters():
    #         clipped_grads[name] += gaussian_noise(clipped_grads[name].shape, args.norm_clip, sigma, device=args.device)
    #
    #     # scale back
    #     for name, param in net.named_parameters():
    #         clipped_grads[name] /= (len(train_idx))
    #
    #     for name, param in net.named_parameters():
    #         param.grad = clipped_grads[name]
    #
    #     # update local model
    #     optimizer.step()
    #     epoch_loss.append(sum(batch_loss) / len(batch_loss))
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

# def clip_paras(args, net, noise_scale):
#     # model_parameter_norm = 0
#     # with torch.no_grad():
#     #     for name, param in net.named_parameters():
#     #         model_parameter_norm += (torch.norm(param) ** 2).item()
#     #
#     #     model_parameter_norm = np.sqrt(model_parameter_norm)
#     #
#     # for name in net.named_parameters():
#     #     net[name] /= max(1, model_parameter_norm / args.norm_clip)  # 增量裁剪
#
#     paras = []
#     for param in net.parameters():
#         paras.append(param.clone())
#     # 将梯度列表展平为一维张量
#     flatten_paras = torch.cat([para.flatten() for para in paras])
#     paras_clipped = flatten_paras / max(1.0, float(torch.norm(flatten_paras, p=2)) / args.norm_clip)
#     # grads_clipped_list = []
#     # for grad in grads:
#     #     grads_clipped = grad.flatten() / max(1.0, float(torch.norm(grad.flatten(), p=2)) / args.norm_clip)
#     #     grads_clipped_list.append(grads_clipped)
#     # grads_concatenated = torch.cat(grads_clipped_list, dim=0)
#     # 将裁剪之后的梯度展平后重新分配给原始梯度
#     start = 0
#     for param in net.parameters():
#         size = param.numel()
#         param.copy_(paras_clipped[start:start + size].view_as(param))
#         start += size
#     for param in net.parameters():
#         noise = torch.normal(0, noise_scale, size=param.shape, device=param.device)
#         param += noise
#
#     return net
#
# def cal_sensitivity(lr, clip, data_size):
#     return 2 * lr * clip / data_size
#
def clip_parameters(args, net):
    for k, v in net.named_parameters():
        v /= max(1, v.norm(2).item() / args.norm_clip)
    return net

def add_noise(args, net):
    sensitivity = cal_sensitivity_up(args.lr, args.norm_clip)
    with torch.no_grad():
        for k, v in net.named_parameters():
            noise = Gaussian_Simple(epsilon=args.eps0, delta=args.delta, sensitivity=sensitivity, size=v.shape)
            noise = torch.from_numpy(noise).to(args.device)
            v += noise
    return net

def cal_sensitivity_up(lr, clip):
    return 2 * lr * clip

def Gaussian_Simple(epsilon, delta, sensitivity, size):
    noise_scale = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
    return np.random.normal(0, noise_scale, size=size)

# def get_sigma_or_epsilon(iter,args,filename):
#     """DP setting"""
#     delta = args.dp_delta  # DP budget
#     c = args.c
#     epsilon = args.dp_epsilon  # DP budget
#     noise_multiplier = get_noise_multiplier_from_epsilon(
#         #隐私预算
#         epsilon = epsilon,
#         #总步数，CL-DP中是通信轮数
#         steps = args.epochs,
#         #每轮客户端采样率
#         sample_rate = args.frac,
#         #给定的delta
#         delta = args.dp_delta,
#         #采用什么隐私机制
#         mechanism = args.accountant,
#     )
#     #计算加噪量，sigma
#     sigma_averaged = (noise_multiplier * c / math.sqrt(args.num_users * args.frac))
#     return epsilon,noise_multiplier
#
def clip_and_add_noise_our(w,noise_multiplier,args):
    l2_norm = cal_clip(w)
    with torch.no_grad():
        for name in w.keys():
            noise = torch.FloatTensor(w[name].shape).normal_(0, noise_multiplier * args.norm_clip /np.sqrt(args.num_users))
            noise = noise.cpu().numpy()
            noise = torch.from_numpy(noise).type(torch.FloatTensor).to(w[name].device)
            w[name] = w[name].float() * min(1, args.norm_clip / l2_norm)
            w[name] = w[name].add_(noise)
    return w

def cal_clip(w):
    norm = 0.0
    for name in w.keys():
        norm += pow(w[name].float().norm(2), 2)
    total_norm = np.sqrt(norm.cpu().numpy()).reshape(1)
    return total_norm[0]