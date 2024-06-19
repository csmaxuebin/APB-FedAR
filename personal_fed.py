import math
from decimal import Decimal

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
from torchsummary import summary
import time
import random
import logging
import json
from hashlib import md5
import copy
import easydict
import os
import sys
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
import pickle
import dill

# Directory where the json file of arguments will be present
# 参数的json文件所在的目录
directory = './Parse_Files'


# Import different files required for loading dataset, model, testing, training
from utility.LoadSplit import Load_Dataset, Load_Model
# from utility.options import args_parser
from models.Update import train_client, test_client, finetune_client
from models.Fed import FedAvg, DiffPrivFedAvg, FedAvg_delta, DR_FedAvg, W_FedAvg, project, AFL
from models.test import test_img

torch.manual_seed(0)

if __name__ == '__main__':
    
    # Initialize argument dictionary 初始化参数字典
    args = {}

    # From Parse_Files folder, get the name of required parse file which is provided while running this script from bash
    # 从Parse_file文件夹中，获取从bash运行此脚本时提供的所需解析文件的名称

    f = directory+'/'+str(sys.argv[1])  # str()：将参数转换成字符串类型
    print(f)
    with open(f) as json_file:  
        args = json.load(json_file)

    # Taking hash of config values and using it as filename for storing model parameters and logs
    # 获取配置值的哈希并将其用作存储模型参数和日志的文件名
    param_str = json.dumps(args)  # json.dumps()：将字典转化为字符串
    file_name = md5(param_str.encode()).hexdigest()

    # Converting args to easydict to access elements as args.device rather than args[device]
    # 将args转换为easydict，获取args.device形式的元素而不是args[device]形式的元素
    args = easydict.EasyDict(args)
    print(args)

    # Save configurations by making a file using hash value
    # 通过使用哈希值创建文件来保存配置
    with open('./config/parser_{}.txt'.format(file_name),'w') as outfile:
        json.dump(args,outfile,indent=4)

    SUMMARY = os.path.join('./results',file_name)
    args.summary=SUMMARY
    os.makedirs(SUMMARY)
        
    # Setting the device - GPU or CPU
    # 设置设备
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # net_glob = Load_Model(args=args) 
    # print(net_glob) 
    # exit()
    # Load the training and testing datasets
    dataset_train, dataset_test, dict_users = Load_Dataset(args=args)

    # Initialize Global Server Model
    # 初始化全局模型
    net_glob = Load_Model(args=args)
    net_glob.to(args.device)
    # for k,v in net_glob.state_dict().items():
    #     print(k)
    # print(net_glob)
    net_glob.train()

    # Print name of the architecture - 'MobileNet or ResNet or NewNet'
    # 打印模型结构，MobileNet或ResNet或NewNet
    print(args.model)

    # copy weights
    # 将net_glob中的所有可学习参数和对应的权重保存在一个字典w_glob中
    w_glob = net_glob.state_dict()
    w_old_glob = net_glob.state_dict()

    # Set up log file
    # 设置日志文件
    logging.basicConfig(filename='./log/{}.log'.format(file_name),format='%(message)s',level=logging.DEBUG)

    tree=lambda : defaultdict(tree)
    stats=tree()
    writer = SummaryWriter(args.summary)

    # splitting user data into training and testing parts
    # 将用户数据拆分为训练和测试部分
    train_data_users = {}
    test_data_users = {}
    N_train = 0  # 保存所有客户端训练样本总数
    for i in range(args.num_users):
        dict_users[i] = list(dict_users[i])
        train_data_users[i] = list(random.sample(dict_users[i],int(args.split_ratio*len(dict_users[i]))))
        test_data_users[i] = list(set(dict_users[i])-set(train_data_users[i]))
        N_train += len(train_data_users[i])

    pk = [len(train_data_users[i]) / N_train for i in range(args.num_users)]  # pk是一个列表，保存每个客户端样本数量占比

    # exit()
    # local models for each client
    # 每个客户端的本地模型
    local_nets = {}  # 字典，保存本地模型，键是idx，值是第idx个客户端的本地模型

    for i in range(0, args.num_users):
        local_nets[i] = Load_Model(args=args)
        local_nets[i].train()
        # 将全局模型参数w_glob加载到第i个本地模型local_nets[i]中，load_state_dict()是将状态字典加载到模型中
        local_nets[i].load_state_dict(w_glob)

    # Start training

    logging.info("Training")

    start = time.time()
    for iter in range(args.epochs):

        print('Round {}'.format(iter))
        logging.info("---------Round {}---------".format(iter))

        if iter < (args.epochs) * 0.4:
            if iter % 2 == 0:

                w_locals, loss_locals = [], []
                w_select_locals, loss_select_locals = [], []
                
                # hyperparameter = number of layers we want to keep in the base part
                base_layers = args.base_layers

                for idx in range(args.num_users):
                    w_locals.append(local_nets[idx].state_dict())

                m = max(int(args.frac * args.num_users), 1)
                idxs_users = np.random.choice(range(args.num_users), m, replace=False)

           
                for i in range(args.num_users):
                    acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], net_glob)
                    loss_locals.append(copy.deepcopy(loss_test))
                    loss_history[i][iter] = loss_test

                avg_loss = 0
                for i in range(args.num_users):
                    avg_loss += loss_history[i][iter]
                avg_loss /= args.num_users
                loss_history_global_avg.append(copy.deepcopy(avg_loss))
    
                for idx in idxs_users:
                    w, loss = train_client(args, dataset_train, train_data_users[idx], net=local_nets[idx],
                                           eps_user=args.eps0)
                    w_select_locals.append(w)
                    loss_select_locals.append(copy.deepcopy(loss))

                # store testing and training accuracies of the model before global aggregation 测试本地模型聚合前的训练集和测试集准确度
                logging.info("Testing Client Models before aggregation")
                logging.info("")
                s = 0
                for i in idxs_users:
                    logging.info("Client {}:".format(i))
                    acc_train, loss_train = test_client(args, dataset_train, train_data_users[i], local_nets[i])
                    acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], local_nets[i])
                    logging.info("Training accuracy: {:.3f}".format(acc_train))
                    logging.info("Testing accuracy: {:.3f}".format(acc_test))
                    logging.info("")
                    # print(acc_test)
                    stats[i][iter]['Before Training accuracy'] = acc_train
                    stats[i][iter]['Before Test accuracy'] = acc_test
                    writer.add_scalar(str(i) + '/Before Training accuracy', acc_train, iter)
                    writer.add_scalar(str(i) + '/Before Test accuracy', acc_test, iter)

                    s += acc_test
                s /= m
                logging.info("Average Client accuracy on their test data: {: .3f}".format(s))
                stats['Before Average'][iter] = s
                writer.add_scalar('Average' + '/Before Test accuracy', s, iter)

                # update global weights
                w_glob, e = DR_FedAvg(args, w_select_locals, select_pk, select_Fk)

                # for idx in idxs_users:
                #     user_weight_history[idx][iter] = e[idx]

                # copy weight to net_glob
                net_glob.load_state_dict(w_glob)  # 将全局模型参数w_glob加载到全局模型中，load_state_dict()是将状态字典加载到模型中

                # Updating base layers of the clients and keeping the personalized layers same  # 更新本地模型的所有层
                for idx in range(args.num_users):
                    for i in list(w_glob.keys()):
                        w_locals[idx][i] = copy.deepcopy(w_glob[i])  # w_locals[idx][i]：表示第idx个本地模型的第i层

                    local_nets[idx].load_state_dict(w_locals[idx])

                # store train and test accuracies after updating local models
                logging.info("Testing global Models after aggregation")
                logging.info("")
                s = 0
                var = 0
                for i in range(args.num_users):
                    logging.info("Client {}:".format(i))
                    acc_train, loss_train = test_client(args, dataset_train, train_data_users[i], net_glob)
                    acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], net_glob)
                    logging.info("Training accuracy: {:.3f}".format(acc_train))
                    logging.info("Testing accuracy: {:.3f}".format(acc_test))
                    logging.info("")

                    stats[i][iter]['After Training accuracy'] = acc_train
                    stats[i][iter]['After Test accuracy'] = acc_test
                    writer.add_scalar(str(i) + '/After Training accuracy', acc_train, iter)
                    writer.add_scalar(str(i) + '/After Test accuracy', acc_test, iter)

                    s += acc_test
                s /= args.num_users
                for i in range(args.num_users):
                    acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], net_glob)
                    var += (acc_test - s) ** 2
                var /= args.num_users
                logging.info("Average global model accuracy on their test data: {: .3f}".format(s))
                logging.info("Average global model Variance on their test data: {: .3f}".format(var))

                stats['After Average'][iter] = s
                writer.add_scalar('Average' + '/After Test accuracy', s, iter)

                loss_avg = sum(loss_select_locals) / len(loss_select_locals)
                logging.info('Average loss of clients: {:.3f}'.format(loss_avg))

           
