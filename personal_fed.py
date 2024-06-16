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
    vari = []  # 保存历史方差
    base_layers = args.base_layers
    loss_history_global_avg = []  # 保存全局模型历史平均损失
    delta_F = [0] * args.epochs  # 保存全局模型历史平均损失的减小量
    eps_c = 0  # 保存已经使用的隐私预算
    eps = 100  # 总隐私预算
    eps_s = 0.9 * eps / 100  # 初始隐私预算
    loss_history = [[0] * args.epochs for i in range(args.num_users)]  # 保存所有客户端历史损失，10*100的二维列表，初始化为全0
    user_loss_history = [[0] * args.epochs for i in range(args.num_users)]  # 保存所有客户端历史衰减损失
    user_weight_history = [[0] * args.epochs for i in range(args.num_users)]  # 保存所有客户端历史权重
    latest_lambdas = np.ones(args.num_users) * 1.0 / args.num_users
    eps_history = [0] * 10  # 保存所有客户端历史消耗的总预算
    user_eps_history = [[0] * args.epochs for i in range(args.num_users)]  # 保存所有客户端每轮历史预算，10*100的二维列表，初始化为全0

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

######################################################FedVF#######################################################
    #     print('Round {}'.format(iter))
    #     logging.info("---------Round {}---------".format(iter))
    #
    #     w_locals, loss_locals = [], []  # w_locals保存客户端本地模型参数，loss_locals保存本地模型的loss
    #     w_select_locals, loss_select_locals = [], []
    #     select_Fk = []  # select_Fk是一个列表，保存每个客户端的本地损失
    #     select_pk = []
    #     # hyperparameter = number of layers we want to keep in the base part
    #     base_layers = args.base_layers
    #
    #     for idx in range(args.num_users):
    #         w_locals.append(local_nets[idx].state_dict())
    #
    #     m = max(int(args.frac * args.num_users), 1)
    #     idxs_users = np.random.choice(range(args.num_users), m, replace=False)
    #
    #     if iter < (args.epochs) * 0.4:
    #         if iter % 2 == 0:
    #             for idx in idxs_users:
    #                 w, loss = train_client(args, dataset_train, train_data_users[idx], net=local_nets[idx])  # w是保存本地模型的字典
    #                 loss_select_locals.append(copy.deepcopy(loss))
    #                 w_select_locals.append(w)
    #
    #             # store testing and training accuracies of the model before global aggregation 测试本地模型聚合前的训练集和测试集准确度
    #             logging.info("Testing Client Models before aggregation")
    #             logging.info("")
    #             s = 0
    #             for i in idxs_users:
    #                 logging.info("Client {}:".format(i))
    #                 acc_train, loss_train = test_client(args, dataset_train, train_data_users[i], local_nets[i])
    #                 acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], local_nets[i])
    #                 logging.info("Training accuracy: {:.3f}".format(acc_train))
    #                 logging.info("Testing accuracy: {:.3f}".format(acc_test))
    #                 logging.info("")
    #                 # print(acc_test)
    #                 stats[i][iter]['Before Training accuracy'] = acc_train
    #                 stats[i][iter]['Before Test accuracy'] = acc_test
    #                 writer.add_scalar(str(i) + '/Before Training accuracy', acc_train, iter)
    #                 writer.add_scalar(str(i) + '/Before Test accuracy', acc_test, iter)
    #
    #                 s += acc_test
    #             s /= m
    #             logging.info("Average Client accuracy on their test data: {: .3f}".format(s))
    #             stats['Before Average'][iter] = s
    #             writer.add_scalar('Average' + '/Before Test accuracy', s, iter)
    #
    #             # update global weights
    #             w_glob = FedAvg(w_select_locals)
    #
    #             # copy weight to net_glob
    #             net_glob.load_state_dict(w_glob)  # 将全局模型参数w_glob加载到全局模型中，load_state_dict()是将状态字典加载到模型中
    #
    #             # Updating base layers of the clients and keeping the personalized layers same  # 更新本地模型的所有层
    #             for idx in range(args.num_users):
    #                 for i in list(w_glob.keys()):
    #                     w_locals[idx][i] = copy.deepcopy(w_glob[i])  # w_locals[idx][i]：表示第idx个本地模型的第i层
    #
    #                 local_nets[idx].load_state_dict(w_locals[idx])
    #
    #             # store train and test accuracies after updating local models
    #             logging.info("Testing Client Models after aggregation")
    #             logging.info("")
    #             s = 0
    #             var = 0
    #             for i in range(args.num_users):
    #                 logging.info("Client {}:".format(i))
    #                 acc_train, loss_train = test_client(args, dataset_train, train_data_users[i], net_glob)
    #                 acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], net_glob)
    #                 logging.info("Training accuracy: {:.3f}".format(acc_train))
    #                 logging.info("Testing accuracy: {:.3f}".format(acc_test))
    #                 logging.info("")
    #
    #                 stats[i][iter]['After Training accuracy'] = acc_train
    #                 stats[i][iter]['After Test accuracy'] = acc_test
    #                 writer.add_scalar(str(i) + '/After Training accuracy', acc_train, iter)
    #                 writer.add_scalar(str(i) + '/After Test accuracy', acc_test, iter)
    #
    #                 s += acc_test
    #             s /= args.num_users
    #             for i in range(args.num_users):
    #                 acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], net_glob)
    #                 var += (acc_test - s) ** 2
    #             var /= args.num_users
    #             logging.info("Average Client accuracy on their test data: {: .3f}".format(s))
    #             logging.info("Average Client accuracy Variance on their test data: {: .3f}".format(var))
    #
    #             stats['After Average'][iter] = s
    #             writer.add_scalar('Average' + '/After Test accuracy', s, iter)
    #
    #             loss_avg = sum(loss_select_locals) / len(loss_select_locals)
    #             logging.info('Average loss of clients: {:.3f}'.format(loss_avg))
    #
    #         else:
    #             for idx in idxs_users:
    #                 w, loss = train_client(args, dataset_train, train_data_users[idx], net=local_nets[idx])  # w是保存本地模型的字典
    #                 loss_select_locals.append(copy.deepcopy(loss))
    #                 w_select_locals.append(w)
    #
    #             # store testing and training accuracies of the model before global aggregation 测试本地模型聚合前的训练集和测试集准确度
    #             logging.info("Testing Client Models before aggregation")
    #             logging.info("")
    #             s = 0
    #             for i in idxs_users:
    #                 logging.info("Client {}:".format(i))
    #                 acc_train, loss_train = test_client(args, dataset_train, train_data_users[i], local_nets[i])
    #                 acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], local_nets[i])
    #                 logging.info("Training accuracy: {:.3f}".format(acc_train))
    #                 logging.info("Testing accuracy: {:.3f}".format(acc_test))
    #                 logging.info("")
    #                 # print(acc_test)
    #                 stats[i][iter]['Before Training accuracy'] = acc_train
    #                 stats[i][iter]['Before Test accuracy'] = acc_test
    #                 writer.add_scalar(str(i) + '/Before Training accuracy', acc_train, iter)
    #                 writer.add_scalar(str(i) + '/Before Test accuracy', acc_test, iter)
    #
    #                 s += acc_test
    #             s /= m
    #             logging.info("Average Client accuracy on their test data: {: .3f}".format(s))
    #             stats['Before Average'][iter] = s
    #             writer.add_scalar('Average' + '/Before Test accuracy', s, iter)
    #
    #             # update global weights
    #             w_new_glob = FedAvg(w_select_locals)
    #
    #             for i in list(w_glob.keys())[0:base_layers]:
    #                 w_glob[i] = copy.deepcopy(w_new_glob[i])
    #
    #             # copy weight to net_glob
    #             net_glob.load_state_dict(w_glob)  # 将全局模型参数w_glob加载到全局模型中，load_state_dict()是将状态字典加载到模型中
    #
    #             # Updating base layers of the clients and keeping the personalized layers same  # 更新本地模型的基础层
    #             for idx in range(args.num_users):
    #                 for i in list(w_glob.keys())[0:base_layers]:
    #                     w_locals[idx][i] = copy.deepcopy(w_glob[i])  # w_locals[idx][i]：表示第idx个本地模型的第i层
    #
    #                 local_nets[idx].load_state_dict(w_locals[idx])
    #
    #             # store train and test accuracies after updating local models
    #             logging.info("Testing Client Models after aggregation")
    #             logging.info("")
    #             s = 0
    #             var = 0
    #             for i in range(args.num_users):
    #                 logging.info("Client {}:".format(i))
    #                 acc_train, loss_train = test_client(args, dataset_train, train_data_users[i], net_glob)
    #                 acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], net_glob)
    #                 logging.info("Training accuracy: {:.3f}".format(acc_train))
    #                 logging.info("Testing accuracy: {:.3f}".format(acc_test))
    #                 logging.info("")
    #
    #                 stats[i][iter]['After Training accuracy'] = acc_train
    #                 stats[i][iter]['After Test accuracy'] = acc_test
    #                 writer.add_scalar(str(i) + '/After Training accuracy', acc_train, iter)
    #                 writer.add_scalar(str(i) + '/After Test accuracy', acc_test, iter)
    #
    #                 s += acc_test
    #             s /= args.num_users
    #             for i in range(args.num_users):
    #                 acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], net_glob)
    #                 var += (acc_test - s) ** 2
    #             var /= args.num_users
    #             logging.info("Average Client accuracy on their test data: {: .3f}".format(s))
    #             logging.info("Average Client accuracy Variance on their test data: {: .3f}".format(var))
    #
    #             stats['After Average'][iter] = s
    #             writer.add_scalar('Average' + '/After Test accuracy', s, iter)
    #
    #             loss_avg = sum(loss_select_locals) / len(loss_select_locals)
    #             logging.info('Average loss of clients: {:.3f}'.format(loss_avg))
    #
    #     else:
    #         if iter % 4 == 0:
    #             for idx in idxs_users:
    #                 w, loss = train_client(args, dataset_train, train_data_users[idx], net=local_nets[idx])  # w是保存本地模型的字典
    #                 loss_select_locals.append(copy.deepcopy(loss))
    #                 w_select_locals.append(w)
    #
    #             # store testing and training accuracies of the model before global aggregation 测试本地模型聚合前的训练集和测试集准确度
    #             logging.info("Testing Client Models before aggregation")
    #             logging.info("")
    #             s = 0
    #             for i in idxs_users:
    #                 logging.info("Client {}:".format(i))
    #                 acc_train, loss_train = test_client(args, dataset_train, train_data_users[i], local_nets[i])
    #                 acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], local_nets[i])
    #                 logging.info("Training accuracy: {:.3f}".format(acc_train))
    #                 logging.info("Testing accuracy: {:.3f}".format(acc_test))
    #                 logging.info("")
    #                 # print(acc_test)
    #                 stats[i][iter]['Before Training accuracy'] = acc_train
    #                 stats[i][iter]['Before Test accuracy'] = acc_test
    #                 writer.add_scalar(str(i) + '/Before Training accuracy', acc_train, iter)
    #                 writer.add_scalar(str(i) + '/Before Test accuracy', acc_test, iter)
    #
    #                 s += acc_test
    #             s /= m
    #             logging.info("Average Client accuracy on their test data: {: .3f}".format(s))
    #             stats['Before Average'][iter] = s
    #             writer.add_scalar('Average' + '/Before Test accuracy', s, iter)
    #
    #             # update global weights
    #             w_glob = FedAvg(w_select_locals)
    #
    #             # copy weight to net_glob
    #             net_glob.load_state_dict(w_glob)  # 将全局模型参数w_glob加载到全局模型中，load_state_dict()是将状态字典加载到模型中
    #
    #             # Updating base layers of the clients and keeping the personalized layers same  # 更新本地模型的所有层
    #             for idx in range(args.num_users):
    #                 for i in list(w_glob.keys()):
    #                     w_locals[idx][i] = copy.deepcopy(w_glob[i])  # w_locals[idx][i]：表示第idx个本地模型的第i层
    #
    #                 local_nets[idx].load_state_dict(w_locals[idx])
    #
    #             # store train and test accuracies after updating local models
    #             logging.info("Testing Client Models after aggregation")
    #             logging.info("")
    #             s = 0
    #             var = 0
    #             for i in range(args.num_users):
    #                 logging.info("Client {}:".format(i))
    #                 acc_train, loss_train = test_client(args, dataset_train, train_data_users[i], net_glob)
    #                 acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], net_glob)
    #                 logging.info("Training accuracy: {:.3f}".format(acc_train))
    #                 logging.info("Testing accuracy: {:.3f}".format(acc_test))
    #                 logging.info("")
    #
    #                 stats[i][iter]['After Training accuracy'] = acc_train
    #                 stats[i][iter]['After Test accuracy'] = acc_test
    #                 writer.add_scalar(str(i) + '/After Training accuracy', acc_train, iter)
    #                 writer.add_scalar(str(i) + '/After Test accuracy', acc_test, iter)
    #
    #                 s += acc_test
    #             s /= args.num_users
    #             for i in range(args.num_users):
    #                 acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], net_glob)
    #                 var += (acc_test - s) ** 2
    #             var /= args.num_users
    #             logging.info("Average Client accuracy on their test data: {: .3f}".format(s))
    #             logging.info("Average Client accuracy Variance on their test data: {: .3f}".format(var))
    #
    #             stats['After Average'][iter] = s
    #             writer.add_scalar('Average' + '/After Test accuracy', s, iter)
    #
    #             loss_avg = sum(loss_select_locals) / len(loss_select_locals)
    #             logging.info('Average loss of clients: {:.3f}'.format(loss_avg))
    #
    #         else:
    #             for idx in idxs_users:
    #                 w, loss = train_client(args, dataset_train, train_data_users[idx], net=local_nets[idx])  # w是保存本地模型的字典
    #                 loss_select_locals.append(copy.deepcopy(loss))
    #                 w_select_locals.append(w)
    #
    #             # store testing and training accuracies of the model before global aggregation 测试本地模型聚合前的训练集和测试集准确度
    #             logging.info("Testing Client Models before aggregation")
    #             logging.info("")
    #             s = 0
    #             for i in idxs_users:
    #                 logging.info("Client {}:".format(i))
    #                 acc_train, loss_train = test_client(args, dataset_train, train_data_users[i], local_nets[i])
    #                 acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], local_nets[i])
    #                 logging.info("Training accuracy: {:.3f}".format(acc_train))
    #                 logging.info("Testing accuracy: {:.3f}".format(acc_test))
    #                 logging.info("")
    #                 # print(acc_test)
    #                 stats[i][iter]['Before Training accuracy'] = acc_train
    #                 stats[i][iter]['Before Test accuracy'] = acc_test
    #                 writer.add_scalar(str(i) + '/Before Training accuracy', acc_train, iter)
    #                 writer.add_scalar(str(i) + '/Before Test accuracy', acc_test, iter)
    #
    #                 s += acc_test
    #             s /= m
    #             logging.info("Average Client accuracy on their test data: {: .3f}".format(s))
    #             stats['Before Average'][iter] = s
    #             writer.add_scalar('Average' + '/Before Test accuracy', s, iter)
    #
    #             # update global weights
    #             w_new_glob = FedAvg(w_select_locals)
    #
    #             for i in list(w_glob.keys())[0:base_layers]:
    #                 w_glob[i] = copy.deepcopy(w_new_glob[i])
    #
    #             # copy weight to net_glob
    #             net_glob.load_state_dict(w_glob)  # 将全局模型参数w_glob加载到全局模型中，load_state_dict()是将状态字典加载到模型中
    #
    #             # Updating base layers of the clients and keeping the personalized layers same  # 更新本地模型的基础层
    #             for idx in range(args.num_users):
    #                 for i in list(w_glob.keys())[0:base_layers]:
    #                     w_locals[idx][i] = copy.deepcopy(w_glob[i])  # w_locals[idx][i]：表示第idx个本地模型的第i层
    #
    #                 local_nets[idx].load_state_dict(w_locals[idx])
    #
    #             # store train and test accuracies after updating local models
    #             logging.info("Testing Client Models after aggregation")
    #             logging.info("")
    #             s = 0
    #             var = 0
    #             for i in range(args.num_users):
    #                 logging.info("Client {}:".format(i))
    #                 acc_train, loss_train = test_client(args, dataset_train, train_data_users[i], net_glob)
    #                 acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], net_glob)
    #                 logging.info("Training accuracy: {:.3f}".format(acc_train))
    #                 logging.info("Testing accuracy: {:.3f}".format(acc_test))
    #                 logging.info("")
    #
    #                 stats[i][iter]['After Training accuracy'] = acc_train
    #                 stats[i][iter]['After Test accuracy'] = acc_test
    #                 writer.add_scalar(str(i) + '/After Training accuracy', acc_train, iter)
    #                 writer.add_scalar(str(i) + '/After Test accuracy', acc_test, iter)
    #
    #                 s += acc_test
    #             s /= args.num_users
    #             for i in range(args.num_users):
    #                 acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], net_glob)
    #                 var += (acc_test - s) ** 2
    #             var /= args.num_users
    #             logging.info("Average Client accuracy on their test data: {: .3f}".format(s))
    #             logging.info("Average Client accuracy Variance on their test data: {: .3f}".format(var))
    #
    #             stats['After Average'][iter] = s
    #             writer.add_scalar('Average' + '/After Test accuracy', s, iter)
    #
    #             loss_avg = sum(loss_select_locals) / len(loss_select_locals)
    #             logging.info('Average loss of clients: {:.3f}'.format(loss_avg))
    #
    # end = time.time()
    # logging.info("Training Time: {}s".format(end - start))
    # logging.info("End of Training")
    #
    # # save model parameters
    # # 保存每个客户端的本地模型参数
    # torch.save(net_glob.state_dict(), './state_dict/server_{}.pt'.format(file_name))
    # for i in range(args.num_users):
    #     torch.save(local_nets[i].state_dict(), './state_dict/client_{}_{}.pt'.format(i, file_name))
    #
    # # test global model on training set and testing set
    # # 在训练集和测试集上测试全局模型
    #
    # logging.info("")
    # logging.info("Testing")
    #
    # logging.info("Global Server Model")
    # net_glob.eval()
    # acc_train, loss_train = test_img(net_glob, dataset_train, args)
    # acc_test, loss_test = test_img(net_glob, dataset_test, args)
    # logging.info("Training accuracy of Server: {:.3f}".format(acc_train))
    # logging.info("Training loss of Server: {:.3f}".format(loss_train))
    # logging.info("Testing accuracy of Server: {:.3f}".format(acc_test))
    # logging.info("Testing loss of Server: {:.3f}".format(loss_test))
    # logging.info("End of Server Model Testing")
    # logging.info("")
    #
    # logging.info("Client Models")
    # s = 0
    # # testing local models
    # # 测试本地模型
    # for i in range(args.num_users):
    #     logging.info("Client {}:".format(i))
    #     acc_train, loss_train = test_client(args, dataset_train, train_data_users[i], local_nets[i])
    #     acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], local_nets[i])
    #     logging.info("Training accuracy: {:.3f}".format(acc_train))
    #     logging.info("Training loss: {:.3f}".format(loss_train))
    #     logging.info("Testing accuracy: {:.3f}".format(acc_test))
    #     logging.info("Testing loss: {:.3f}".format(loss_test))
    #     logging.info("")
    #     s += acc_test
    # s /= args.num_users
    # logging.info("Average Client accuracy on their test data: {: .3f}".format(s))
    # logging.info("End of Client Model testing")
    #
    # logging.info("")
    # logging.info("Testing global model on individual client's test data")
    #
    # # testing global model on individual client's test data
    # # 在每个客户的测试数据上测试全局模型
    # s = 0
    # var = 0
    # for i in range(args.num_users):
    #     logging.info("Client {}".format(i))
    #     acc_train, loss_train = test_client(args, dataset_train, train_data_users[i], net_glob)
    #     acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], net_glob)
    #     logging.info("Training accuracy: {:.3f}".format(acc_train))
    #     logging.info("Testing accuracy: {:.3f}".format(acc_test))
    #     s += acc_test
    # s /= args.num_users  # 测试集的平均准确度
    # for i in range(args.num_users):
    #     acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], net_glob)
    #     var += (acc_test - s) ** 2
    # var /= args.num_users
    # logging.info("Average Client accuracy of global model on each client's test data: {: .3f}".format(s))
    # logging.info("Average Client accuracy Variance of global model on each client's test data: {: .3f}".format(var))
    #
    # dill.dump(stats, open(os.path.join(args.summary, 'stats.pkl'), 'wb'))
    # writer.close()
    # print(stats['After Average'])
    # print(stats['After finetune Average'])


##########################################################FedAR#########################################################

        print('Round {}'.format(iter))
        logging.info("---------Round {}---------".format(iter))

        if iter < (args.epochs) * 0.4:
            if iter % 2 == 0:

                w_locals, loss_locals = [], []
                w_select_locals, loss_select_locals = [], []
                select_Fk = []  # select_Fk是一个列表，保存被选择客户端的历史损失
                select_pk = []
                eps_user = [1] * args.num_users  # 保存每个用户的隐私预算
                # hyperparameter = number of layers we want to keep in the base part
                base_layers = args.base_layers

                for idx in range(args.num_users):
                    w_locals.append(local_nets[idx].state_dict())

                m = max(int(args.frac * args.num_users), 1)
                idxs_users = np.random.choice(range(args.num_users), m, replace=False)

                # 测试全局模型在所有用户的损失
                for i in range(args.num_users):
                    acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], net_glob)
                    loss_locals.append(copy.deepcopy(loss_test))
                    loss_history[i][iter] = loss_test

                # 计算全局模型在所有用户的平均损失
                avg_loss = 0
                for i in range(args.num_users):
                    avg_loss += loss_history[i][iter]
                avg_loss /= args.num_users
                loss_history_global_avg.append(copy.deepcopy(avg_loss))

                # if iter < 2:
                #     delta_F[iter] = 0.5
                # else:
                #     delta_loss = loss_history_global_avg[iter - 1] - loss_history_global_avg[iter - 2]
                #     if delta_loss >= args.delta and delta_loss <= delta_F[iter - 1]:
                #         delta_F[iter] = delta_loss
                #     elif delta_loss < 0 or delta_loss > delta_F[iter - 1]:
                #         delta_F[iter] = delta_F[iter - 1]
                #     else:
                #         delta_F[iter] = 0
                # if iter < (args.epochs):
                #     eps_t = math.exp(-delta_F[iter]) * ((eps - eps_c) / (args.epochs - iter + 1))  # 本轮的衰减预算
                #     eps_c += eps_t
                # else:
                #     eps_t = eps - eps_c

                # print("exp(-delta_F[iter]):", math.exp(-delta_F[iter]))

                # for idx in idxs_users:
                #     user_eps_history[idx][iter] = eps_t
                #     eps_history[idx] += eps_t

                # print("本轮的衰减预算:", eps_t)

                # 隐私预算均匀递增
                eps_t = eps_s + iter * 0.002 * eps / 100
                print("本轮的预算:", eps_t)

                # 计算所有客户端的历史衰减损失user_loss_history
                for i in range(args.num_users):
                    F_i = 0
                    F_m = 0
                    for j in range(0, iter + 1):
                        F_i += (args.rou ** (iter - j)) * loss_history[i][j]
                        F_m += (args.rou ** (iter - j))
                    user_loss_history[i][iter] = F_i / F_m

                for idx in idxs_users:
                    select_Fk.append(user_loss_history[idx][iter])
                    select_pk.append(pk[idx])

                # # 计算每个用户本轮的隐私预算
                # F_max = max(select_Fk)
                # for idx in idxs_users:
                #     eps_user[idx] = eps_t * (select_Fk[idx] / F_max)
                #     eps_user[idx] = max(eps_user[idx], 0.8 * eps_t)
                #     eps_history[idx] += eps_user[idx]
                #     user_eps_history[idx][iter] = eps_user[idx]


                for idx in idxs_users:
                    w, loss = train_client(args, dataset_train, train_data_users[idx], net=local_nets[idx],
                                           eps_user=args.eps0)
                    w_select_locals.append(w)
                    loss_select_locals.append(copy.deepcopy(loss))

                # for idx in idxs_users:
                #     w, loss = train_client(args, dataset_train, train_data_users[idx], net=local_nets[idx])
                #     w_select_locals.append(w)
                #     acc_test, loss_test = test_client(args, dataset_train, test_data_users[idx], net_glob)
                #     loss_select_locals.append(copy.deepcopy(loss_test))
                #     loss_history[idx][iter] = loss_test


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

            else:

                w_locals, loss_locals = [], []
                w_select_locals, loss_select_locals = [], []
                select_Fk = []  # select_Fk是一个列表，保存被选择客户端的历史损失
                select_pk = []
                eps_user = [1] * args.num_users  # 保存每个用户的隐私预算
                # hyperparameter = number of layers we want to keep in the base part
                base_layers = args.base_layers

                for idx in range(args.num_users):
                    w_locals.append(local_nets[idx].state_dict())

                m = max(int(args.frac * args.num_users), 1)
                idxs_users = np.random.choice(range(args.num_users), m, replace=False)

                # 测试全局模型在所有用户的损失
                for i in range(args.num_users):
                    acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], net_glob)
                    loss_locals.append(copy.deepcopy(loss_test))
                    loss_history[i][iter] = loss_test

                # 计算全局模型在所有用户的平均损失
                avg_loss = 0
                for i in range(args.num_users):
                    avg_loss += loss_history[i][iter]
                avg_loss /= args.num_users
                loss_history_global_avg.append(copy.deepcopy(avg_loss))

                # if iter < 2:
                #     delta_F[iter] = 0.5
                # else:
                #     delta_loss = loss_history_global_avg[iter - 1] - loss_history_global_avg[iter - 2]
                #     if delta_loss >= args.delta and delta_loss <= delta_F[iter - 1]:
                #         delta_F[iter] = delta_loss
                #     elif delta_loss < 0 or delta_loss > delta_F[iter - 1]:
                #         delta_F[iter] = delta_F[iter - 1]
                #     else:
                #         delta_F[iter] = 0
                # if iter < (args.epochs):
                #     eps_t = math.exp(-delta_F[iter]) * ((eps - eps_c) / (args.epochs - iter + 1))  # 本轮的衰减预算
                #     eps_c += eps_t
                # else:
                #     eps_t = eps - eps_c

                # print("exp(-delta_F[iter]):", math.exp(-delta_F[iter]))

                # for idx in idxs_users:
                #     user_eps_history[idx][iter] = eps_t
                #     eps_history[idx] += eps_t

                # print("本轮的衰减预算:", eps_t)

                # 隐私预算均匀递增
                eps_t = eps_s + iter * 0.002 * eps / 100
                print("本轮的预算:", eps_t)

                # 计算所有客户端的历史衰减损失user_loss_history
                for i in range(args.num_users):
                    F_i = 0
                    F_m = 0
                    for j in range(0, iter + 1):
                        F_i += (args.rou ** (iter - j)) * loss_history[i][j]
                        F_m += (args.rou ** (iter - j))
                    user_loss_history[i][iter] = F_i / F_m

                for idx in idxs_users:
                    select_Fk.append(user_loss_history[idx][iter])
                    select_pk.append(pk[idx])

                # # 计算每个用户本轮的隐私预算
                # F_max = max(select_Fk)
                # for idx in idxs_users:
                #     eps_user[idx] = eps_t * (select_Fk[idx] / F_max)
                #     eps_user[idx] = max(eps_user[idx], 0.8 * eps_t)
                #     eps_history[idx] += eps_user[idx]
                #     user_eps_history[idx][iter] = eps_user[idx]

                for idx in idxs_users:
                    w, loss = train_client(args, dataset_train, train_data_users[idx], net=local_nets[idx],
                                           eps_user=args.eps0)
                    w_select_locals.append(w)
                    loss_select_locals.append(copy.deepcopy(loss))

                # for idx in idxs_users:
                #     w, loss = train_client(args, dataset_train, train_data_users[idx], net=local_nets[idx])
                #     w_select_locals.append(w)
                #     acc_test, loss_test = test_client(args, dataset_train, test_data_users[idx], net_glob)
                #     loss_select_locals.append(copy.deepcopy(loss_test))
                #     loss_history[idx][iter] = loss_test

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

                # Updating base layers of the clients and keeping the personalized layers same  # 更新本地模型的基础层
                for idx in range(args.num_users):
                    for i in list(w_glob.keys())[0:base_layers]:
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

            # 保存客户端历史40轮消耗的隐私预算
            logging.info("Historical Consumption Budget in 40 rounds: %s", eps_history)

        else:
            if iter % 4 == 0:

                w_locals, loss_locals = [], []
                w_select_locals, loss_select_locals = [], []
                select_Fk = []  # select_Fk是一个列表，保存被选择客户端的历史损失
                select_pk = []
                eps_user = [1] * args.num_users  # 保存每个用户的隐私预算
                # hyperparameter = number of layers we want to keep in the base part
                base_layers = args.base_layers

                for idx in range(args.num_users):
                    w_locals.append(local_nets[idx].state_dict())

                m = max(int(args.frac * args.num_users), 1)
                idxs_users = np.random.choice(range(args.num_users), m, replace=False)

                # 测试全局模型在所有用户的损失
                for i in range(args.num_users):
                    acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], net_glob)
                    loss_locals.append(copy.deepcopy(loss_test))
                    loss_history[i][iter] = loss_test

                # 计算全局模型在所有用户的平均损失
                avg_loss = 0
                for i in range(args.num_users):
                    avg_loss += loss_history[i][iter]
                avg_loss /= args.num_users
                loss_history_global_avg.append(copy.deepcopy(avg_loss))

                # if iter < 2:
                #     delta_F[iter] = 0.5
                # else:
                #     delta_loss = loss_history_global_avg[iter - 1] - loss_history_global_avg[iter - 2]
                #     if delta_loss >= args.delta and delta_loss <= delta_F[iter - 1]:
                #         delta_F[iter] = delta_loss
                #     elif delta_loss < 0 or delta_loss > delta_F[iter - 1]:
                #         delta_F[iter] = delta_F[iter - 1]
                #     else:
                #         delta_F[iter] = 0
                # if iter < (args.epochs):
                #     eps_t = math.exp(-delta_F[iter]) * ((eps - eps_c) / (args.epochs - iter + 1))  # 本轮的衰减预算
                #     eps_c += eps_t
                # else:
                #     eps_t = eps - eps_c
                #
                # print("exp(-delta_F[iter]):", math.exp(-delta_F[iter]))

                # for idx in idxs_users:
                #     user_eps_history[idx][iter] = eps_t
                #     eps_history[idx] += eps_t

                # print("本轮的衰减预算:", eps_t)

                # 隐私预算均匀递增
                eps_t = eps_s + iter * 0.002 * eps / 100
                print("本轮的预算:", eps_t)

                # 计算所有客户端的历史衰减损失user_loss_history
                for i in range(args.num_users):
                    F_i = 0
                    F_m = 0
                    for j in range(0, iter + 1):
                        F_i += (args.rou ** (iter - j)) * loss_history[i][j]
                        F_m += (args.rou ** (iter - j))
                    user_loss_history[i][iter] = F_i / F_m

                for idx in idxs_users:
                    select_Fk.append(user_loss_history[idx][iter])
                    select_pk.append(pk[idx])

                # # 计算每个用户本轮的隐私预算
                # F_max = max(select_Fk)
                # for idx in idxs_users:
                #     eps_user[idx] = eps_t * (select_Fk[idx] / F_max)
                #     eps_user[idx] = max(eps_user[idx], 0.8 * eps_t)
                #     eps_history[idx] += eps_user[idx]
                #     user_eps_history[idx][iter] = eps_user[idx]

                for idx in idxs_users:
                    w, loss = train_client(args, dataset_train, train_data_users[idx], net=local_nets[idx],
                                           eps_user=args.eps0)
                    w_select_locals.append(w)
                    loss_select_locals.append(copy.deepcopy(loss))

                # for idx in idxs_users:
                #     w, loss = train_client(args, dataset_train, train_data_users[idx], net=local_nets[idx])
                #     w_select_locals.append(w)
                #     acc_test, loss_test = test_client(args, dataset_train, test_data_users[idx], net_glob)
                #     loss_select_locals.append(copy.deepcopy(loss_test))
                #     loss_history[idx][iter] = loss_test

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

            else:

                w_locals, loss_locals = [], []
                w_select_locals, loss_select_locals = [], []
                select_Fk = []  # select_Fk是一个列表，保存被选择客户端的历史损失
                select_pk = []
                eps_user = [1] * args.num_users  # 保存每个用户的隐私预算
                # hyperparameter = number of layers we want to keep in the base part
                base_layers = args.base_layers

                for idx in range(args.num_users):
                    w_locals.append(local_nets[idx].state_dict())

                m = max(int(args.frac * args.num_users), 1)
                idxs_users = np.random.choice(range(args.num_users), m, replace=False)

                # 测试全局模型在所有用户的损失
                for i in range(args.num_users):
                    acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], net_glob)
                    loss_locals.append(copy.deepcopy(loss_test))
                    loss_history[i][iter] = loss_test

                # 计算全局模型在所有用户的平均损失
                avg_loss = 0
                for i in range(args.num_users):
                    avg_loss += loss_history[i][iter]
                avg_loss /= args.num_users
                loss_history_global_avg.append(copy.deepcopy(avg_loss))

                # if iter < 2:
                #     delta_F[iter] = 0.5
                # else:
                #     delta_loss = loss_history_global_avg[iter - 1] - loss_history_global_avg[iter - 2]
                #     if delta_loss >= args.delta and delta_loss <= delta_F[iter - 1]:
                #         delta_F[iter] = delta_loss
                #     elif delta_loss < 0 or delta_loss > delta_F[iter - 1]:
                #         delta_F[iter] = delta_F[iter - 1]
                #     else:
                #         delta_F[iter] = 0
                # if iter < (args.epochs):
                #     eps_t = math.exp(-delta_F[iter]) * ((eps - eps_c) / (args.epochs - iter + 1))  # 本轮的衰减预算
                #     eps_c += eps_t
                # else:
                #     eps_t = eps - eps_c
                #
                # print("exp(-delta_F[iter]):", math.exp(-delta_F[iter]))

                # for idx in idxs_users:
                #     user_eps_history[idx][iter] = eps_t
                #     eps_history[idx] += eps_t

                # print("本轮的衰减预算:", eps_t)

                # 隐私预算均匀递增
                eps_t = eps_s + iter * 0.002 * eps / 100
                print("本轮的预算:", eps_t)

                # 计算所有客户端的历史衰减损失user_loss_history
                for i in range(args.num_users):
                    F_i = 0
                    F_m = 0
                    for j in range(0, iter + 1):
                        F_i += (args.rou ** (iter - j)) * loss_history[i][j]
                        F_m += (args.rou ** (iter - j))
                    user_loss_history[i][iter] = F_i / F_m

                for idx in idxs_users:
                    select_Fk.append(user_loss_history[idx][iter])
                    select_pk.append(pk[idx])

                # # 计算每个用户本轮的隐私预算
                # F_max = max(select_Fk)
                # for idx in idxs_users:
                #     eps_user[idx] = eps_t * (select_Fk[idx] / F_max)
                #     eps_user[idx] = max(eps_user[idx], 0.8 * eps_t)
                #     eps_history[idx] += eps_user[idx]
                #     user_eps_history[idx][iter] = eps_user[idx]

                for idx in idxs_users:
                    w, loss = train_client(args, dataset_train, train_data_users[idx], net=local_nets[idx],
                                           eps_user=args.eps0)
                    w_select_locals.append(w)
                    loss_select_locals.append(copy.deepcopy(loss))

                # for idx in idxs_users:
                #     w, loss = train_client(args, dataset_train, train_data_users[idx], net=local_nets[idx])
                #     w_select_locals.append(w)
                #     acc_test, loss_test = test_client(args, dataset_train, test_data_users[idx], net_glob)
                #     loss_select_locals.append(copy.deepcopy(loss_test))
                #     loss_history[idx][iter] = loss_test

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

                # Updating base layers of the clients and keeping the personalized layers same  # 更新本地模型的基础层
                for idx in range(args.num_users):
                    for i in list(w_glob.keys())[0:base_layers]:
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

            # 保存客户端历史40轮消耗的隐私预算
            logging.info("Historical Consumption Budget in 80 rounds: %s", eps_history)

        # for idx in idxs_users:
        #     if eps_history[idx] >= eps:
        #         terminate_outer_loop = True
        #         break  # 终止内循环
        #
        # if terminate_outer_loop:
        #     break  # 终止外循环

    end = time.time()
    logging.info("Training Time: {}s".format(end - start))
    logging.info("End of Training")

    # save model parameters
    # 保存每个客户端的本地模型参数
    torch.save(net_glob.state_dict(), './state_dict/server_{}.pt'.format(file_name))
    for i in range(args.num_users):
        torch.save(local_nets[i].state_dict(), './state_dict/client_{}_{}.pt'.format(i, file_name))

    # test global model on training set and testing set
    # 在训练集和测试集上测试全局模型

    logging.info("")
    logging.info("Testing")

    logging.info("Global Server Model")
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    logging.info("Training accuracy of Server: {:.3f}".format(acc_train))
    logging.info("Training loss of Server: {:.3f}".format(loss_train))
    logging.info("Testing accuracy of Server: {:.3f}".format(acc_test))
    logging.info("Testing loss of Server: {:.3f}".format(loss_test))
    logging.info("End of Server Model Testing")
    logging.info("")

    logging.info("Client Models")
    s = 0
    # testing local models
    # 测试本地模型
    for i in range(args.num_users):
        logging.info("Client {}:".format(i))
        acc_train, loss_train = test_client(args, dataset_train, train_data_users[i], local_nets[i])
        acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], local_nets[i])
        logging.info("Training accuracy: {:.3f}".format(acc_train))
        logging.info("Training loss: {:.3f}".format(loss_train))
        logging.info("Testing accuracy: {:.3f}".format(acc_test))
        logging.info("Testing loss: {:.3f}".format(loss_test))
        logging.info("")
        s += acc_test
    s /= args.num_users
    logging.info("Average Client accuracy on their test data: {: .3f}".format(s))
    logging.info("End of Client Model testing")

    logging.info("")
    logging.info("Testing global model on individual client's test data")

    # testing global model on individual client's test data
    # 在每个客户的测试数据上测试全局模型
    s = 0
    var = 0
    for i in range(args.num_users):
        logging.info("Client {}".format(i))
        acc_train, loss_train = test_client(args, dataset_train, train_data_users[i], net_glob)
        acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], net_glob)
        logging.info("Training accuracy: {:.3f}".format(acc_train))
        logging.info("Testing accuracy: {:.3f}".format(acc_test))
        s += acc_test
    s /= args.num_users  # 测试集的平均准确度
    for i in range(args.num_users):
        acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], net_glob)
        var += (acc_test - s) ** 2
    var /= args.num_users
    logging.info("Average Client accuracy of global model on each client's test data: {: .3f}".format(s))
    logging.info("Average Client accuracy Variance of global model on each client's test data: {: .3f}".format(var))

    # 保存客户端历史权重
    logging.info("Historical weight: %s", user_weight_history)

    # 保存客户端历史消耗的隐私预算
    logging.info("Historical Consumption Budget: %s", eps_history)

    # 保存客户端历史每轮消耗的隐私预算
    logging.info("Historical privacy budget per round: %s", user_eps_history)

    dill.dump(stats, open(os.path.join(args.summary, 'stats.pkl'), 'wb'))
    writer.close()
    print(stats['After Average'])
    print(stats['After finetune Average'])


######################################################DR-FedVF#######################################################

    #     print('Round {}'.format(iter))
    #     logging.info("---------Round {}---------".format(iter))
    #
    #     w_locals, loss_locals = [], []  # w_locals保存客户端本地模型参数，loss_locals保存本地模型的loss
    #     w_select_locals, loss_select_locals = [], []
    #     select_Fk = []  # select_Fk是一个列表，保存被选择客户端的损失
    #     select_pk = []
    #     # hyperparameter = number of layers we want to keep in the base part
    #     base_layers = args.base_layers
    #
    #     for idx in range(args.num_users):
    #         w_locals.append(local_nets[idx].state_dict())
    #
    #     m = max(int(args.frac * args.num_users), 1)
    #     idxs_users = np.random.choice(range(args.num_users), m, replace=False)
    #
    #     if iter < (args.epochs) * 0.4:
    #         if iter % 2 == 0:
    #             for idx in idxs_users:
    #                 w, loss = train_client(args, dataset_train, train_data_users[idx], net=local_nets[idx])
    #                 loss_select_locals.append(copy.deepcopy(loss))
    #                 w_select_locals.append(w)
    #                 acc_test, loss_test = test_client(args, dataset_train, test_data_users[idx], net_glob)
    #                 select_Fk.append(loss_test)
    #                 select_pk.append(pk[idx])
    #
    #             # store testing and training accuracies of the model before global aggregation 测试本地模型聚合前的训练集和测试集准确度
    #             logging.info("Testing Client Models before aggregation")
    #             logging.info("")
    #             s = 0
    #             for i in idxs_users:
    #                 logging.info("Client {}:".format(i))
    #                 acc_train, loss_train = test_client(args, dataset_train, train_data_users[i], local_nets[i])
    #                 acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], local_nets[i])
    #                 logging.info("Training accuracy: {:.3f}".format(acc_train))
    #                 logging.info("Testing accuracy: {:.3f}".format(acc_test))
    #                 logging.info("")
    #                 # print(acc_test)
    #                 stats[i][iter]['Before Training accuracy'] = acc_train
    #                 stats[i][iter]['Before Test accuracy'] = acc_test
    #                 writer.add_scalar(str(i) + '/Before Training accuracy', acc_train, iter)
    #                 writer.add_scalar(str(i) + '/Before Test accuracy', acc_test, iter)
    #
    #                 s += acc_test
    #             s /= m
    #             logging.info("Average Client accuracy on their test data: {: .3f}".format(s))
    #             stats['Before Average'][iter] = s
    #             writer.add_scalar('Average' + '/Before Test accuracy', s, iter)
    #
    #             # update global weights
    #             w_glob, e = DR_FedAvg(args, w_select_locals, select_pk, select_Fk)
    #
    #             # for idx in idxs_users:
    #             #     user_weight_history[idx][iter] = e[idx]
    #
    #             # copy weight to net_glob
    #             net_glob.load_state_dict(w_glob)  # 将全局模型参数w_glob加载到全局模型中，load_state_dict()是将状态字典加载到模型中
    #
    #             # Updating base layers of the clients and keeping the personalized layers same  # 更新本地模型的所有层
    #             for idx in range(args.num_users):
    #                 for i in list(w_glob.keys()):
    #                     w_locals[idx][i] = copy.deepcopy(w_glob[i])  # w_locals[idx][i]：表示第idx个本地模型的第i层
    #
    #                 local_nets[idx].load_state_dict(w_locals[idx])
    #
    #             # store train and test accuracies after updating local models
    #             logging.info("Testing global Models after aggregation")
    #             logging.info("")
    #             s = 0
    #             var = 0
    #             for i in range(args.num_users):
    #                 logging.info("Client {}:".format(i))
    #                 acc_train, loss_train = test_client(args, dataset_train, train_data_users[i], net_glob)
    #                 acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], net_glob)
    #                 logging.info("Training accuracy: {:.3f}".format(acc_train))
    #                 logging.info("Testing accuracy: {:.3f}".format(acc_test))
    #                 logging.info("")
    #
    #                 stats[i][iter]['After Training accuracy'] = acc_train
    #                 stats[i][iter]['After Test accuracy'] = acc_test
    #                 writer.add_scalar(str(i) + '/After Training accuracy', acc_train, iter)
    #                 writer.add_scalar(str(i) + '/After Test accuracy', acc_test, iter)
    #
    #                 s += acc_test
    #             s /= args.num_users
    #             for i in range(args.num_users):
    #                 acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], net_glob)
    #                 var += (acc_test - s) ** 2
    #             var /= args.num_users
    #             logging.info("Average global model accuracy on their test data: {: .3f}".format(s))
    #             logging.info("Average global model Variance on their test data: {: .3f}".format(var))
    #
    #             stats['After Average'][iter] = s
    #             writer.add_scalar('Average' + '/After Test accuracy', s, iter)
    #
    #             loss_avg = sum(loss_select_locals) / len(loss_select_locals)
    #             logging.info('Average loss of clients: {:.3f}'.format(loss_avg))
    #
    #         else:
    #             for idx in idxs_users:
    #                 w, loss = train_client(args, dataset_train, train_data_users[idx], net=local_nets[idx])
    #                 loss_select_locals.append(copy.deepcopy(loss))
    #                 w_select_locals.append(w)
    #                 acc_test, loss_test = test_client(args, dataset_train, test_data_users[idx], net_glob)
    #                 select_Fk.append(loss_test)
    #                 select_pk.append(pk[idx])
    #
    #             # store testing and training accuracies of the model before global aggregation 测试本地模型聚合前的训练集和测试集准确度
    #             logging.info("Testing Client Models before aggregation")
    #             logging.info("")
    #             s = 0
    #             for i in idxs_users:
    #                 logging.info("Client {}:".format(i))
    #                 acc_train, loss_train = test_client(args, dataset_train, train_data_users[i], local_nets[i])
    #                 acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], local_nets[i])
    #                 logging.info("Training accuracy: {:.3f}".format(acc_train))
    #                 logging.info("Testing accuracy: {:.3f}".format(acc_test))
    #                 logging.info("")
    #                 # print(acc_test)
    #                 stats[i][iter]['Before Training accuracy'] = acc_train
    #                 stats[i][iter]['Before Test accuracy'] = acc_test
    #                 writer.add_scalar(str(i) + '/Before Training accuracy', acc_train, iter)
    #                 writer.add_scalar(str(i) + '/Before Test accuracy', acc_test, iter)
    #
    #                 s += acc_test
    #             s /= m
    #             logging.info("Average Client accuracy on their test data: {: .3f}".format(s))
    #             stats['Before Average'][iter] = s
    #             writer.add_scalar('Average' + '/Before Test accuracy', s, iter)
    #
    #             # # update global weights
    #             # w_new_glob = DR_FedAvg(args, w_select_locals, select_pk, select_Fk)
    #             #
    #             # for i in list(w_glob.keys())[0:base_layers_1]:
    #             #     w_glob[i] = copy.deepcopy(w_new_glob[i])
    #
    #             # update global weights
    #             w_glob, e = DR_FedAvg(args, w_select_locals, select_pk, select_Fk)
    #
    #             # for idx in idxs_users:
    #             #     user_weight_history[idx][iter] = e[idx]
    #
    #             # copy weight to net_glob
    #             net_glob.load_state_dict(w_glob)  # 将全局模型参数w_glob加载到全局模型中，load_state_dict()是将状态字典加载到模型中
    #
    #             # Updating base layers of the clients and keeping the personalized layers same  # 更新本地模型的基础层
    #             for idx in range(args.num_users):
    #                 for i in list(w_glob.keys())[0:base_layers]:
    #                     w_locals[idx][i] = copy.deepcopy(w_glob[i])  # w_locals[idx][i]：表示第idx个本地模型的第i层
    #
    #                 local_nets[idx].load_state_dict(w_locals[idx])
    #
    #             # store train and test accuracies after updating local models
    #             logging.info("Testing global Models after aggregation")
    #             logging.info("")
    #             s = 0
    #             var = 0
    #             for i in range(args.num_users):
    #                 logging.info("Client {}:".format(i))
    #                 acc_train, loss_train = test_client(args, dataset_train, train_data_users[i], net_glob)
    #                 acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], net_glob)
    #                 logging.info("Training accuracy: {:.3f}".format(acc_train))
    #                 logging.info("Testing accuracy: {:.3f}".format(acc_test))
    #                 logging.info("")
    #
    #                 stats[i][iter]['After Training accuracy'] = acc_train
    #                 stats[i][iter]['After Test accuracy'] = acc_test
    #                 writer.add_scalar(str(i) + '/After Training accuracy', acc_train, iter)
    #                 writer.add_scalar(str(i) + '/After Test accuracy', acc_test, iter)
    #
    #                 s += acc_test
    #             s /= args.num_users
    #             for i in range(args.num_users):
    #                 acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], net_glob)
    #                 var += (acc_test - s) ** 2
    #             var /= args.num_users
    #             logging.info("Average global model accuracy on their test data: {: .3f}".format(s))
    #             logging.info("Average global model Variance on their test data: {: .3f}".format(var))
    #
    #             stats['After Average'][iter] = s
    #             writer.add_scalar('Average' + '/After Test accuracy', s, iter)
    #
    #             loss_avg = sum(loss_select_locals) / len(loss_select_locals)
    #             logging.info('Average loss of clients: {:.3f}'.format(loss_avg))
    #
    #     else:
    #         if iter % 4 == 0:
    #             for idx in idxs_users:
    #                 w, loss = train_client(args, dataset_train, train_data_users[idx], net=local_nets[idx])
    #                 loss_select_locals.append(copy.deepcopy(loss))
    #                 w_select_locals.append(w)
    #                 acc_test, loss_test = test_client(args, dataset_train, test_data_users[idx], net_glob)
    #                 select_Fk.append(loss_test)
    #                 select_pk.append(pk[idx])
    #
    #
    #             # store testing and training accuracies of the model before global aggregation 测试本地模型聚合前的训练集和测试集准确度
    #             logging.info("Testing Client Models before aggregation")
    #             logging.info("")
    #             s = 0
    #             for i in idxs_users:
    #                 logging.info("Client {}:".format(i))
    #                 acc_train, loss_train = test_client(args, dataset_train, train_data_users[i], local_nets[i])
    #                 acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], local_nets[i])
    #                 logging.info("Training accuracy: {:.3f}".format(acc_train))
    #                 logging.info("Testing accuracy: {:.3f}".format(acc_test))
    #                 logging.info("")
    #                 # print(acc_test)
    #                 stats[i][iter]['Before Training accuracy'] = acc_train
    #                 stats[i][iter]['Before Test accuracy'] = acc_test
    #                 writer.add_scalar(str(i) + '/Before Training accuracy', acc_train, iter)
    #                 writer.add_scalar(str(i) + '/Before Test accuracy', acc_test, iter)
    #
    #                 s += acc_test
    #             s /= m
    #             logging.info("Average Client accuracy on their test data: {: .3f}".format(s))
    #             stats['Before Average'][iter] = s
    #             writer.add_scalar('Average' + '/Before Test accuracy', s, iter)
    #
    #             # update global weights
    #             w_glob, e = DR_FedAvg(args, w_select_locals, select_pk, select_Fk)
    #
    #             # for idx in idxs_users:
    #             #     user_weight_history[idx][iter] = e[idx]
    #
    #             # copy weight to net_glob
    #             net_glob.load_state_dict(w_glob)  # 将全局模型参数w_glob加载到全局模型中，load_state_dict()是将状态字典加载到模型中
    #
    #             # Updating base layers of the clients and keeping the personalized layers same  # 更新本地模型的所有层
    #             for idx in range(args.num_users):
    #                 for i in list(w_glob.keys()):
    #                     w_locals[idx][i] = copy.deepcopy(w_glob[i])  # w_locals[idx][i]：表示第idx个本地模型的第i层
    #
    #                 local_nets[idx].load_state_dict(w_locals[idx])
    #
    #             # store train and test accuracies after updating local models
    #             logging.info("Testing global Models after aggregation")
    #             logging.info("")
    #             s = 0
    #             var = 0
    #             for i in range(args.num_users):
    #                 logging.info("Client {}:".format(i))
    #                 acc_train, loss_train = test_client(args, dataset_train, train_data_users[i], net_glob)
    #                 acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], net_glob)
    #                 logging.info("Training accuracy: {:.3f}".format(acc_train))
    #                 logging.info("Testing accuracy: {:.3f}".format(acc_test))
    #                 logging.info("")
    #
    #                 stats[i][iter]['After Training accuracy'] = acc_train
    #                 stats[i][iter]['After Test accuracy'] = acc_test
    #                 writer.add_scalar(str(i) + '/After Training accuracy', acc_train, iter)
    #                 writer.add_scalar(str(i) + '/After Test accuracy', acc_test, iter)
    #
    #                 s += acc_test
    #             s /= args.num_users
    #             for i in range(args.num_users):
    #                 acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], net_glob)
    #                 var += (acc_test - s) ** 2
    #             var /= args.num_users
    #             logging.info("Average global model accuracy on their test data: {: .3f}".format(s))
    #             logging.info("Average global model Variance on their test data: {: .3f}".format(var))
    #
    #             stats['After Average'][iter] = s
    #             writer.add_scalar('Average' + '/After Test accuracy', s, iter)
    #
    #             loss_avg = sum(loss_select_locals) / len(loss_select_locals)
    #             logging.info('Average loss of clients: {:.3f}'.format(loss_avg))
    #
    #         else:
    #             for idx in idxs_users:
    #                 w, loss = train_client(args, dataset_train, train_data_users[idx], net=local_nets[idx])
    #                 loss_select_locals.append(copy.deepcopy(loss))
    #                 w_select_locals.append(w)
    #                 acc_test, loss_test = test_client(args, dataset_train, test_data_users[idx], net_glob)
    #                 select_Fk.append(loss_test)
    #                 select_pk.append(pk[idx])
    #
    #             # store testing and training accuracies of the model before global aggregation 测试本地模型聚合前的训练集和测试集准确度
    #             logging.info("Testing Client Models before aggregation")
    #             logging.info("")
    #             s = 0
    #             for i in idxs_users:
    #                 logging.info("Client {}:".format(i))
    #                 acc_train, loss_train = test_client(args, dataset_train, train_data_users[i], local_nets[i])
    #                 acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], local_nets[i])
    #                 logging.info("Training accuracy: {:.3f}".format(acc_train))
    #                 logging.info("Testing accuracy: {:.3f}".format(acc_test))
    #                 logging.info("")
    #                 # print(acc_test)
    #                 stats[i][iter]['Before Training accuracy'] = acc_train
    #                 stats[i][iter]['Before Test accuracy'] = acc_test
    #                 writer.add_scalar(str(i) + '/Before Training accuracy', acc_train, iter)
    #                 writer.add_scalar(str(i) + '/Before Test accuracy', acc_test, iter)
    #
    #                 s += acc_test
    #             s /= m
    #             logging.info("Average Client accuracy on their test data: {: .3f}".format(s))
    #             stats['Before Average'][iter] = s
    #             writer.add_scalar('Average' + '/Before Test accuracy', s, iter)
    #
    #             # # update global weights
    #             # w_new_glob = DR_FedAvg(args, w_select_locals, select_pk, select_Fk)
    #             #
    #             # for i in list(w_glob.keys())[0:base_layers_2]:
    #             #     w_glob[i] = copy.deepcopy(w_new_glob[i])
    #
    #             # update global weights
    #             w_glob, e = DR_FedAvg(args, w_select_locals, select_pk, select_Fk)
    #
    #             # for idx in idxs_users:
    #             #     user_weight_history[idx][iter] = e[idx]
    #
    #             # copy weight to net_glob
    #             net_glob.load_state_dict(w_glob)  # 将全局模型参数w_glob加载到全局模型中，load_state_dict()是将状态字典加载到模型中
    #
    #             # Updating base layers of the clients and keeping the personalized layers same  # 更新本地模型的基础层
    #             for idx in range(args.num_users):
    #                 for i in list(w_glob.keys())[0:base_layers]:
    #                     w_locals[idx][i] = copy.deepcopy(w_glob[i])  # w_locals[idx][i]：表示第idx个本地模型的第i层
    #
    #                 local_nets[idx].load_state_dict(w_locals[idx])
    #
    #             # store train and test accuracies after updating local models
    #             logging.info("Testing global Models after aggregation")
    #             logging.info("")
    #             s = 0
    #             var = 0
    #             for i in range(args.num_users):
    #                 logging.info("Client {}:".format(i))
    #                 acc_train, loss_train = test_client(args, dataset_train, train_data_users[i], net_glob)
    #                 acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], net_glob)
    #                 logging.info("Training accuracy: {:.3f}".format(acc_train))
    #                 logging.info("Testing accuracy: {:.3f}".format(acc_test))
    #                 logging.info("")
    #
    #                 stats[i][iter]['After Training accuracy'] = acc_train
    #                 stats[i][iter]['After Test accuracy'] = acc_test
    #                 writer.add_scalar(str(i) + '/After Training accuracy', acc_train, iter)
    #                 writer.add_scalar(str(i) + '/After Test accuracy', acc_test, iter)
    #
    #                 s += acc_test
    #             s /= args.num_users
    #             for i in range(args.num_users):
    #                 acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], net_glob)
    #                 var += (acc_test - s) ** 2
    #             var /= args.num_users
    #             logging.info("Average global model accuracy on their test data: {: .3f}".format(s))
    #             logging.info("Average global model Variance on their test data: {: .3f}".format(var))
    #
    #             stats['After Average'][iter] = s
    #             writer.add_scalar('Average' + '/After Test accuracy', s, iter)
    #
    #             loss_avg = sum(loss_select_locals) / len(loss_select_locals)
    #             logging.info('Average loss of clients: {:.3f}'.format(loss_avg))
    #
    # end = time.time()
    # logging.info("Training Time: {}s".format(end - start))
    # logging.info("End of Training")
    #
    # # save model parameters
    # # 保存每个客户端的本地模型参数
    # torch.save(net_glob.state_dict(), './state_dict/server_{}.pt'.format(file_name))
    # for i in range(args.num_users):
    #     torch.save(local_nets[i].state_dict(), './state_dict/client_{}_{}.pt'.format(i, file_name))
    #
    # # test global model on training set and testing set
    # # 在训练集和测试集上测试全局模型
    #
    # logging.info("")
    # logging.info("Testing")
    #
    # logging.info("Global Server Model")
    # net_glob.eval()
    # acc_train, loss_train = test_img(net_glob, dataset_train, args)
    # acc_test, loss_test = test_img(net_glob, dataset_test, args)
    # logging.info("Training accuracy of Server: {:.3f}".format(acc_train))
    # logging.info("Training loss of Server: {:.3f}".format(loss_train))
    # logging.info("Testing accuracy of Server: {:.3f}".format(acc_test))
    # logging.info("Testing loss of Server: {:.3f}".format(loss_test))
    # logging.info("End of Server Model Testing")
    # logging.info("")
    #
    # logging.info("Client Models")
    # s = 0
    # # testing local models
    # # 测试本地模型
    # for i in range(args.num_users):
    #     logging.info("Client {}:".format(i))
    #     acc_train, loss_train = test_client(args, dataset_train, train_data_users[i], local_nets[i])
    #     acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], local_nets[i])
    #     logging.info("Training accuracy: {:.3f}".format(acc_train))
    #     logging.info("Training loss: {:.3f}".format(loss_train))
    #     logging.info("Testing accuracy: {:.3f}".format(acc_test))
    #     logging.info("Testing loss: {:.3f}".format(loss_test))
    #     logging.info("")
    #     s += acc_test
    # s /= args.num_users
    # logging.info("Average Client accuracy on their test data: {: .3f}".format(s))
    # logging.info("End of Client Model testing")
    #
    # logging.info("")
    # logging.info("Testing global model on individual client's test data")
    #
    # # testing global model on individual client's test data
    # # 在每个客户的测试数据上测试全局模型
    # s = 0
    # var = 0
    # for i in range(args.num_users):
    #     logging.info("Client {}".format(i))
    #     acc_train, loss_train = test_client(args, dataset_train, train_data_users[i], net_glob)
    #     acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], net_glob)
    #     logging.info("Training accuracy: {:.3f}".format(acc_train))
    #     logging.info("Testing accuracy: {:.3f}".format(acc_test))
    #     s += acc_test
    # s /= args.num_users  # 测试集的平均准确度
    # for i in range(args.num_users):
    #     acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], net_glob)
    #     var += (acc_test - s) ** 2
    # var /= args.num_users
    # logging.info("Average Client accuracy of global model on each client's test data: {: .3f}".format(s))
    # logging.info("Average Client accuracy Variance of global model on each client's test data: {: .3f}".format(var))
    #
    # # 保存客户端历史权重
    # logging.info("Historical weight: %s", user_weight_history)
    #
    # dill.dump(stats, open(os.path.join(args.summary, 'stats.pkl'), 'wb'))
    # writer.close()
    # print(stats['After Average'])
    # print(stats['After finetune Average'])



######################################################FedPer#######################################################

    #     print('Round {}'.format(iter))
    #
    #     logging.info("---------Round {}---------".format(iter))
    #
    #     w_locals, loss_locals = [], []  # w_locals保存本地模型，loss_locals保存本地模型的loss
    #     w_select_locals, loss_select_locals = [], []
    #     # hyperparameter = number of layers we want to keep in the base part
    #     base_layers = args.base_layers
    #
    #     for idx in range(args.num_users):
    #         w_locals.append(local_nets[idx].state_dict())
    #     m = max(int(args.frac * args.num_users), 1)
    #     idxs_users = np.random.choice(range(args.num_users), m, replace=False)
    #
    #     for idx in idxs_users:
    #         w, loss = train_client(args, dataset_train, train_data_users[idx], net=local_nets[idx])
    #         loss_select_locals.append(copy.deepcopy(loss))
    #         w_select_locals.append(w)
    #
    #     # store testing and training accuracies of the model before global aggregation 测试本地模型聚合前的训练集和测试集准确度
    #     logging.info("Testing Client Models before aggregation")
    #     logging.info("")
    #     s = 0
    #     for i in idxs_users:
    #         logging.info("Client {}:".format(i))
    #         acc_train, loss_train = test_client(args, dataset_train, train_data_users[i], local_nets[i])
    #         acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], local_nets[i])
    #         logging.info("Training accuracy: {:.3f}".format(acc_train))
    #         logging.info("Testing accuracy: {:.3f}".format(acc_test))
    #         logging.info("")
    #         # print(acc_test)
    #         stats[i][iter]['Before Training accuracy'] = acc_train
    #         stats[i][iter]['Before Test accuracy'] = acc_test
    #         writer.add_scalar(str(i) + '/Before Training accuracy', acc_train, iter)
    #         writer.add_scalar(str(i) + '/Before Test accuracy', acc_test, iter)
    #
    #         s += acc_test
    #     s /= m
    #     logging.info("Average Client accuracy on their test data: {: .3f}".format(s))
    #     stats['Before Average'][iter] = s
    #     writer.add_scalar('Average' + '/Before Test accuracy', s, iter)
    #
    #     # 更新全局模型
    #     w_new_glob = FedAvg(w_select_locals)  # 传本地模型，返回更新后的全局模型
    #
    #     for i in list(w_glob.keys())[0:base_layers]:
    #         w_glob[i] = copy.deepcopy(w_new_glob[i])
    #
    #     # copy weight to net_glob
    #     if iter == args.epochs:
    #         net_glob.load_state_dict(w_new_glob)  # 将全局模型参数w_glob加载到全局模型中，load_state_dict()是将状态字典加载到模型中
    #
    #     else:
    #         net_glob.load_state_dict(w_glob)
    #     # Updating base layers of the clients and keeping the personalized layers same  更新本地模型的基础层
    #     for idx in range(args.num_users):
    #         for i in list(w_glob.keys())[0:base_layers]:
    #             w_locals[idx][i] = copy.deepcopy(w_glob[i])  # w_locals[idx][i]：表示第idx个本地模型的第i层
    #
    #         local_nets[idx].load_state_dict(w_locals[idx])
    #
    #     # store train and test accuracies after updating local models
    #     logging.info("Testing Client Models after aggregation")
    #     logging.info("")
    #     s = 0
    #     var = 0
    #     for i in range(args.num_users):
    #         logging.info("Client {}:".format(i))
    #         acc_train, loss_train = test_client(args, dataset_train, train_data_users[i], net_glob)
    #         acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], net_glob)
    #         logging.info("Training accuracy: {:.3f}".format(acc_train))
    #         logging.info("Testing accuracy: {:.3f}".format(acc_test))
    #         logging.info("")
    #
    #         stats[i][iter]['After Training accuracy'] = acc_train
    #         stats[i][iter]['After Test accuracy'] = acc_test
    #         writer.add_scalar(str(i) + '/After Training accuracy', acc_train, iter)
    #         writer.add_scalar(str(i) + '/After Test accuracy', acc_test, iter)
    #
    #         s += acc_test
    #     s /= args.num_users
    #     for i in range(args.num_users):
    #         acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], net_glob)
    #         var += (acc_test - s) ** 2
    #     var /= args.num_users
    #     logging.info("Average Client accuracy on their test data: {: .3f}".format(s))
    #     logging.info("Average Client accuracy Variance on their test data: {: .3f}".format(var))
    #
    #     stats['After Average'][iter] = s
    #     writer.add_scalar('Average' + '/After Test accuracy', s, iter)
    #     loss_avg = sum(loss_select_locals) / len(loss_select_locals)
    #
    #     logging.info('Average loss of clients: {:.3f}'.format(loss_avg))
    #
    # logging.info("")
    # logging.info("Testing global model on individual client's test data")
    #
    # # test global model on training set and testing set
    #
    # logging.info("")
    # logging.info("Testing")
    #
    # logging.info("Global Server Model")
    # net_glob.eval()
    # acc_train, loss_train = test_img(net_glob, dataset_train, args)
    # acc_test, loss_test = test_img(net_glob, dataset_test, args)
    # logging.info("Training accuracy of Server: {:.3f}".format(acc_train))
    # logging.info("Training loss of Server: {:.3f}".format(loss_train))
    # logging.info("Testing accuracy of Server: {:.3f}".format(acc_test))
    # logging.info("Testing loss of Server: {:.3f}".format(loss_test))
    # logging.info("End of Server Model Testing")
    # logging.info("")
    #
    # logging.info("Client Models")
    # s = 0
    # # testing local models
    # for i in range(args.num_users):
    #     logging.info("Client {}:".format(i))
    #     acc_train, loss_train = test_client(args, dataset_train, train_data_users[i], local_nets[i])
    #     acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], local_nets[i])
    #     logging.info("Training accuracy: {:.3f}".format(acc_train))
    #     logging.info("Training loss: {:.3f}".format(loss_train))
    #     logging.info("Testing accuracy: {:.3f}".format(acc_test))
    #     logging.info("Testing loss: {:.3f}".format(loss_test))
    #     logging.info("")
    #     s += acc_test
    # s /= args.num_users
    # logging.info("Average Client accuracy on their test data: {: .3f}".format(s))
    # logging.info("End of Client Model testing")
    #
    # logging.info("")
    # logging.info("Testing global model on individual client's test data")
    #
    # # testing global model on individual client's test data
    # # 在每个客户的测试数据上测试全局模型
    # s = 0
    # var = 0
    # for i in range(args.num_users):
    #     logging.info("Client {}".format(i))
    #     acc_train, loss_train = test_client(args, dataset_train, train_data_users[i], net_glob)
    #     acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], net_glob)
    #     logging.info("Training accuracy: {:.3f}".format(acc_train))
    #     logging.info("Testing accuracy: {:.3f}".format(acc_test))
    #     s += acc_test
    # s /= args.num_users  # 测试集的平均准确度
    # for i in range(args.num_users):
    #     acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], net_glob)
    #     var += (acc_test - s) ** 2
    # var /= args.num_users
    # logging.info("Average Client accuracy of global model on each client's test data: {: .3f}".format(s))
    # logging.info("Average Client accuracy Variance of global model on each client's test data: {: .3f}".format(var))
    #
    # dill.dump(stats,open(os.path.join(args.summary,'stats.pkl'),'wb'))
    # writer.close()
    # # print(stats['After Average'])
    # # print(stats['After finetune Average'])


######################################################FedAvg#######################################################

    #     print('Round {}'.format(iter))
    #
    #     logging.info("---------Round {}---------".format(iter))
    #
    #     w_locals, loss_locals = [], []
    #     w_select_locals, loss_select_locals = [], []
    #     select_Fk = []  # select_Fk是一个列表，保存被选择客户端的历史损失
    #     select_pk = []
    #     eps_user = [1] * args.num_users  # 保存每个用户的隐私预算
    #
    #     for idx in range(args.num_users):
    #         w_locals.append(local_nets[idx].state_dict())
    #
    #     m = max(int(args.frac * args.num_users), 1)
    #     idxs_users = np.random.choice(range(args.num_users), m, replace=False)
    #
    #     # 测试全局模型在所有用户的损失
    #     for i in range(args.num_users):
    #         acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], net_glob)
    #         loss_locals.append(copy.deepcopy(loss_test))
    #         loss_history[i][iter] = loss_test
    #
    #     # 计算全局模型在所有用户的平均损失
    #     avg_loss = 0
    #     for i in range(args.num_users):
    #         avg_loss += loss_history[i][iter]
    #     avg_loss /= args.num_users
    #     loss_history_global_avg.append(copy.deepcopy(avg_loss))
    #
    #     if iter < 2:
    #         delta_F[iter] = 0.5
    #     else:
    #         delta_loss = loss_history_global_avg[iter - 1] - loss_history_global_avg[iter - 2]
    #         if delta_loss >= args.delta and delta_loss <= delta_F[iter - 1]:
    #             delta_F[iter] = delta_loss
    #         elif delta_loss < 0 or delta_loss > delta_F[iter - 1]:
    #             delta_F[iter] = delta_F[iter - 1]
    #         else:
    #             delta_F[iter] = 0
    #     if iter < (args.epochs):
    #         eps_t = math.exp(-delta_F[iter]) * ((eps - eps_c) / (args.epochs - iter + 1))  # 本轮的衰减预算
    #         eps_c += eps_t
    #     else:
    #         eps_t = eps - eps_c
    #     # print("本轮的衰减预算:", eps_t)
    #
    #     # 隐私预算均匀递增
    #     eps_t = eps_s + iter * 0.002 * eps / 100
    #     print("本轮的预算:", eps_t)
    #
    #     # # 计算所有客户端的历史衰减损失user_loss_history
    #     # for i in range(args.num_users):
    #     #     F_i = 0
    #     #     F_m = 0
    #     #     for j in range(0, iter + 1):
    #     #         F_i += (args.rou ** (iter - j)) * loss_history[i][j]
    #     #         F_m += (args.rou ** (iter - j))
    #     #     user_loss_history[i][iter] = F_i / F_m
    #     #
    #     # for idx in idxs_users:
    #     #     select_Fk.append(user_loss_history[idx][iter])
    #     #     select_pk.append(pk[idx])
    #
    #     # # 计算每个用户本轮的隐私预算
    #     # F_max = max(select_Fk)
    #     # for idx in idxs_users:
    #     #     eps_user[idx] = eps_t * (select_Fk[idx] / F_max)
    #
    #     for idx in idxs_users:
    #         w, loss = train_client(args, dataset_train, train_data_users[idx], net=local_nets[idx], eps_user=eps_t)
    #         w_select_locals.append(w)
    #         loss_select_locals.append(copy.deepcopy(loss))
    #
    #     # store testing and training accuracies of the model before global aggregation 测试本地模型聚合前的训练集和测试集准确度
    #     logging.info("Testing Client Models before aggregation")
    #     logging.info("")
    #     s = 0
    #     for i in idxs_users:
    #         logging.info("Client {}:".format(i))
    #         acc_train, loss_train = test_client(args, dataset_train, train_data_users[i], local_nets[i])
    #         acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], local_nets[i])
    #         logging.info("Training accuracy: {:.3f}".format(acc_train))
    #         logging.info("Testing accuracy: {:.3f}".format(acc_test))
    #         logging.info("")
    #         # print(acc_test)
    #         stats[i][iter]['Before Training accuracy'] = acc_train
    #         stats[i][iter]['Before Test accuracy'] = acc_test
    #         writer.add_scalar(str(i) + '/Before Training accuracy', acc_train, iter)
    #         writer.add_scalar(str(i) + '/Before Test accuracy', acc_test, iter)
    #
    #         s += acc_test
    #     s /= (args.num_users * args.frac)
    #     logging.info("Average Client accuracy on their test data: {: .3f}".format(s))
    #     stats['Before Average'][iter] = s
    #     writer.add_scalar('Average' + '/Before Test accuracy', s, iter)
    #
    #     # update global weights
    #     w_glob = FedAvg(w_select_locals)
    #
    #     # copy weight to net_glob
    #     net_glob.load_state_dict(w_glob)  # 将全局模型参数w_glob加载到全局模型中，load_state_dict()是将状态字典加载到模型中
    #
    #     # Updating base layers of the clients and keeping the personalized layers same  # 更新本地模型
    #     for idx in range(args.num_users):
    #         # 遍历w_glob字典中的所有键，list(w_glob.keys())返回w_glob字典的所有键，然后for循环依次将这些键赋值给变量i，并执行循环中的代码
    #         for i in list(w_glob.keys()):
    #             w_locals[idx][i] = copy.deepcopy(w_glob[i])  # w_locals[idx][i]：表示第idx个本地模型的第i层
    #
    #         local_nets[idx].load_state_dict(w_locals[idx])  # 从字典获取模型参数加载到模型中
    #
    #     # store train and test accuracies after updating local models
    #     logging.info("Testing Client Models after aggregation")
    #     logging.info("")
    #     s = 0
    #     var = 0
    #     for i in range(args.num_users):
    #         logging.info("Client {}:".format(i))
    #         acc_train, loss_train = test_client(args, dataset_train, train_data_users[i], net_glob)
    #         acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], net_glob)
    #         logging.info("Training accuracy: {:.3f}".format(acc_train))
    #         logging.info("Testing accuracy: {:.3f}".format(acc_test))
    #         logging.info("")
    #
    #         stats[i][iter]['After Training accuracy'] = acc_train
    #         stats[i][iter]['After Test accuracy'] = acc_test
    #         writer.add_scalar(str(i) + '/After Training accuracy', acc_train, iter)
    #         writer.add_scalar(str(i) + '/After Test accuracy', acc_test, iter)
    #
    #         s += acc_test
    #     s /= args.num_users
    #     for i in range(args.num_users):
    #         acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], net_glob)
    #         var += (acc_test - s) ** 2
    #     var /= args.num_users
    #     logging.info("Average Client accuracy on their test data: {: .3f}".format(s))
    #     logging.info("Average Client accuracy Variance on their test data: {: .3f}".format(var))
    #
    #     stats['After Average'][iter] = s
    #     writer.add_scalar('Average' + '/After Test accuracy', s, iter)
    #
    #     loss_avg = sum(loss_select_locals) / len(loss_select_locals)
    #     logging.info('Average loss of clients: {:.3f}'.format(loss_avg))
    #
    # end = time.time()
    #
    # logging.info("Training Time: {}s".format(end-start))
    # logging.info("End of Training")
    #
    # # save model parameters
    # # 保存每个客户端的本地模型参数
    # torch.save(net_glob.state_dict(),'./state_dict/server_{}.pt'.format(file_name))
    # for i in range(args.num_users):
    #     torch.save(local_nets[i].state_dict(),'./state_dict/client_{}_{}.pt'.format(i,file_name))
    #
    # # test global model on training set and testing set
    # # 在训练集和测试集上测试全局模型
    #
    # logging.info("")
    # logging.info("Testing")
    #
    # logging.info("Global Server Model")
    # net_glob.eval()
    # acc_train, loss_train = test_img(net_glob, dataset_train, args)
    # acc_test, loss_test = test_img(net_glob, dataset_test, args)
    # logging.info("Training accuracy of Server: {:.3f}".format(acc_train))
    # logging.info("Training loss of Server: {:.3f}".format(loss_train))
    # logging.info("Testing accuracy of Server: {:.3f}".format(acc_test))
    # logging.info("Testing loss of Server: {:.3f}".format(loss_test))
    # logging.info("End of Server Model Testing")
    # logging.info("")
    #
    # logging.info("Client Models")
    # s = 0
    # # testing local models
    # # 测试本地模型
    # for i in range(args.num_users):
    #     logging.info("Client {}:".format(i))
    #     acc_train, loss_train = test_client(args,dataset_train,train_data_users[i],local_nets[i])
    #     acc_test, loss_test = test_client(args,dataset_train,test_data_users[i],local_nets[i])
    #     logging.info("Training accuracy: {:.3f}".format(acc_train))
    #     logging.info("Training loss: {:.3f}".format(loss_train))
    #     logging.info("Testing accuracy: {:.3f}".format(acc_test))
    #     logging.info("Testing loss: {:.3f}".format(loss_test))
    #     logging.info("")
    #     s += acc_test
    # s /= args.num_users
    # logging.info("Average Client accuracy on their test data: {: .3f}".format(s))
    # logging.info("End of Client Model testing")
    #
    #
    # logging.info("")
    # logging.info("Testing global model on individual client's test data")
    #
    # # testing global model on individual client's test data
    # # 在每个客户的测试数据上测试全局模型
    # s = 0
    # var = 0
    # for i in range(args.num_users):
    #     logging.info("Client {}".format(i))
    #     acc_train, loss_train = test_client(args, dataset_train, train_data_users[i], net_glob)
    #     acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], net_glob)
    #     logging.info("Training accuracy: {:.3f}".format(acc_train))
    #     logging.info("Testing accuracy: {:.3f}".format(acc_test))
    #     s += acc_test
    # s /= args.num_users  # 测试集的平均准确度
    # for i in range(args.num_users):
    #     acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], net_glob)
    #     var += (acc_test - s) ** 2
    # var /= args.num_users
    # logging.info("Average Client accuracy of global model on each client's test data: {: .3f}".format(s))
    # logging.info("Average Client accuracy Variance of global model on each client's test data: {: .3f}".format(var))
    #
    # dill.dump(stats,open(os.path.join(args.summary,'stats.pkl'),'wb'))
    # writer.close()
    # print(stats['After Average'])
    # print(stats['After finetune Average'])


######################################################FedProx#######################################################

    #     print('Round {}'.format(iter))
    #
    #     logging.info("---------Round {}---------".format(iter))
    #
    #     w_locals, loss_locals = [], []
    #     w_select_locals, loss_select_locals = [], []
    #
    #     for idx in range(args.num_users):
    #         w_locals.append(local_nets[idx].state_dict())
    #
    #     m = max(int(args.frac * args.num_users), 1)
    #     idxs_users = np.random.choice(range(args.num_users), m, replace=False)
    #
    #     for idx in idxs_users:
    #         w, loss = train_client(args, dataset_train, train_data_users[idx], net=local_nets[idx], netg=net_glob)  # w是保存本地模型的字典
    #         loss_select_locals.append(copy.deepcopy(loss))
    #         w_select_locals.append(w)
    #
    #     # store testing and training accuracies of the model before global aggregation 测试本地模型聚合前的训练集和测试集准确度
    #     logging.info("Testing Client Models before aggregation")
    #     logging.info("")
    #     s = 0
    #     for i in idxs_users:
    #         logging.info("Client {}:".format(i))
    #         acc_train, loss_train = test_client(args, dataset_train, train_data_users[i], local_nets[i])
    #         acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], local_nets[i])
    #         logging.info("Training accuracy: {:.3f}".format(acc_train))
    #         logging.info("Testing accuracy: {:.3f}".format(acc_test))
    #         logging.info("")
    #         # print(acc_test)
    #         stats[i][iter]['Before Training accuracy'] = acc_train
    #         stats[i][iter]['Before Test accuracy'] = acc_test
    #         writer.add_scalar(str(i) + '/Before Training accuracy', acc_train, iter)
    #         writer.add_scalar(str(i) + '/Before Test accuracy', acc_test, iter)
    #
    #         s += acc_test
    #     s /= (args.num_users * args.frac)
    #     logging.info("Average Client accuracy on their test data: {: .3f}".format(s))
    #     stats['Before Average'][iter] = s
    #     writer.add_scalar('Average' + '/Before Test accuracy', s, iter)
    #
    #     # update global weights
    #     w_glob = FedAvg(w_select_locals)
    #
    #     # copy weight to net_glob
    #     net_glob.load_state_dict(w_glob)  # 将全局模型参数w_glob加载到全局模型中，load_state_dict()是将状态字典加载到模型中
    #
    #     # Updating base layers of the clients and keeping the personalized layers same  # 更新本地模型
    #     for idx in range(args.num_users):
    #         # 遍历w_glob字典中的所有键，list(w_glob.keys())返回w_glob字典的所有键，然后for循环依次将这些键赋值给变量i，并执行循环中的代码
    #         for i in list(w_glob.keys()):
    #             w_locals[idx][i] = copy.deepcopy(w_glob[i])  # w_locals[idx][i]：表示第idx个本地模型的第i层
    #
    #         local_nets[idx].load_state_dict(w_locals[idx])  # 从字典获取模型参数加载到模型中
    #
    #     # store train and test accuracies after updating local models
    #     logging.info("Testing Client Models after aggregation")
    #     logging.info("")
    #     s = 0
    #     var = 0
    #     for i in range(args.num_users):
    #         logging.info("Client {}:".format(i))
    #         acc_train, loss_train = test_client(args, dataset_train, train_data_users[i], net_glob)
    #         acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], net_glob)
    #         logging.info("Training accuracy: {:.3f}".format(acc_train))
    #         logging.info("Testing accuracy: {:.3f}".format(acc_test))
    #         logging.info("")
    #
    #         stats[i][iter]['After Training accuracy'] = acc_train
    #         stats[i][iter]['After Test accuracy'] = acc_test
    #         writer.add_scalar(str(i) + '/After Training accuracy', acc_train, iter)
    #         writer.add_scalar(str(i) + '/After Test accuracy', acc_test, iter)
    #
    #         s += acc_test
    #     s /= args.num_users
    #     for i in range(args.num_users):
    #         acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], net_glob)
    #         var += (acc_test - s) ** 2
    #     var /= args.num_users
    #     logging.info("Average Client accuracy on their test data: {: .3f}".format(s))
    #     logging.info("Average Client accuracy Variance on their test data: {: .3f}".format(var))
    #
    #     stats['After Average'][iter] = s
    #     writer.add_scalar('Average' + '/After Test accuracy', s, iter)
    #
    #     loss_avg = sum(loss_select_locals) / len(loss_select_locals)
    #     logging.info('Average loss of clients: {:.3f}'.format(loss_avg))
    #
    # end = time.time()
    #
    # logging.info("Training Time: {}s".format(end-start))
    # logging.info("End of Training")
    #
    # # save model parameters
    # # 保存每个客户端的本地模型参数
    # torch.save(net_glob.state_dict(),'./state_dict/server_{}.pt'.format(file_name))
    # for i in range(args.num_users):
    #     torch.save(local_nets[i].state_dict(),'./state_dict/client_{}_{}.pt'.format(i,file_name))
    #
    # # test global model on training set and testing set
    # # 在训练集和测试集上测试全局模型
    #
    # logging.info("")
    # logging.info("Testing")
    #
    # logging.info("Global Server Model")
    # net_glob.eval()
    # acc_train, loss_train = test_img(net_glob, dataset_train, args)
    # acc_test, loss_test = test_img(net_glob, dataset_test, args)
    # logging.info("Training accuracy of Server: {:.3f}".format(acc_train))
    # logging.info("Training loss of Server: {:.3f}".format(loss_train))
    # logging.info("Testing accuracy of Server: {:.3f}".format(acc_test))
    # logging.info("Testing loss of Server: {:.3f}".format(loss_test))
    # logging.info("End of Server Model Testing")
    # logging.info("")
    #
    # logging.info("Client Models")
    # s = 0
    # # testing local models
    # # 测试本地模型
    # for i in range(args.num_users):
    #     logging.info("Client {}:".format(i))
    #     acc_train, loss_train = test_client(args,dataset_train,train_data_users[i],local_nets[i])
    #     acc_test, loss_test = test_client(args,dataset_train,test_data_users[i],local_nets[i])
    #     logging.info("Training accuracy: {:.3f}".format(acc_train))
    #     logging.info("Training loss: {:.3f}".format(loss_train))
    #     logging.info("Testing accuracy: {:.3f}".format(acc_test))
    #     logging.info("Testing loss: {:.3f}".format(loss_test))
    #     logging.info("")
    #     s += acc_test
    # s /= args.num_users
    # logging.info("Average Client accuracy on their test data: {: .3f}".format(s))
    # logging.info("End of Client Model testing")
    #
    #
    # logging.info("")
    # logging.info("Testing global model on individual client's test data")
    #
    # # testing global model on individual client's test data
    # # 在每个客户的测试数据上测试全局模型
    # s = 0
    # var = 0
    # for i in range(args.num_users):
    #     logging.info("Client {}".format(i))
    #     acc_train, loss_train = test_client(args, dataset_train, train_data_users[i], net_glob)
    #     acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], net_glob)
    #     logging.info("Training accuracy: {:.3f}".format(acc_train))
    #     logging.info("Testing accuracy: {:.3f}".format(acc_test))
    #     s += acc_test
    # s /= args.num_users  # 测试集的平均准确度
    # for i in range(args.num_users):
    #     acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], net_glob)
    #     var += (acc_test - s) ** 2
    # var /= args.num_users
    # logging.info("Average Client accuracy of global model on each client's test data: {: .3f}".format(s))
    # logging.info("Average Client accuracy Variance of global model on each client's test data: {: .3f}".format(var))
    #
    # dill.dump(stats,open(os.path.join(args.summary,'stats.pkl'),'wb'))
    # writer.close()
    # print(stats['After Average'])
    # print(stats['After finetune Average'])

    ######################################################DR-FedAvg#######################################################

    #     print('Round {}'.format(iter))
    #
    #     logging.info("---------Round {}---------".format(iter))
    #
    #     w_locals, loss_locals = [], []  # w_locals保存客户端本地模型参数，loss_locals保存本地模型的loss
    #     Fk = []  # Fk是一个列表，保存每个客户端的本地损失
    #
    #     for idx in range(0, args.num_users):
    #         w, loss = train_client(args, dataset_train, train_data_users[idx], net=local_nets[idx])  # w是保存本地模型的字典
    #         w_locals.append(w)
    #         loss_locals.append(copy.deepcopy(loss))
    #         acc_test, loss_test = test_client(args, dataset_train, test_data_users[idx], net_glob)
    #         Fk.append(loss_test)
    #
    #     # store testing and training accuracies of the model before global aggregation 测试本地模型聚合前的训练集和测试集准确度
    #     logging.info("Testing Client Models before aggregation")
    #     logging.info("")
    #     s = 0
    #     for i in range(args.num_users):
    #         logging.info("Client {}:".format(i))
    #         acc_train, loss_train = test_client(args, dataset_train, train_data_users[i], local_nets[i])
    #         acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], local_nets[i])
    #         logging.info("Training accuracy: {:.3f}".format(acc_train))
    #         logging.info("Testing accuracy: {:.3f}".format(acc_test))
    #         logging.info("")
    #         # print(acc_test)
    #         stats[i][iter]['Before Training accuracy'] = acc_train
    #         stats[i][iter]['Before Test accuracy'] = acc_test
    #         writer.add_scalar(str(i) + '/Before Training accuracy', acc_train, iter)
    #         writer.add_scalar(str(i) + '/Before Test accuracy', acc_test, iter)
    #
    #         s += acc_test
    #     s /= args.num_users
    #     logging.info("Average Client accuracy on their test data: {: .3f}".format(s))
    #     stats['Before Average'][iter] = s
    #     writer.add_scalar('Average' + '/Before Test accuracy', s, iter)
    #
    #     # update global weights
    #     w_glob = DR_FedAvg(args, w_locals, pk, Fk)
    #
    #     # copy weight to net_glob
    #     net_glob.load_state_dict(w_glob)  # 将全局模型参数w_glob加载到全局模型中，load_state_dict()是将状态字典加载到模型中
    #
    #     # Updating base layers of the clients and keeping the personalized layers same  # 更新本地模型
    #     for idx in range(args.num_users):
    #         # 遍历w_glob字典中的所有键，list(w_glob.keys())返回w_glob字典的所有键，然后for循环依次将这些键赋值给变量i，并执行循环中的代码
    #         for i in list(w_glob.keys()):
    #             w_locals[idx][i] = copy.deepcopy(w_glob[i])  # w_locals[idx][i]：表示第idx个本地模型的第i层
    #
    #         local_nets[idx].load_state_dict(w_locals[idx])  # 从字典获取模型参数加载到模型中
    #
    #     # store train and test accuracies after updating local models 测试本地模型聚合后的训练集和测试集准确度
    #     logging.info("Testing Client Models after aggregation")
    #     logging.info("")
    #     s = 0
    #     var = 0
    #     for i in range(args.num_users):
    #         logging.info("Client {}:".format(i))
    #         acc_train, loss_train = test_client(args, dataset_train, train_data_users[i], local_nets[i])
    #         acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], local_nets[i])
    #         logging.info("Training accuracy: {:.3f}".format(acc_train))
    #         logging.info("Testing accuracy: {:.3f}".format(acc_test))
    #         logging.info("")
    #
    #         stats[i][iter]['After Training accuracy'] = acc_train
    #         stats[i][iter]['After Test accuracy'] = acc_test
    #         writer.add_scalar(str(i) + '/After Training accuracy', acc_train, iter)
    #         writer.add_scalar(str(i) + '/After Test accuracy', acc_test, iter)
    #
    #         s += acc_test
    #     s /= args.num_users
    #     for i in range(args.num_users):
    #         acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], local_nets[i])
    #         var += (acc_test - s) ** 2
    #     var /= args.num_users
    #     logging.info("Average Client accuracy on their test data: {: .3f}".format(s))
    #     logging.info("Average Client accuracy Variance on their test data: {: .3f}".format(var))
    #
    #     stats['After Average'][iter] = s
    #     writer.add_scalar('Average' + '/After Test accuracy', s, iter)
    #
    #     # loss_avg = sum(loss_locals) / len(loss_locals)
    #     # logging.info('Average loss of clients: {:.3f}'.format(loss_avg))
    #
    # end = time.time()
    #
    # logging.info("Training Time: {}s".format(end - start))
    # logging.info("End of Training")
    #
    # # save model parameters
    # # 保存每个客户端的本地模型参数
    # torch.save(net_glob.state_dict(), './state_dict/server_{}.pt'.format(file_name))
    # for i in range(args.num_users):
    #     torch.save(local_nets[i].state_dict(), './state_dict/client_{}_{}.pt'.format(i, file_name))
    #
    # # test global model on training set and testing set
    # # 在训练集和测试集上测试全局模型
    #
    # logging.info("")
    # logging.info("Testing")
    #
    # logging.info("Global Server Model")
    # net_glob.eval()
    # acc_train, loss_train = test_img(net_glob, dataset_train, args)
    # acc_test, loss_test = test_img(net_glob, dataset_test, args)
    # logging.info("Training accuracy of Server: {:.3f}".format(acc_train))
    # logging.info("Training loss of Server: {:.3f}".format(loss_train))
    # logging.info("Testing accuracy of Server: {:.3f}".format(acc_test))
    # logging.info("Testing loss of Server: {:.3f}".format(loss_test))
    # logging.info("End of Server Model Testing")
    # logging.info("")
    #
    # logging.info("Client Models")
    # s = 0
    # # testing local models
    # # 测试本地模型
    # for i in range(args.num_users):
    #     logging.info("Client {}:".format(i))
    #     acc_train, loss_train = test_client(args, dataset_train, train_data_users[i], local_nets[i])
    #     acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], local_nets[i])
    #     logging.info("Training accuracy: {:.3f}".format(acc_train))
    #     logging.info("Training loss: {:.3f}".format(loss_train))
    #     logging.info("Testing accuracy: {:.3f}".format(acc_test))
    #     logging.info("Testing loss: {:.3f}".format(loss_test))
    #     logging.info("")
    #     s += acc_test
    # s /= args.num_users
    # logging.info("Average Client accuracy on their test data: {: .3f}".format(s))
    # logging.info("End of Client Model testing")
    #
    # logging.info("")
    # logging.info("Testing global model on individual client's test data")
    #
    # # testing global model on individual client's test data
    # # 在每个客户的测试数据上测试全局模型
    # s = 0
    # var = 0
    # for i in range(args.num_users):
    #     logging.info("Client {}".format(i))
    #     acc_train, loss_train = test_client(args, dataset_train, train_data_users[i], net_glob)
    #     acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], net_glob)
    #     logging.info("Training accuracy: {:.3f}".format(acc_train))
    #     logging.info("Testing accuracy: {:.3f}".format(acc_test))
    #     s += acc_test
    # s /= args.num_users  # 测试集的平均准确度
    # for i in range(args.num_users):
    #     acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], net_glob)
    #     var += (acc_test - s) ** 2
    # var /= args.num_users
    # logging.info("Average Client accuracy of global model on each client's test data: {: .3f}".format(s))
    # logging.info("Average Client accuracy Variance of global model on each client's test data: {: .3f}".format(var))
    #
    # dill.dump(stats, open(os.path.join(args.summary, 'stats.pkl'), 'wb'))
    # writer.close()
    # print(stats['After Average'])
    # print(stats['After finetune Average'])


##########################################################q-FedAvg#######################################################

    #     print('Round {}'.format(iter))
    #
    #     logging.info("---------Round {}---------".format(iter))
    #
    #     w_locals, loss_locals = [], []  # w_locals保存客户端本地模型参数，loss_locals保存本地模型的loss
    #     Fk = []  # Fk是一个列表，保存每个客户端的本地损失
    #
    #     for idx in range(0, args.num_users):
    #         w, loss = train_client(args, dataset_train, train_data_users[idx], net=local_nets[idx])  # w是保存本地模型的字典
    #         w_locals.append(w)
    #         loss_locals.append(copy.deepcopy(loss))
    #         acc_test, loss_test = test_client(args, dataset_train, test_data_users[idx], net_glob)
    #         Fk.append(loss_test)
    #
    #     # store testing and training accuracies of the model before global aggregation 测试本地模型聚合前的训练集和测试集准确度
    #     logging.info("Testing Client Models before aggregation")
    #     logging.info("")
    #     s = 0
    #     for i in range(args.num_users):
    #         logging.info("Client {}:".format(i))
    #         acc_train, loss_train = test_client(args, dataset_train, train_data_users[i], local_nets[i])
    #         acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], local_nets[i])
    #         logging.info("Training accuracy: {:.3f}".format(acc_train))
    #         logging.info("Testing accuracy: {:.3f}".format(acc_test))
    #         logging.info("")
    #         # print(acc_test)
    #         stats[i][iter]['Before Training accuracy'] = acc_train
    #         stats[i][iter]['Before Test accuracy'] = acc_test
    #         writer.add_scalar(str(i) + '/Before Training accuracy', acc_train, iter)
    #         writer.add_scalar(str(i) + '/Before Test accuracy', acc_test, iter)
    #
    #         s += acc_test
    #     s /= args.num_users
    #     logging.info("Average Client accuracy on their test data: {: .3f}".format(s))
    #     stats['Before Average'][iter] = s
    #     writer.add_scalar('Average' + '/Before Test accuracy', s, iter)
    #
    #     # update global weights
    #     w_glob,e = q_FedAvg(args, w_locals, pk, Fk)
    #     print("e: ", e)
    #
    #     for idx in range(args.num_users):
    #         user_weight_history[idx][iter] = e[idx]
    #
    #     # copy weight to net_glob
    #     net_glob.load_state_dict(w_glob)  # 将全局模型参数w_glob加载到全局模型中，load_state_dict()是将状态字典加载到模型中
    #
    #     # Updating base layers of the clients and keeping the personalized layers same  # 更新本地模型
    #     for idx in range(args.num_users):
    #         # 遍历w_glob字典中的所有键，list(w_glob.keys())返回w_glob字典的所有键，然后for循环依次将这些键赋值给变量i，并执行循环中的代码
    #         for i in list(w_glob.keys()):
    #             w_locals[idx][i] = copy.deepcopy(w_glob[i])  # w_locals[idx][i]：表示第idx个本地模型的第i层
    #
    #         local_nets[idx].load_state_dict(w_locals[idx])  # 从字典获取模型参数加载到模型中
    #
    #     # store train and test accuracies after updating local models 测试本地模型聚合后的训练集和测试集准确度
    #     logging.info("Testing Client Models after aggregation")
    #     logging.info("")
    #     s = 0
    #     var = 0
    #     for i in range(args.num_users):
    #         logging.info("Client {}:".format(i))
    #         acc_train, loss_train = test_client(args, dataset_train, train_data_users[i], local_nets[i])
    #         acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], local_nets[i])
    #         logging.info("Training accuracy: {:.3f}".format(acc_train))
    #         logging.info("Testing accuracy: {:.3f}".format(acc_test))
    #         logging.info("")
    #
    #         stats[i][iter]['After Training accuracy'] = acc_train
    #         stats[i][iter]['After Test accuracy'] = acc_test
    #         writer.add_scalar(str(i) + '/After Training accuracy', acc_train, iter)
    #         writer.add_scalar(str(i) + '/After Test accuracy', acc_test, iter)
    #
    #         s += acc_test
    #     s /= args.num_users
    #     for i in range(args.num_users):
    #         acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], local_nets[i])
    #         var += (acc_test - s) ** 2
    #     var /= args.num_users
    #     logging.info("Average Client accuracy on their test data: {: .3f}".format(s))
    #     logging.info("Average Client accuracy Variance on their test data: {: .3f}".format(var))
    #
    #     stats['After Average'][iter] = s
    #     writer.add_scalar('Average' + '/After Test accuracy', s, iter)
    #
    #     # loss_avg = sum(loss_locals) / len(loss_locals)
    #     # logging.info('Average loss of clients: {:.3f}'.format(loss_avg))
    #
    # end = time.time()
    #
    # logging.info("Training Time: {}s".format(end - start))
    # logging.info("End of Training")
    #
    # # save model parameters
    # # 保存每个客户端的本地模型参数
    # torch.save(net_glob.state_dict(), './state_dict/server_{}.pt'.format(file_name))
    # for i in range(args.num_users):
    #     torch.save(local_nets[i].state_dict(), './state_dict/client_{}_{}.pt'.format(i, file_name))
    #
    # # test global model on training set and testing set
    # # 在训练集和测试集上测试全局模型
    #
    # logging.info("")
    # logging.info("Testing")
    #
    # logging.info("Global Server Model")
    # net_glob.eval()
    # acc_train, loss_train = test_img(net_glob, dataset_train, args)
    # acc_test, loss_test = test_img(net_glob, dataset_test, args)
    # logging.info("Training accuracy of Server: {:.3f}".format(acc_train))
    # logging.info("Training loss of Server: {:.3f}".format(loss_train))
    # logging.info("Testing accuracy of Server: {:.3f}".format(acc_test))
    # logging.info("Testing loss of Server: {:.3f}".format(loss_test))
    # logging.info("End of Server Model Testing")
    # logging.info("")
    #
    # logging.info("Client Models")
    # s = 0
    # # testing local models
    # # 测试本地模型
    # for i in range(args.num_users):
    #     logging.info("Client {}:".format(i))
    #     acc_train, loss_train = test_client(args, dataset_train, train_data_users[i], local_nets[i])
    #     acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], local_nets[i])
    #     logging.info("Training accuracy: {:.3f}".format(acc_train))
    #     logging.info("Training loss: {:.3f}".format(loss_train))
    #     logging.info("Testing accuracy: {:.3f}".format(acc_test))
    #     logging.info("Testing loss: {:.3f}".format(loss_test))
    #     logging.info("")
    #     s += acc_test
    # s /= args.num_users
    # logging.info("Average Client accuracy on their test data: {: .3f}".format(s))
    # logging.info("End of Client Model testing")
    #
    # logging.info("")
    # logging.info("Testing global model on individual client's test data")
    #
    # # testing global model on individual client's test data
    # # 在每个客户的测试数据上测试全局模型
    # s = 0
    # var = 0
    # for i in range(args.num_users):
    #     logging.info("Client {}".format(i))
    #     acc_train, loss_train = test_client(args, dataset_train, train_data_users[i], net_glob)
    #     acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], net_glob)
    #     logging.info("Training accuracy: {:.3f}".format(acc_train))
    #     logging.info("Testing accuracy: {:.3f}".format(acc_test))
    #     s += acc_test
    # s /= args.num_users  # 测试集的平均准确度
    # for i in range(args.num_users):
    #     acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], net_glob)
    #     var += (acc_test - s) ** 2
    # var /= args.num_users
    # logging.info("Average Client accuracy of global model on each client's test data: {: .3f}".format(s))
    # logging.info("Average Client accuracy Variance of global model on each client's test data: {: .3f}".format(var))
    #
    # # 保存客户端历史权重
    # logging.info("Historical weight: %s", user_weight_history)
    #
    # dill.dump(stats, open(os.path.join(args.summary, 'stats.pkl'), 'wb'))
    # writer.close()
    # print(stats['After Average'])
    # print(stats['After finetune Average'])



##########################################################AFL-FedAvg#######################################################

    #     print('Round {}'.format(iter))
    #
    #     logging.info("---------Round {}---------".format(iter))
    #
    #     w_locals, loss_locals = [], []  # w_locals保存客户端本地模型参数，loss_locals保存本地模型的loss
    #     Fk = []  # Fk是一个列表，保存每个客户端的本地损失
    #
    #     for idx in range(0, args.num_users):
    #         w, loss = train_client(args, dataset_train, train_data_users[idx], net=local_nets[idx])  # w是保存本地模型的字典
    #         w_locals.append(w)
    #         loss_locals.append(copy.deepcopy(loss))
    #         acc_test, loss_test = test_client(args, dataset_train, test_data_users[idx], net_glob)
    #         Fk.append(loss_test)
    #
    #     # store testing and training accuracies of the model before global aggregation 测试本地模型聚合前的训练集和测试集准确度
    #     logging.info("Testing Client Models before aggregation")
    #     logging.info("")
    #     s = 0
    #     for i in range(args.num_users):
    #         logging.info("Client {}:".format(i))
    #         acc_train, loss_train = test_client(args, dataset_train, train_data_users[i], local_nets[i])
    #         acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], local_nets[i])
    #         logging.info("Training accuracy: {:.3f}".format(acc_train))
    #         logging.info("Testing accuracy: {:.3f}".format(acc_test))
    #         logging.info("")
    #         # print(acc_test)
    #         stats[i][iter]['Before Training accuracy'] = acc_train
    #         stats[i][iter]['Before Test accuracy'] = acc_test
    #         writer.add_scalar(str(i) + '/Before Training accuracy', acc_train, iter)
    #         writer.add_scalar(str(i) + '/Before Test accuracy', acc_test, iter)
    #
    #         s += acc_test
    #     s /= args.num_users
    #     logging.info("Average Client accuracy on their test data: {: .3f}".format(s))
    #     stats['Before Average'][iter] = s
    #     writer.add_scalar('Average' + '/Before Test accuracy', s, iter)
    #
    #
    #     for idx in range(len(latest_lambdas)):
    #         latest_lambdas[idx] += args.lr * loss_locals[idx]
    #
    #     latest_lambdas = project(latest_lambdas)
    #     print("latest_lambdas: ",latest_lambdas)
    #
    #     # update global weights
    #     w_glob = AFL(args, w_locals, latest_lambdas)
    #
    #     for idx in range(args.num_users):
    #         user_weight_history[idx][iter] = latest_lambdas[idx]
    #
    #     # copy weight to net_glob
    #     net_glob.load_state_dict(w_glob)  # 将全局模型参数w_glob加载到全局模型中，load_state_dict()是将状态字典加载到模型中
    #
    #     # Updating base layers of the clients and keeping the personalized layers same  # 更新本地模型
    #     for idx in range(args.num_users):
    #         # 遍历w_glob字典中的所有键，list(w_glob.keys())返回w_glob字典的所有键，然后for循环依次将这些键赋值给变量i，并执行循环中的代码
    #         for i in list(w_glob.keys()):
    #             w_locals[idx][i] = copy.deepcopy(w_glob[i])  # w_locals[idx][i]：表示第idx个本地模型的第i层
    #
    #         local_nets[idx].load_state_dict(w_locals[idx])  # 从字典获取模型参数加载到模型中
    #
    #     # store train and test accuracies after updating local models 测试本地模型聚合后的训练集和测试集准确度
    #     logging.info("Testing Client Models after aggregation")
    #     logging.info("")
    #     s = 0
    #     var = 0
    #     for i in range(args.num_users):
    #         logging.info("Client {}:".format(i))
    #         acc_train, loss_train = test_client(args, dataset_train, train_data_users[i], local_nets[i])
    #         acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], local_nets[i])
    #         logging.info("Training accuracy: {:.3f}".format(acc_train))
    #         logging.info("Testing accuracy: {:.3f}".format(acc_test))
    #         logging.info("")
    #
    #         stats[i][iter]['After Training accuracy'] = acc_train
    #         stats[i][iter]['After Test accuracy'] = acc_test
    #         writer.add_scalar(str(i) + '/After Training accuracy', acc_train, iter)
    #         writer.add_scalar(str(i) + '/After Test accuracy', acc_test, iter)
    #
    #         s += acc_test
    #     s /= args.num_users
    #     for i in range(args.num_users):
    #         acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], local_nets[i])
    #         var += (acc_test - s) ** 2
    #     var /= args.num_users
    #     logging.info("Average Client accuracy on their test data: {: .3f}".format(s))
    #     logging.info("Average Client accuracy Variance on their test data: {: .3f}".format(var))
    #
    #     stats['After Average'][iter] = s
    #     writer.add_scalar('Average' + '/After Test accuracy', s, iter)
    #
    #     # loss_avg = sum(loss_locals) / len(loss_locals)
    #     # logging.info('Average loss of clients: {:.3f}'.format(loss_avg))
    #
    # end = time.time()
    #
    # logging.info("Training Time: {}s".format(end - start))
    # logging.info("End of Training")
    #
    # # save model parameters
    # # 保存每个客户端的本地模型参数
    # torch.save(net_glob.state_dict(), './state_dict/server_{}.pt'.format(file_name))
    # for i in range(args.num_users):
    #     torch.save(local_nets[i].state_dict(), './state_dict/client_{}_{}.pt'.format(i, file_name))
    #
    # # test global model on training set and testing set
    # # 在训练集和测试集上测试全局模型
    #
    # logging.info("")
    # logging.info("Testing")
    #
    # logging.info("Global Server Model")
    # net_glob.eval()
    # acc_train, loss_train = test_img(net_glob, dataset_train, args)
    # acc_test, loss_test = test_img(net_glob, dataset_test, args)
    # logging.info("Training accuracy of Server: {:.3f}".format(acc_train))
    # logging.info("Training loss of Server: {:.3f}".format(loss_train))
    # logging.info("Testing accuracy of Server: {:.3f}".format(acc_test))
    # logging.info("Testing loss of Server: {:.3f}".format(loss_test))
    # logging.info("End of Server Model Testing")
    # logging.info("")
    #
    # logging.info("Client Models")
    # s = 0
    # # testing local models
    # # 测试本地模型
    # for i in range(args.num_users):
    #     logging.info("Client {}:".format(i))
    #     acc_train, loss_train = test_client(args, dataset_train, train_data_users[i], local_nets[i])
    #     acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], local_nets[i])
    #     logging.info("Training accuracy: {:.3f}".format(acc_train))
    #     logging.info("Training loss: {:.3f}".format(loss_train))
    #     logging.info("Testing accuracy: {:.3f}".format(acc_test))
    #     logging.info("Testing loss: {:.3f}".format(loss_test))
    #     logging.info("")
    #     s += acc_test
    # s /= args.num_users
    # logging.info("Average Client accuracy on their test data: {: .3f}".format(s))
    # logging.info("End of Client Model testing")
    #
    # logging.info("")
    # logging.info("Testing global model on individual client's test data")
    #
    # # testing global model on individual client's test data
    # # 在每个客户的测试数据上测试全局模型
    # s = 0
    # var = 0
    # for i in range(args.num_users):
    #     logging.info("Client {}".format(i))
    #     acc_train, loss_train = test_client(args, dataset_train, train_data_users[i], net_glob)
    #     acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], net_glob)
    #     logging.info("Training accuracy: {:.3f}".format(acc_train))
    #     logging.info("Testing accuracy: {:.3f}".format(acc_test))
    #     s += acc_test
    # s /= args.num_users  # 测试集的平均准确度
    # for i in range(args.num_users):
    #     acc_test, loss_test = test_client(args, dataset_train, test_data_users[i], net_glob)
    #     var += (acc_test - s) ** 2
    # var /= args.num_users
    # logging.info("Average Client accuracy of global model on each client's test data: {: .3f}".format(s))
    # logging.info("Average Client accuracy Variance of global model on each client's test data: {: .3f}".format(var))
    #
    # # 保存客户端历史权重
    # logging.info("Historical weight: %s", user_weight_history)
    #
    # dill.dump(stats, open(os.path.join(args.summary, 'stats.pkl'), 'wb'))
    # writer.close()
    # print(stats['After Average'])
    # print(stats['After finetune Average'])