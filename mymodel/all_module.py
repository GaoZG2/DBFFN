import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report,cohen_kappa_score
import random
import copy
import time
from datetime import datetime

from distance_suppress_module import DistanceSuppressionModule
from spectral_attention import SpectralAttention
from spatial_conv import SpatialConv
from classifier import Classifier
import config


class MyModel(nn.Module):
    def __init__(self, processed_data_depth, spatial_conv_kernel_size, 
                 width, height, class_num, classifier_depth, device, batch_size):
        """
        参数说明:
        processed_data_depth: 经过处理后的数据维度
        kernel_size: 卷积核大小，用于相关卷积操作的参数设置
        width: 数据在宽度维度上的大小
        height: 数据在高度维度上的大小
        class_num: 分类任务中的类别数量，用于分类器部分的设置
        """
        super(MyModel, self).__init__()
        self.distance_suppress_module = DistanceSuppressionModule(batch_size, width, height, processed_data_depth, device)
        self.spectral_attention = SpectralAttention(width, height, processed_data_depth)
        self.spatial_conv = SpatialConv(spatial_conv_kernel_size, width, height, processed_data_depth)
        self.classifier = Classifier(class_num, width, height, classifier_depth)

    def forward(self, x):
        x = self.spectral_attention(x)
        x = self.distance_suppress_module(x)
        x = self.spatial_conv(x)
        output = self.classifier(x)
        return output

""" Training dataset"""
class TrainDS(torch.utils.data.Dataset): 
    def __init__(self, Xtrain, ytrain):
        self.len = Xtrain.shape[0]
        self.x_data = torch.FloatTensor(Xtrain)
        self.y_data = torch.LongTensor(ytrain)        
    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]
    def __len__(self): 
        # 返回文件数据的数目
        return self.len

""" Validating dataset"""
class ValidateDS(torch.utils.data.Dataset): 
    def __init__(self, Xvali, yvali):
        self.len = Xvali.shape[0]
        self.x_data = torch.FloatTensor(Xvali)
        self.y_data = torch.LongTensor(yvali)        
    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]
    def __len__(self): 
        # 返回文件数据的数目
        return self.len

""" Testing dataset"""
class TestDS(torch.utils.data.Dataset): 
    def __init__(self, Xtest, ytest):
        self.len = Xtest.shape[0]
        self.x_data = torch.FloatTensor(Xtest)
        self.y_data = torch.LongTensor(ytest)
    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]
    def __len__(self): 
        # 返回文件数据的数目
        return self.len
def model_train(selected_data, processed_data_depth, class_num, validateNtest_ratio, test_ratio):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.gpu_used == False:
        device = torch.device("cpu")
    print(device)
    print('dataset: ', selected_data)

    # 创建 trainloader 和 testloader
    Xtrain_path = config.all_data_dir + '/' + selected_data + '/' + 'Xtrain.npy'
    ytrain_path = config.all_data_dir + '/' + selected_data + '/' + 'ytrain.npy'
    Xvalidate_path = config.all_data_dir + '/' + selected_data + '/' + 'Xvalidate.npy'
    yvalidate_path = config.all_data_dir + '/' + selected_data + '/' + 'yvalidate.npy'
    Xtest_path = config.all_data_dir + '/' + selected_data + '/' + 'Xtest.npy'
    ytest_path = config.all_data_dir + '/' + selected_data + '/' + 'ytest.npy'
    Xtrain_path = Xtrain_path
    ytrain_path = ytrain_path
    Xvalidate_path = Xvalidate_path
    yvalidate_path = yvalidate_path
    Xtest_path = Xtest_path
    ytest_path = ytest_path
    Xtrain = np.load(Xtrain_path)
    ytrain = np.load(ytrain_path)
    Xvalidate = np.load(Xvalidate_path)
    yvalidate = np.load(yvalidate_path)
    Xtest = np.load(Xtest_path)
    ytest = np.load(ytest_path)
    trainset = TrainDS(Xtrain, ytrain)
    validateset = ValidateDS(Xvalidate, yvalidate)
    testset  = TestDS(Xtest, ytest)
    train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=config.batch_size, shuffle=True)
    validate_loader = torch.utils.data.DataLoader(dataset=validateset, batch_size=config.batch_size, shuffle=False)
    test_loader  = torch.utils.data.DataLoader(dataset=testset,  batch_size=config.batch_size, shuffle=False)

    train_losses_all = []
    val_losses_all = []
    train_accuracy_all  = []
    val_accuracy_all  = []

    models_trained = []
    for submodel_idx in range(config.submodel_num):
        best_accuracy = 0
        best_net_num_epochs = 0
        best_net_last_epoch_loss = 1
        best_net = None
        accuracies = []  # 用于记录每次训练后模型在测试集上的准确率
        kernel_spatial = config.kernel_spatial_list[submodel_idx]
        for i in range(config.net_num):  # 训练 n 次不同初始化的net
            # 网络放到GPU上，每次重新初始化net
            time_begin_cur_net = time.time()
            begin_time_formatted = datetime.fromtimestamp(time_begin_cur_net).strftime('%Y-%m-%d %H:%M:%S')
            print(f'训练开始时间: {begin_time_formatted}')
            net = MyModel(
                processed_data_depth=processed_data_depth,
                spatial_conv_kernel_size=kernel_spatial,
                width=config.width,
                height=config.height,
                class_num=class_num,
                classifier_depth=32,
                device=device,
                batch_size=config.batch_size
            ).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(net.parameters(), lr=config.learning_rate)
            print(f'submodel {submodel_idx + 1} info: kernel_spatial={kernel_spatial}')

            # 开始训练
            random_number = random.randint(config.epoch_num_min, config.epoch_num_max)
            print(f"The epoch number of training {i + 1} is {random_number}")
            last_epoch_loss = 0
            train_losses = []
            val_losses = []
            train_accuracy  = []
            val_accuracy  = []
            for epoch in range(random_number):
                cur_epoch_loss = 0
                net.train()
                train_total = 0
                train_correct = 0
                for inputs, labels in train_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    # 优化器梯度归零
                    optimizer.zero_grad()
                    # 正向传播 +　反向传播 + 优化 
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    cur_epoch_loss += loss.item()
                    # 计算预测结果
                    predicted = torch.argmax(outputs.data, dim=1)
                    if config.record_infos_lossNaccuracy == True:
                        # 累加总样本数
                        train_total += labels.size(0)
                        # 累加正确预测的样本数
                        train_correct += (predicted == labels).sum().item()
                
                if config.print_epoch_loss:
                    print('[Epoch: %d]   [current epoch train loss: %.8f]' % (
                    epoch + 1, cur_epoch_loss / (1 - validateNtest_ratio)))
                if epoch + 5 >= random_number:
                    last_epoch_loss += cur_epoch_loss
                    # 记录当前轮次的训练集损失
                if config.record_infos_lossNaccuracy == True:
                    train_losses.append(cur_epoch_loss / (1 - validateNtest_ratio))
                    train_accuracy.append(train_correct / train_total)

                if config.record_infos_lossNaccuracy == True:
                    # 验证集评估
                    net.eval()  # 设置模型为评估模式
                    val_epoch_loss = 0
                    val_total = 0
                    val_correct = 0
                    with torch.no_grad():
                        for inputs, labels in validate_loader:
                            inputs = inputs.to(device)
                            labels = labels.to(device)
                            outputs = net(inputs)
                            loss = criterion(outputs, labels)
                            val_epoch_loss += loss.item()
                            # 计算预测结果
                            predicted = torch.argmax(outputs.data, dim=1)
                            # 累加总样本数
                            val_total += labels.size(0)
                            # 累加正确预测的样本数
                            val_correct += (predicted == labels).sum().item()
                    # print(inputs.shape)

                    # 记录当前轮次的验证集损失
                    val_losses.append(val_epoch_loss / validateNtest_ratio / (1 - test_ratio))
                    val_accuracy.append(val_correct / val_total)

                    if config.print_epoch_loss:
                        print('[Epoch: %d]   [current epoch val   loss: %.8f]   [current epoch val   accuracy: %.8f]' % (
                            epoch + 1, val_epoch_loss / validateNtest_ratio / (1 - test_ratio), val_correct / val_total),end='\n\n')
            print(f'Finished Training {i + 1}')
            # 存储到 .npy 文件
            if config.record_infos_lossNaccuracy == True:
                train_losses_all.append(train_losses)
                val_losses_all.append(val_losses)
                train_accuracy_all.append(train_accuracy)
                val_accuracy_all.append(val_accuracy)

            # 模型测试
            net.eval()
            y_pred_test = []
            for inputs, _ in validate_loader:
                inputs = inputs.to(device)
                outputs = net(inputs)
                outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
                y_pred_test.extend(outputs)

            accuracy = accuracy_score(yvalidate, np.array(y_pred_test))
            accuracies.append(accuracy)
            print(f"(validate set) Accuracy of training {i + 1} is {accuracy}, number_epochs is {random_number}")
            if accuracy > best_accuracy or (accuracy == best_accuracy and last_epoch_loss < best_net_last_epoch_loss):
                best_accuracy = accuracy
                best_net = net
                best_net_num_epochs = random_number
                best_net_last_epoch_loss = last_epoch_loss

            net.eval()
            y_pred_test = []
            y_t = []
            for inputs, label in train_loader:
                inputs = inputs.to(device)
                outputs = net(inputs)
                outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
                y_pred_test.extend(outputs)
                y_t.extend(label)

            accuracy = accuracy_score(np.array(y_t), np.array(y_pred_test))
            print(f"(train set) Accuracy of training {i + 1} is {accuracy}")

            net.eval()
            y_pred_test = []
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                outputs = net(inputs)
                outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
                y_pred_test.extend(outputs)

            accuracy = accuracy_score(ytest, np.array(y_pred_test))
            print(f"(test set) Accuracy of training {i + 1} is {accuracy}")
            print(f"Best accuracy among first {i + 1} trainings is {best_accuracy}, number_epochs is {best_net_num_epochs}, last 5 epoch_loss is {best_net_last_epoch_loss}")
            time_end_cur_net = time.time()
            # 计算训练所用时间
            training_time = time_end_cur_net - time_begin_cur_net

            # 格式化训练结束时间
            end_time_formatted = datetime.fromtimestamp(time_end_cur_net).strftime('%Y-%m-%d %H:%M:%S')

            print(f'训练结束时间: {end_time_formatted}')
            print(f'训练所用时间: {training_time} 秒', end='\n\n')
        print(f'submodel {submodel_idx + 1} info: kernel_spatial={kernel_spatial}')
        print("Best accuracy among all trainings:", best_accuracy, "number_epochs is", best_net_num_epochs, f", last_epoch_loss is {best_net_last_epoch_loss}")
        
        models_trained.append(copy.deepcopy(best_net))
        # 生成分类报告等评估指标
        best_net.eval()
        y_pred_test_best = []
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = best_net(inputs)
            outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            y_pred_test_best.extend(outputs)

        classification = classification_report(ytest, np.array(y_pred_test_best), digits=4)
        cm = confusion_matrix(ytest, np.array(y_pred_test_best))
        accuracy = accuracy_score(ytest, np.array(y_pred_test_best))
        kappa = cohen_kappa_score(ytest, np.array(y_pred_test_best))
        print(classification, end='\n\n')
        print(cm, end='\n\n')
        print(accuracy, end='\n\n')
        print(kappa, end='\n\n')
        model_suffix = "_{}_{}_{}_".format(
            config.kernel_spatial_list[submodel_idx],
            best_net_num_epochs,
            selected_data) 
        model_dir = config.model_dir_path + '/' + selected_data
        torch.save(best_net, model_dir + '/' + config.model_list[submodel_idx] + model_suffix + "{:.5f}".format(accuracy)[2:] + '.pth')
        if config.record_infos_lossNaccuracy == True:
            np.save(f'./train_infos/{selected_data}/train_losses_{selected_data}_{config.patch_size}_{str(config.learning_rate)[2:]}_{config.kernel_spatial_list[submodel_idx]}_{config.net_num}.npy', np.array(train_losses_all))
            np.save(f'./train_infos/{selected_data}/val_losses_{selected_data}_{config.patch_size}_{str(config.learning_rate)[2:]}_{config.kernel_spatial_list[submodel_idx]}_{config.net_num}.npy', np.array(val_losses_all))
            np.save(f'./train_infos/{selected_data}/train_accuracies_{selected_data}_{config.patch_size}_{str(config.learning_rate)[2:]}_{config.kernel_spatial_list[submodel_idx]}_{config.net_num}.npy', np.array(train_accuracy_all))
            np.save(f'./train_infos/{selected_data}/val_accuracies_{selected_data}_{config.patch_size}_{str(config.learning_rate)[2:]}_{config.kernel_spatial_list[submodel_idx]}_{config.net_num}.npy', np.array(val_accuracy_all))
    y_pred_test_best = None

    for md in models_trained:
        md.eval()
        y_pred_test_best_cur = np.empty((0, class_num))

        for inputs, label in test_loader:
            inputs = inputs.to(device)
            outputs = md(inputs)
            y_pred_test_best_cur = np.vstack((y_pred_test_best_cur, outputs.detach().cpu().numpy()))
        if y_pred_test_best is None:
            y_pred_test_best = y_pred_test_best_cur
        else:
            y_pred_test_best = (y_pred_test_best + y_pred_test_best_cur)
    predicted_classes = np.argmax(y_pred_test_best, axis=1)

    classification = classification_report(ytest, np.array(predicted_classes), digits=4)
    cm = confusion_matrix(ytest, np.array(predicted_classes))
    accuracy = accuracy_score(ytest, np.array(predicted_classes))
    kappa = cohen_kappa_score(ytest, np.array(predicted_classes))
    print(classification, end='\n\n')
    print(cm, end='\n\n')
    print(accuracy, end='\n\n')
    print(kappa, end='\n\n')