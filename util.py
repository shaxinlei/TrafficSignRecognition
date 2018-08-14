import numpy as np
import matplotlib.pyplot as plt
import os
import re


def plot_loss(loss_list, lr_list):
    plt.figure(figsize=(8, 6))
    x = np.arange(0, 500, 5)

    for i in range(len(loss_list)):
        loss = []
        for j in x:
            loss.append(loss_list[i][j])
        if i is 0:
            plt.plot(x, loss, label='QoI-RA DataSet', marker='s', linestyle='-',markersize=8)
        else: #新的数据集
        # if i is 1: #老的数据
            plt.plot(x, loss, label='Original DataSet',marker='^',linestyle='-',markersize=8)

    plt.legend(loc='upper right', fontsize=20)
    plt.xlim(0, 500)
    plt.ylim(0, 4)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Step', fontsize=20)
    plt.ylabel('Loss Value', fontsize=20)
    #plt.title('Loss Value VS Step', fontsize=20)
    plt.show()


def load_loss_data(data_dir):
    loss_list = []
    lr_list = []
    file_names = [f for f in os.listdir(data_dir) if  f.startswith('loss')]
    for file_name in file_names:
        loss = []
        lr = float(re.findall(r'loss(.+)\.txt', file_name)[0])
        lr_list.append(lr)
        file_dir = os.path.join(data_dir, file_name)
        with open(file_dir, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line == '':
                    print('null string')
                    break
                loss.append(float(line))
            loss_list.append(loss)
    return loss_list, lr_list


def load_accuracy_data(data_dir):
    accuracy_list = []
    batchSize_list = []
    file_names = [f for f in os.listdir(data_dir) if f.startswith('accuracy')]
    for file_name in file_names:
        accuracy = []
        batchSize = float(re.findall(r'accuracy(.+)\.txt', file_name)[0])
        batchSize_list.append(batchSize)
        file_dir = os.path.join(data_dir, file_name)
        with open(file_dir, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line == '':
                    print('null string')
                    break
                accuracy.append(float(line))
            accuracy_list.append(accuracy)
    return accuracy_list, batchSize_list


def plot_accuracy(accuracy_list, batchSize_list):
    plt.figure(figsize=(8, 6))
    x = np.arange(0, 500, 5)
    marker_list = ['|', '^', 'x']
    color_list = ['red', 'orange', 'green']
    for i in range(len(accuracy_list)):
        accuracy = []
        for j in x:
            accuracy.append(accuracy_list[i][j])
        plt.plot(x, accuracy, label='Batch Size:'+str(batchSize_list[i]), color=color_list[i], marker=marker_list[i],markersize=10)
    plt.legend(loc='lower right', fontsize=20)
    plt.xlabel('Step', fontsize=20)
    plt.ylabel('Classification Accuracy', fontsize=20)
    plt.xlim(0, 500)
    plt.ylim(0, 1)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title('Classification Accuracy VS Step', fontsize=20)
    plt.show()

def plot_accuracy(x, y):
    plt.figure(figsize=(8, 6))
    marker_list = ['^', 's']
    color_list = ['blue', 'red']
    label_list = [ 'Original DataSet','QoI-RA DataSet']
    for i in range(len(y)):
        plt.plot(x, y[i], label= label_list[i], color=color_list[i], marker=marker_list[i],markersize=10)
    plt.legend(loc='lower right', fontsize=17)
    plt.xlabel('Step', fontsize=19)
    plt.ylabel('Training Classification Accuracy', fontsize=19)
    plt.xlim(0, 500)
    plt.ylim(0.73, 1)
    plt.xticks(fontsize=19)
    plt.yticks(fontsize=19)
   # plt.title('Classification Accuracy VS Step', fontsize=20)
    plt.show()

if __name__ == '__main__':

    # accuracy_list, batchSize_list = load_accuracy_data('logs')
    # plot_accuracy(accuracy_list, batchSize_list)

    # loss_list, lr_list = load_loss_data('logs')
    # plot_loss(loss_list, lr_list)
    x = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    y = []
    y2 = [0.77, 0.83, 0.9, 0.91, 0.94, 0.948, 0.953, 0.958, 0.975, 0.991]
    y1 = [0.75, 0.8203125, 0.890625, 0.8984375, 0.9175, 0.9396875, 0.929375, 0.935, 0.9584375, 0.968]
    y.append(y1)
    y.append(y2)
    plot_accuracy(x, y)





