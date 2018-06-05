import numpy as np
import matplotlib.pyplot as plt
import os
import re


def plot_loss(loss_list, lr_list):
    plt.figure(figsize=(8, 5))
    x = np.arange(0, 500, 5)

    for i in range(len(loss_list)):
        loss = []
        for j in x:
            loss.append(loss_list[i][j])
        if i is 0:
            plt.plot(x, loss, label='QoS-RC DataSet', linestyle='-')
        else:
            plt.plot(x, loss, label='Original DataSet', linestyle='-')
    plt.legend(loc='upper right')
    plt.ylim(0, 4)
    plt.xlabel('step')
    plt.ylabel('loss value')
    plt.title('loss change over step')
    plt.show()


def load_data(data_dir):
    loss_list = []
    lr_list = []
    file_names = [f for f in os.listdir(data_dir) if f.startswith('loss')]
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


if __name__ == '__main__':
    loss_list, lr_list = load_data('logs')
    plot_loss(loss_list, lr_list)




