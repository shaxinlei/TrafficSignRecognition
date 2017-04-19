import os
import random
import skimage.data
import skimage.transform
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def load_data(data_dir):
    """Loads a data set and returns two lists:
    images: a list of Numpy arrays, each representing an image.
    labels: a list of numbers that represent the images labels.
    """
    # Get all subdirectories of data_dir. Each represents a label.
    directories = [d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d))]
    # Loop through the label directories and collect the data in
    # two lists, labels and images.
    labels = []
    images = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f)
                      for f in os.listdir(label_dir) if f.endswith(".ppm")]
        # For each label, load it's images and add them to the images list.
        # And add the label number (i.e. directory name) to the labels list.
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels

# Load training and testing datasets.
#train_data_dir = os.path.join("C:\\Users\\sxl\\Desktop\\traffic-signs-tensorflow\\datasets","Training")
#test_data_dir = os.path.join("C:\\Users\\sxl\\Desktop\\traffic-signs-tensorflow\\datasets","Testing")
train_data_dir = os.path.join("C:\\temp\\traffic-signs-tensorflow\\datasets","Training")
test_data_dir = os.path.join("C:\\temp\\traffic-signs-tensorflow\\datasets","Testing")
# train_data_dir = os.path.join("D:\\project\\traffic\\", "Training")
# test_data_dir = os.path.join("D:\\project\\traffic\\", "Testing")

#加载训练数据集
images, labels = load_data(train_data_dir)

#输出每个目录下的第一个图像
def display_images_and_labels(images, labels):
    """Display the first image of each label."""
    unique_labels = set(labels)
    plt.figure(figsize=(15, 15))
    i = 1
    for label in unique_labels:
        # Pick the first image for each label.
        image = images[labels.index(label)]
        plt.subplot(8, 8, i)  # A grid of 8 rows x 8 columns
        plt.axis('off')          #不显示坐标
        plt.title("Label {0} ({1})".format(label, labels.count(label)))
        i += 1
        _ = plt.imshow(image)    #show image
    plt.show()
#输出目录下的第一个图像
#display_images_and_labels(images,labels)

'''
#将label向量转化为矩阵[branch,62]
def labelsToMatrix(input,len):
    labels_matraix = np.zeros([len,62])
    i = 0
    for k in input:
        labels_matraix[i][k] = 1
        i = i+1
    return labels_matraix
'''


#从数据集中随机选择n张图片
def batch(images,lanels,n):
    sample_indexes = random.sample(range(len(images)), n)
    sample_images = [images[i] for i in sample_indexes]
    label_s = [lanels[i] for i in sample_indexes]
   # print("选择标签")
  #  print(label_s)
    #sample_labels = labelsToMatrix(label_s,len(label_s))
    return sample_images,label_s

# Resize images
images32 = [skimage.transform.resize(image, (32, 32))
                for image in images]

## 将label列表转化为矩阵
#labels_train_all = labelsToMatrix(labels,len(labels))
labels_train_all = np.array(labels)
images_train_all = np.array(images32)

#加载测试数据集
test_images, test_labels = load_data(test_data_dir)
## 将label列表转化为矩阵
#labels_test_all = labelsToMatrix(test_labels,len(test_labels))

# Transform the images, just like we did with the training set.
test_images32 = [skimage.transform.resize(image, (32, 32))
                 for image in test_images]
images_test_all = np.array(test_images32)
labels_test_all = np.array(test_labels)

'''
# 计算分类精确度
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = session.run(prediction, feed_dict={images_ph: v_xs, keep_prob:1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))    #argmax函数返回每行最大值的索引（axis=0每一列）
    print("预测分类")
    print(session.run(tf.argmax(y_pre,1), feed_dict={images_ph: v_xs, labels_ph: v_ys, keep_prob:1}))
    print("实际分类")
    print(session.run(tf.argmax(v_ys,1), feed_dict={images_ph: v_xs, labels_ph: v_ys, keep_prob:1}))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))   # 将correct_prediction 转为float32类型，然后求平均值
    result = session.run(accuracy, feed_dict={images_ph: v_xs, labels_ph: v_ys})
    return result
'''


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)  # 变量的初始值为截断正太随机分布，标准差为0.1
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)    # 生成常量tensor ，值为0.1
    return tf.Variable(initial)

def conv2d(x, W):
    """
       tf.nn.conv2d功能：给定4维的input和filter，计算出一个2维的卷积结果
       前几个参数分别是input, filter, strides, padding, use_cudnn_on_gpu, ...
       input   的格式要求为一个张量，[batch, in_height, in_width, in_channels],批次数，图像高度，图像宽度，通道数
       filter  的格式为[filter_height, filter_width, in_channels, out_channels]，滤波器高度，宽度，输入通道数，输出通道数
       strides 一个长为4的list. 表示每次卷积以后在input中滑动的距离
       padding 有SAME和VALID两种选项，表示是否要保留不完全卷积的部分。如果是SAME，则保留
       use_cudnn_on_gpu 是否使用cudnn加速。默认是True
       """
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')   #进行卷积操作，步长为1,padding为SAME

def max_pool_2x2(x):
    """
    tf.nn.max_pool 进行最大值池化操作,而avg_pool 则进行平均值池化操作
    几个参数分别是：value, ksize, strides, padding,
    value:  一个4D张量，格式为[batch, height, width, channels]，与conv2d中input格式一样
    ksize:  长为4的list,表示池化窗口的尺寸
    strides: 窗口的滑动值，与conv2d中的一样
    padding: 与conv2d中用法一样。
    """
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')    #进行池化操作，池化窗口大小为[1,2,2,1],窗口步长为[1,2,2,1]

# Placeholders for inputs and labels.
images_ph = tf.placeholder(tf.float32, [None, 32, 32, 3])    #输入图像shape为[batch,32,32,3]   32*32像素 3通道
labels_ph = tf.placeholder(tf.int32, [None])


## conv1 layer ##
"""
# 第一层
# 卷积核(filter)的尺寸是5*5, 通道数为1，输出通道为32，即feature map 数目为32
# 又因为strides=[1,1,1,1] 所以单个通道的输出尺寸应该跟输入图像一样。即总的卷积输出应该为?*32*32*32
# 也就是单个通道输出为32*32，共有32个通道,共有?个批次
# 在池化阶段，ksize=[1,2,2,1] 那么卷积结果经过池化以后的结果，其尺寸应该是？*16*16*32
"""
W_conv1 = weight_variable([5,5,3,32]) # patch 5x5, in size 3, out size 32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(images_ph, W_conv1) + b_conv1) # output size 32x32x32
h_pool1 = max_pool_2x2(h_conv1)                            # output size 16x16x32

## conv2 layer ##
"""
# 第二层
# 卷积核5*5，输入通道为32，输出通道为64。
# 卷积前图像的尺寸为 ?*16*16*32， 卷积后为?*16*16*64
# 池化后，输出的图像尺寸为?*8*8*64
"""
W_conv2 = weight_variable([5,5, 32, 64]) # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 16x16x64
h_pool2 = max_pool_2x2(h_conv2)                          # output size 8x8x64

## fc1 layer ##
# 第三层 是个全连接层,输入维数8*8*64, 输出维数为2048
W_fc1 = weight_variable([8*8*64,1024])
b_fc1 = bias_variable([1024])
# [n_samples, 8, 8, 64] ->> [n_samples, 8*8*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


## fc2 layer ##
# 第四层，输入1024维，输出62维，也就是具体的0~61分类
W_fc2 = weight_variable([1024, 62])
b_fc2 = bias_variable([62])
logits = tf.add(tf.matmul(h_fc1, W_fc2) , b_fc2)

predicted_labels = tf.argmax(logits, 1)

xlabels = tf.cast(labels_ph,tf.int64)

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits( labels=xlabels,logits=logits))

#cross_entropy = tf.reduce_mean(-tf.reduce_sum(labels_ph * tf.log(prediction), reduction_indices=[1]))      # 损失函数，交叉熵
#cross_entropy = -tf.reduce_sum(labels_ph * tf.log(prediction))
train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)   #使用adam优化

# Create a session to run the graph we created.
session = tf.Session()

# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()

# First step is always to initialize all variables.
session.run(init)

for i in range(100):
    image_train,labels_train = batch(images32,labels,128)   #从训练集中随机选取128张图片
    #print("训练数据")
    #print(labels_train)
    session.run(train_step, feed_dict={images_ph: image_train, labels_ph: labels_train})   #利用训练集的数据训练模型
    print("step %d" %i)
    image_test, labes_test = batch(test_images32, test_labels,128)  # 从测试集中随机选取128张图片
    print("测试数据")
    print(labes_test)
    #print("step %d, training accuracy %g"%(i,compute_accuracy(image_test, labes_test)))  #利用测试集的数据计算分类精确度
    # Run predictions against the full test set.
    predicted = session.run([predicted_labels],
                            feed_dict={images_ph: image_test})[0]
    print("预测数据")
    print(predicted)
    # Calculate how many matches we got.
    match_count = sum([int(y == y_) for y, y_ in zip(labes_test, predicted)])
    accuracy = match_count / len(labes_test)
    print("Accuracy: {:.3f}".format(accuracy))

for i in range(10):
    sample_images, sample_labels = batch(test_images32, test_labels, 10)  # 从测试集中随机选取10张图片
    predicted = session.run([predicted_labels],
                            feed_dict={images_ph: sample_images})[0]
    # Display the predictions and the ground truth visually.
    fig = plt.figure(figsize=(10, 10))
    for i in range(len(sample_images)):
        truth = sample_labels[i]
        prediction = predicted[i]
        plt.subplot(5, 2, 1 + i)
        plt.axis('off')
        color = 'green' if truth == prediction else 'red'
        plt.text(40, 10, "Truth:        {0}\nPrediction: {1}".format(truth, prediction),
                 fontsize=12, color=color)
        plt.imshow(sample_images[i])
    plt.show()

#session.close()