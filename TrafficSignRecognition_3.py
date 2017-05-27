import os
import random
import skimage.data
import skimage.transform
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def load_data(data_dir):
    # 加载数据集并返回两个列表：
    # images：Numpy数组的列表，每个数组表示图像.
    # labels：表示图像标签的数字列表。
    # 获取data_dir的所有子目录，每个子目录代表一个标签.
    directories = [d for d in os.listdir(data_dir)    #os.listdir:获得目录内容
                   if os.path.isdir(os.path.join(data_dir, d))]    #os.path.isdir:判断某一路径是否为目录   os.path.join:拼接路径
    #循环标签目录并收集数据
    #两个列表，标签和图像
    labels = []
    images = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f)
                      for f in os.listdir(label_dir) if f.endswith(".ppm")]
        # 对于每个label，加载它的图像，并将它们添加到image列表。
        # 将标签号（即目录名）添加到标签列表中.
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels

# 加载 training and testing 数据集.
#train_data_dir = os.path.join("C:\\Users\\sxl\\Desktop\\traffic-signs-tensorflow\\datasets","Training")
#test_data_dir = os.path.join("C:\\Users\\sxl\\Desktop\\traffic-signs-tensorflow\\datasets","Testing")
train_data_dir = os.path.join("C:\\temp\\traffic-signs-tensorflow\\datasets", "Training")
test_data_dir = os.path.join("C:\\temp\\traffic-signs-tensorflow\\datasets", "Testing")

#从数据集中随机选择n张图片
def batch(images,labels,n):
    sample_indexes = random.sample(range(len(images)), n)  #random.sample:从指定的序列中，随机的截取指定长度的片断，不作原地修改
    sample_images = [images[i] for i in sample_indexes]
    label_s = [labels[i] for i in sample_indexes]
    return sample_images,label_s

##===============================================================================##
                                    ##start加载数据集##

#加载训练数据集
images, labels = load_data(train_data_dir)

# 调整图像大小
images32 = [skimage.transform.resize(image, (32, 32))
                for image in images]
labels_train_all = np.array(labels)
images_train_all = np.array(images32)

#加载测试数据集
test_images, test_labels = load_data(test_data_dir)

# 调整图像大小
test_images32 = [skimage.transform.resize(image, (32, 32))
                 for image in test_images]
images_test_all = np.array(test_images32)
labels_test_all = np.array(test_labels)

                                 ##加载数据集end##
##===============================================================================##


##===============================================================================##
                                ##定义构建卷积神经网络的函数##

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)  # 返回一个tensor其中的元素服从截断正态分布，标准差为0.1
    return tf.Variable(initial,name = "W")

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)    # 生成常量tensor ，值为0.1
    return tf.Variable(initial,name = "b")

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

##===============================================================================##


##===============================================================================##
                                ##start构建卷积神经网络##

# Placeholders for inputs and labels.
with tf.name_scope('inputs'):
    images_ph = tf.placeholder(tf.float32, [None, 32, 32, 3],name="images_ph")    #输入图像shape为[batch,32,32,3]   32*32像素 3个通道
    labels_ph = tf.placeholder(tf.int32, [None],name="labels_ph")

## conv1 layer ##
"""
# 第一层
# 卷积核(filter)的尺寸是5*5, 通道数为1，输出通道为32，即feature map 数目为32
# 又因为strides=[1,1,1,1] 所以单个通道的输出尺寸应该跟输入图像一样。即总的卷积输出应该为?*32*32*32
# 也就是单个通道输出为32*32，共有32个通道,共有?个批次
# 在池化阶段，ksize=[1,2,2,1] 那么卷积结果经过池化以后的结果，其尺寸应该是？*16*16*32
"""
with tf.name_scope('conv1_layer'):
    with tf.name_scope('Weights'):
        W_conv1 = weight_variable([5,5,3,32]) # patch 5x5, in size 3, out size 32
        tf.summary.histogram('conv1_layer/weights', W_conv1)
    with tf.name_scope('biases'):
        b_conv1 = bias_variable([32])
        tf.summary.histogram('conv1_layer/biases', b_conv1)
    with tf.name_scope('conv1'):
        h_conv1 = tf.nn.relu(conv2d(images_ph, W_conv1) + b_conv1) # 非线性处理 output size 32x32x32
        tf.summary.histogram('conv1_layer/outputs', h_conv1)

with tf.name_scope('pool1_layer'):
    h_pool1 = max_pool_2x2(h_conv1)                            # output size 16x16x32

## conv2 layer ##
"""
# 第二层
# 卷积核5*5，输入通道为32，输出通道为64。
# 卷积前图像的尺寸为 ?*16*16*32， 卷积后为?*16*16*64
# 池化后，输出的图像尺寸为?*8*8*64
"""
with tf.name_scope('conv2_layer'):
    with tf.name_scope('Weights'):
        W_conv2 = weight_variable([5,5, 32, 64])  # patch 5x5, in size 32, out size 64
        tf.summary.histogram('conv2_layer/weights', W_conv2)
    with tf.name_scope('biases'):
        b_conv2 = bias_variable([64])
        tf.summary.histogram('conv2_layer/biases', b_conv2)
    with tf.name_scope('conv2'):
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 16x16x64
        tf.summary.histogram('conv2_layer/outputs', h_conv2)

with tf.name_scope('pool2_layer'):
    h_pool2 = max_pool_2x2(h_conv2)                          # output size 8x8x64

## fc1 layer ##
# 第三层 是个全连接层,输入维数8*8*64, 输出维数为1024
with tf.name_scope('layer3'):
    with tf.name_scope('Weights'):
        W_fc1 = weight_variable([8*8*64,1024])  #扁平化
        tf.summary.histogram('layer3/weights', W_fc1)
    with tf.name_scope('biases'):
        b_fc1 = bias_variable([1024])
        tf.summary.histogram('layer3/biases', b_fc1)
    # [n_samples, 8, 8, 64] ->> [n_samples, 8*8*64]
    h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*64])
    with tf.name_scope('Wx_plus_b'):
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)   #tf.matmul：矩阵相乘
        tf.summary.histogram('layer3/outputs', h_fc1)


## fc2 layer ##
# 第四层，输入1024维，输出62维，也就是具体的0~61分类
with tf.name_scope('layer4'):
    with tf.name_scope('Weights'):
        W_fc2 = weight_variable([1024, 62])
        tf.summary.histogram('layer4/weights', W_fc2)
    with tf.name_scope('biases'):
        b_fc2 = bias_variable([62])
        tf.summary.histogram('layer4/biases', b_fc2)
    with tf.name_scope('Wx_plus_b'):
        logits = tf.add(tf.matmul(h_fc1, W_fc2) , b_fc2)
        tf.summary.histogram('layer4/outputs', logits)

                                    ##神经网络构建end##
##===============================================================================##


predicted_labels = tf.argmax(logits, 1)   # 返回某一维度的最大值
xlabels = tf.cast(labels_ph,tf.int64)      #强制转化，将float转化为int
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=xlabels,logits=logits))
    tf.summary.scalar('loss', loss)
with tf.name_scope('train_step'):
    train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)   #使用adam优化

# 创建一个session来运行我们创建的图.
session = tf.Session()
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()   #不同版本的TensorFlow有不同的参数初始化方法

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/",session.graph)

# 第一步始终是初始化所有变量.
session.run(init)    # 在session里面运行模型，并且进行初始化

for i in range(100):   # 对模型进行训练
    image_train,labels_train = batch(images32,labels,128)   #从训练集中随机选取128张图片

    _,loss_value,result = session.run([train_step,loss,merged], feed_dict={images_ph: image_train, labels_ph: labels_train})  # 每次运行train_step时，将之前所选择的数据，填充至所设置的占位符中，作为模型的输入
    #print("step: %d" %i)
    print("Step: {0}" .format(i))
    writer.add_summary(result,i)

    image_test, labels_test = batch(test_images32, test_labels,128)  # 从测试集中随机选取128张图片
    #print("测试数据")
    #print(labels_test)
    predicted = session.run([predicted_labels],
                            feed_dict={images_ph: image_test})[0]
    #print("预测数据")
    #print(predicted)
    # 计算得到匹配的数量.
    match_count = sum([int(y == y_) for y, y_ in zip(labels_test, predicted)])
    accuracy = match_count / len(labels_test)
    print("Accuracy: {0},    Loss:{1}".format(accuracy,loss_value))

print("\n************Caculate the accuracy of test data**************")
print("\n")
print("\n")
print("测试数据")
print("num_of_testData:{0}".format(images_test_all))
for i in labels_test_all:
    print(i," ",end="")

predicted_all = session.run([predicted_labels],feed_dict={images_ph:images_test_all})[0]
print("\n预测数据")
for i in predicted_all:
    print(i," ",end="")
match_count_all = sum([int(y == y_) for y, y_ in zip(labels_test_all, predicted_all)])
accuracy = match_count_all/ len(labels_test_all)
print("\nAll test images' accuracy: {0}".format(accuracy))



for i in range(10):
    print("================================================================")
    sample_images, sample_labels = batch(test_images32, test_labels, 10)  # 从测试集中随机选取10张图片
    predicted = session.run([predicted_labels],
                            feed_dict={images_ph: sample_images})[0]
    # Display the predictions and the ground truth visually.
    fig = plt.figure(figsize=(10, 10))   #在10英寸*10英寸 的画布上画图
    for i in range(len(sample_images)):
        truth = sample_labels[i]
        prediction = predicted[i]
        plt.subplot(5, 2, 1 + i)  # 整个绘图区域被分成5行和2列，指定创建的象所在的区域....
        plt.axis('off')   # 关掉图像坐标
        color = 'green' if truth == prediction else 'red'
        plt.text(40, 10, "Truth:        {0}\nPrediction: {1}".format(truth, prediction),
                 fontsize=12, color=color)
        plt.imshow(sample_images[i])
    plt.show()


#session.close()