import os
import random
import skimage.data
import skimage.transform
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# training and testing dir
train_data_dir = os.path.join("C:\\Users\\sxl\\Desktop\\traffic-signs-tensorflow\\datasets","Training")
test_data_dir = os.path.join("C:\\Users\\sxl\\Desktop\\traffic-signs-tensorflow\\datasets","Testing")
#train_data_dir = os.path.join("C:\\temp\\traffic-signs-tensorflow\\datasets","Training")
#test_data_dir = os.path.join("C:\\temp\\traffic-signs-tensorflow\\datasets","Testing")
# train_data_dir = os.path.join("D:\\project\\traffic\\", "Training")
# test_data_dir = os.path.join("D:\\project\\traffic\\", "Testing")

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

#从数据集中随机选择n张图片
def batch(images,lanels,n):
    sample_indexes = random.sample(range(len(images)), n)
    sample_images = [images[i] for i in sample_indexes]
    label_s = [lanels[i] for i in sample_indexes]
   # print("选择标签")
   # print(label_s)
    return sample_images,label_s

def _variable_on_cpu(shape, initializer):
  with tf.device('/cpu:0'):
    dtype = tf.float32
    var = tf.get_variable(shape, initializer=initializer, dtype=dtype)
  return var


def _variable_with_weight_decay(shape, stddev, wd):

  dtype = tf.float32
  var = _variable_on_cpu(shape,tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  return var

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)  # 变量的初始值为截断正太随机分布，标准差为0.1
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)    # 生成常量tensor ，值为0.1
    return tf.Variable(initial)

def inference(images,len):
  # conv1
  #kernel = _variable_with_weight_decay(shape=[5, 5, 3, 64], stddev=5e-2, wd=0.0)
  kernel = tf.get_variable(name= "kernel1",shape=[5, 5, 3, 64], initializer=tf.truncated_normal_initializer(stddev=5e-2))
  conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
  #biases = _variable_on_cpu([64], tf.constant_initializer(0.0))
  biases = tf.get_variable(name= "biases1",shape=[64],initializer=tf.constant_initializer(0.0))
  pre_activation = tf.nn.bias_add(conv, biases)
  conv1 = tf.nn.relu(pre_activation)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,)

  # conv2
  #kernel = _variable_with_weight_decay( shape=[5, 5, 64, 64],stddev=5e-2,wd=0.0)
  kernel = tf.get_variable(name= "kernel2",shape=[5, 5, 64, 64], initializer=tf.truncated_normal_initializer(stddev=5e-2))
  conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
 # biases = _variable_on_cpu( [64], tf.constant_initializer(0.1))
  biases = tf.get_variable(name= "biases2",shape=[64], initializer=tf.constant_initializer(0.1))
  pre_activation = tf.nn.bias_add(conv, biases)
  conv2 = tf.nn.relu(pre_activation)

  # norm2
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,)
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

  # local3

  # Move everything into depth so we can perform a single matrix multiply.
  reshape = tf.reshape(pool2, [len, -1])
  #dim = reshape.get_shape()[1].value
  #weights = _variable_with_weight_decay( shape=[dim, 384],stddev=0.04, wd=0.004)
  weights = tf.get_variable(name= "weights1",shape=[8*8*64, 384], initializer=tf.truncated_normal_initializer(stddev=0.04))
  #biases = _variable_on_cpu([384], tf.constant_initializer(0.1))
  biases = tf.get_variable(name= "biases3",shape=[384], initializer=tf.constant_initializer(0.1))
  local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases)

  # local4

  #weights = _variable_with_weight_decay(shape=[384, 192], stddev=0.04, wd=0.004)
  weights = tf.get_variable(name= "weights2",shape=[384, 192], initializer=tf.truncated_normal_initializer(stddev=0.04))
  #biases = _variable_on_cpu([192], tf.constant_initializer(0.1))
  biases = tf.get_variable(name= "biases4",shape=[192], initializer=tf.constant_initializer(0.1))
  local4 = tf.nn.relu(tf.matmul(local3, weights) + biases)

  # linear layer(WX + b),
  # We don't apply softmax here because
  # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
  # and performs the softmax internally for efficiency.

 # weights = _variable_with_weight_decay([192, 62], stddev=1/192.0, wd=0.0)
  weights = tf.get_variable(name= "weights3",shape=[192, 62], initializer=tf.truncated_normal_initializer(stddev=1/192.0))
  biases = tf.get_variable(name= "biases5",shape=[62], initializer=tf.constant_initializer(0.0))
  softmax_linear = tf.add(tf.matmul(local4, weights), biases)

  return softmax_linear


##加载数据集
##======================================================##
#加载训练数据集
images, labels = load_data(train_data_dir)
# Resize images
train_images32 = [skimage.transform.resize(image, (32, 32))
                for image in images]

labels_train_all = np.array(labels)
images_train_all = np.array(train_images32)

#加载测试数据集
test_images, test_labels = load_data(test_data_dir)
# Transform the images, just like we did with the training set.
test_images32 = [skimage.transform.resize(image, (32, 32))
                 for image in test_images]
images_test_all = np.array(test_images32)
labels_test_all = np.array(test_labels)

##==========================================================##


# Placeholders for inputs and labels.
images_ph = tf.placeholder(tf.float32, [None, 32, 32, 3])    #输入图像shape为[batch,32,32,3]   32*32像素 3通道
labels_ph = tf.placeholder(tf.float32, [None])
len = tf.placeholder(tf.int32)
keep_prob = tf.placeholder(tf.float32)

logits = inference(images_ph,len)

predicted_labels = tf.argmax(logits, 1)


labels = tf.cast(labels_ph, tf.int64)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
cross_entropy_mean = tf.reduce_mean(cross_entropy)

# Create training op.
train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy_mean)

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

for i in range(20):
    image_train,labels_train = batch(train_images32,labels,1000)   #从训练集中随机选取100张图片
    #print("训练数据")
    #print(labels_train)
    session.run(train, feed_dict={images_ph: image_train, labels_ph: labels_train, len: 1000, keep_prob: 1})   #利用训练集的数据训练模型
    image_test, labes_test = batch(test_images32, test_labels,1000)  # 从测试集中随机选取100张图片

    # Run predictions against the full test set.
    predicted = session.run([predicted_labels],
                            feed_dict={images_ph: image_test, len: 1000})[0]
    # Calculate how many matches we got.
    match_count = sum([int(y == y_) for y, y_ in zip(labes_test, predicted)])
    accuracy = match_count / len(labes_test)
    print("Accuracy: {:.3f}".format(accuracy))

session.close()