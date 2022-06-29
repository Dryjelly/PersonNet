import os
import cv2
import random
import numpy as np
import tensorflow as tf
from utils import *


# 학습 데이터를 뽑습니다. -------------------------------------------

train_dir = '/home/esdl/tensorflow/cuhk03_release/images/detected'
train_list, train_data1, train_data2, train_CoN = Image_Data(train_dir,5)

data_index = list(range(len(train_data1)))

# 테스트 데이터를 뽑습니다.

test_dir = '/home/esdl/tensorflow/campus'
test_list, test_data1, test_data2, test_CoN = Image_Data(test_dir)

test_data_index = list(range(len(test_data1)))
random.shuffle(test_data_index)


# train !!! -------------------------------------------------------
tf.reset_default_graph()

data_x_size = 160
data_y_size = 60

batch_prob = tf.placeholder(tf.bool, name="batch_prob")

X1 = tf.placeholder(tf.float32, [None, data_x_size, data_y_size, 3], name="input1") # input image 1
X2 = tf.placeholder(tf.float32, [None, data_x_size, data_y_size, 3], name="input2") # input image 2
Y = tf.placeholder(tf.float32, [None, 2]) # output : different = [1 0] / same = [0 1]
print(Y)
#----------------------------------------------------- conv net
with tf.variable_scope("image_filter_1") as scope:
    L1_1 = ConvNet(X1, 3, 32, batch_prob)
    scope.reuse_variables()
    L1_2 = ConvNet(X2, 3, 32, batch_prob)
    L1_1 = tf.nn.max_pool(L1_1, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
    L1_2 = tf.nn.max_pool(L1_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
    print(L1_1,'\n',L1_2)

with tf.variable_scope("image_filter_2") as scope:
    L2_1 = inception2d_v2(L1_1, 32, [10, 70, 4, 4], batch_prob)
    scope.reuse_variables()
    L2_2 = inception2d_v2(L1_2, 32, [10, 70, 4, 4], batch_prob)
    L2_1 = tf.nn.max_pool(L2_1, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
    L2_2 = tf.nn.max_pool(L2_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
    print(L2_1,'\n',L2_2)

with tf.variable_scope("image_filter_3") as scope:
    L3_1 = inception2d_v2(L2_1, 88, [20, 90, 8, 8], batch_prob)
    scope.reuse_variables()
    L3_2 = inception2d_v2(L2_2, 88, [20, 90, 8, 8], batch_prob)
    L3_1 = tf.nn.max_pool(L3_1, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
    L3_2 = tf.nn.max_pool(L3_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
print(L3_1,'\n',L3_2)

with tf.variable_scope("image_filter_4") as scope:
    L4_1 = inception2d_v2(L3_1, 126, [30, 100, 12, 12], batch_prob)
    scope.reuse_variables()
    L4_2 = inception2d_v2(L3_2, 126, [30, 100, 12, 12], batch_prob)
    L4_1 = tf.nn.max_pool(L4_1, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
    L4_2 = tf.nn.max_pool(L4_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
    print(L4_1,'\n',L4_2)

with tf.variable_scope("SP") as scope:
    L4_1 = Seperator_v2_1(L4_1,[1.5, 1.0, 1.5],3)
    L4_2 = Seperator_v2_1(L4_2,[1.5, 1.0, 1.5],3)
    print(L4_1,'\n',L4_2)
    
#----------------------------------------------------- cross neighborhood difference
with tf.variable_scope("CND") as scope:
    L4_1 = tf.transpose(L4_1,[0,3,1,2])
    L4_2 = tf.transpose(L4_2,[0,3,1,2])
    L4 = CrossND_ver2(L4_1,L4_2)
    L4 = tf.transpose(L4,[0,2,3,1])
    L4 = tf.nn.relu(L4)
    print(L4)

#----------------------------------------------------- conv net
with tf.variable_scope("Conv_Net_1") as scope:
    L5 = ConvNet(L4, 154, 154, batch_prob, stride = 3)
    print(L5)

#----------------------------------------------------- conv net
with tf.variable_scope("Conv_Net_2") as scope:
    L5 = ConvNet(L5, 154, 154, batch_prob)
    L5 = tf.nn.avg_pool(L5, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
    print(L5)
    
#----------------------------------------------------- dense
with tf.variable_scope("Dense") as scope:
    model = FC_layer_v2(L5, 5*2*154, batch_prob)
    model = tf.identity(model, "model")
    print(model)


batch_size=100
total_batch=int(len(train_data1) / batch_size)
print("전체 데이터 크기는",len(train_data1),"배치 크기는",batch_size,"전체 배치는",total_batch)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=model))

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.RMSPropOptimizer(0.00006).minimize(cost)

# tf.train.Saver를 이용해서 모델과 파라미터를 저장합니다.
SAVER_DIR = "Re_Id_TEST/New/sp3_top_mid"
saver = tf.train.Saver()
checkpoint_path = os.path.join(SAVER_DIR, "model")
ckpt = tf.train.get_checkpoint_state(SAVER_DIR)
    
init = tf.global_variables_initializer()
sess = tf.Session()

if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(init)

for epoch in range(32):
    total_cost=0
    index_num=0
    for i in range(total_batch):
        
        input_data1 = np.empty((batch_size, data_x_size, data_y_size, 3))
        input_data2 = np.empty((batch_size, data_x_size, data_y_size, 3))
        batch_i=0
        for index_num in data_index[i*batch_size:(i+1)*batch_size]:
            input_data1[batch_i,:,:,:] = cv2.resize(cv2.imread(train_dir+'/'+train_list[train_data1[index_num]],cv2.IMREAD_COLOR),(data_y_size,data_x_size))
            input_data2[batch_i,:,:,:] = cv2.resize(cv2.imread(train_dir+'/'+train_list[train_data2[index_num]],cv2.IMREAD_COLOR),(data_y_size,data_x_size))
            batch_i = batch_i+1
        batch_ys = train_CoN[i*batch_size : (i+1)*batch_size]
        _, cost_val=sess.run([optimizer, cost], feed_dict={X1:input_data1, X2:input_data2, Y:batch_ys, batch_prob : True})
        total_cost += cost_val
    saver.save(sess, checkpoint_path, global_step=epoch+1)
    print('Epoch:', '%04d' % (epoch + 1), 'Avg.cost=', '{:.3f}'.format(total_cost / total_batch))
    if (epoch+1)%5 == 0:
        test_size=100
        is_correct=tf.equal(tf.argmax(tf.nn.softmax(model),1),tf.argmax(Y,1))
        accuracy=tf.reduce_mean(tf.cast(is_correct, tf.float32))
        test_img1 = np.empty((test_size, data_x_size, data_y_size,3))
        test_img2 = np.empty((test_size, data_x_size, data_y_size,3))
        for i in range(test_size):
            test_img1[i,:,:,:] = cv2.resize(cv2.imread(test_dir+'/'+test_list[test_data1[test_data_index[i]]],cv2.IMREAD_COLOR),(data_y_size,data_x_size))
            test_img2[i,:,:,:] = cv2.resize(cv2.imread(test_dir+'/'+test_list[test_data2[test_data_index[i]]],cv2.IMREAD_COLOR),(data_y_size,data_x_size))

        ac = sess.run(accuracy, feed_dict={X1:test_img1,
                                            X2:test_img2,
                                            Y:test_CoN[test_data_index[:test_size]],
                                            batch_prob : False})
        print('정확도:', ac)


print('Optimized!')

















