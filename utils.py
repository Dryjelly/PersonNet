import os
import random
import numpy as np
import tensorflow as tf

# 학습에 필요한 데이터를 만드는 함수 (MPN : 부정 이미지 쌍을 만들때 배수로 증가시켜주는 변수)
def Image_Data(dir_path, MPN = 1):
    image_list = os.listdir(dir_path)
    image_list.sort()
    
    list_len = len(image_list)
    print("이미지 파일 총 갯수: ",list_len)
    
    image1 = []
    image2 = []
    CoN = [] # correct[0 1] or not[0 1]
    
    #--- 묶음 제작
    
    pick_num = 10 # 뽑을 묶음 수
    saved_name = image_list[0][:4] # ID가 이름의 4번째까지 이므로
    front = 0
    back = 0
    
    for index, img_name in enumerate(image_list):
        if img_name[:4] == saved_name and index+1 != list_len:
            continue
        saved_name = img_name[:4]
        back = index
        if back-front == 1:
            front = index
            continue
        for i in range(pick_num):
            D1,D2 = random.sample(range(front,back),2)
            image1.append(D1)
            image2.append(D2)
            CoN.append([0,1])
        for i in range(pick_num*MPN):
            D1 = random.sample(range(front,back),1)
            while(True):
                D2 = random.sample(range(0,list_len),1)
                if front>D2[0] or D2[0]>back: break
            image1.extend(D1)
            image2.extend(D2)
            CoN.append([1,0])
        front = index
        
    CoN=np.asarray(CoN)
    
    nsp, sp = np.sum(CoN,axis=0)
    print("Total: ",len(CoN))
    print("같은 사람: ",sp)
    print("다른 사람: ",nsp)
    
    return image_list, image1, image2, CoN

### For Testing ###
# testing_number 만큼의 사람들을 그 만큼 모두 짝지어 리턴해줍니다.

# Rank 시스템으로 테스트 하기위해 데이터를 만들어주는 함수 (testing_number : 학습에 사용할 총 사람의 수)
def ALL_Image_Data(dir_path, testing_number = 100):
    image_list = os.listdir(dir_path)
    image_list.sort()
    
    list_len = len(image_list)
    print("이미지 파일 총 갯수: ",list_len)
    print("뽑을 사람: ",testing_number)
    
    image1 = []
    image2 = []
    CoN = [] # correct[0 1] or not[0 1]
    
    
    ### testing_number 만큼의 사람이 어디까진지 파악하는 코드 ###
    saved_name = image_list[0][:4] # ID가 이름의 4번째까지 이므로
    number_of_people = 1
    
    for index, img_name in enumerate(image_list):
        if testing_number+1 == number_of_people: break
        if img_name[:4] == saved_name:
            continue
        saved_name = img_name[:4]
        number_of_people += 1
    list_len = index-1
    print(testing_number,"명 만큼의 사람의 사진은",list_len,"장 입니다.")
    ### -------------------------------------------------- ###
    
    saved_name = image_list[0][:4]
    front = 0
    back = 0
    
    for index, img_name in enumerate(image_list[:list_len]):
        
        if img_name[:4] == saved_name and index+1 != list_len:
            continue
        if index+1 == list_len: index = list_len
            
        saved_name = img_name[:4]
        back = index
        
        if back-front == 1:
            front = index
            continue
        
        image1.extend([front] * (list_len-1))
        image2.extend(list(range(0,front))+list(range(front+1,list_len)))
        for i in range(front):CoN.append([1,0])
        for i in range(back-front-1): CoN.append([0,1])
        for i in range(list_len-back): CoN.append([1,0])
        
        front = index
    
    CoN=np.asarray(CoN)
    return image_list, image1, image2, CoN

# 절대값 처리를 하지 않은 CND 함수
def CrossND_ver1 (input1, input2, stride = 3):
    # padding stride 미적용 일단 3으로 할 것
    L = len(input1.shape)
    if L==2: print("3차원 이상만 가능합니다.")
    elif L==3: paddings = tf.constant([[0, 0], [1, 1], [1, 1]])
    elif L==4: paddings = tf.constant([[0, 0], [0, 0], [1, 1], [1, 1]])
    else: print("미구현됨...")
        
    input2 = tf.pad(input2, paddings, "CONSTANT")
    for i in range(input1.shape[L-2]):
        if L==3: F1 = tf.subtract(tf.reshape(input1[:,i,0],[input1.shape[L-3],1,1]),input2[:,i:i+stride,:stride])
        elif L==4: F1 = tf.subtract(tf.reshape(input1[:,:,i,0],[-1,input1.shape[L-3],1,1]),input2[:,:,i:i+stride,:stride])
        for j in range(input1.shape[L-1]):
            if j == 0: continue
            if L==3: F2 = tf.subtract(tf.reshape(input1[:,i,j],[input1.shape[L-3],1,1]),input2[:,i:i+stride,j:j+stride])
            elif L==4: F2 = tf.subtract(tf.reshape(input1[:,:,i,j],[-1,input1.shape[L-3],1,1]),input2[:,:,i:i+stride,j:j+stride])
            F1 = tf.concat([F1,F2],L-1)
        if i==0:
            output = F1
            continue
        output = tf.concat([output,F1],L-2)
    return output

# 절대값 처리를 한 CND 함수
def CrossND_ver2 (input1, input2, stride = 3):
    # padding stride 미적용 일단 3으로 할 것
    L = len(input1.shape)
    if L==2: print("3차원 이상만 가능합니다.")
    elif L==3: paddings = tf.constant([[0, 0], [1, 1], [1, 1]])
    elif L==4: paddings = tf.constant([[0, 0], [0, 0], [1, 1], [1, 1]])
    else: print("미구현됨...")
        
    input2 = tf.pad(input2, paddings, "CONSTANT")
    for i in range(input1.shape[L-2]):
        if L==3: F1 = tf.subtract(tf.reshape(input1[:,i,0],[input1.shape[L-3],1,1]),input2[:,i:i+stride,:stride])
        elif L==4: F1 = tf.subtract(tf.reshape(input1[:,:,i,0],[-1,input1.shape[L-3],1,1]),input2[:,:,i:i+stride,:stride])
        F1 = tf.abs(F1)
        for j in range(input1.shape[L-1]):
            if j == 0: continue
            if L==3: F2 = tf.subtract(tf.reshape(input1[:,i,j],[input1.shape[L-3],1,1]),input2[:,i:i+stride,j:j+stride])
            elif L==4: F2 = tf.subtract(tf.reshape(input1[:,:,i,j],[-1,input1.shape[L-3],1,1]),input2[:,:,i:i+stride,j:j+stride])
            F2 = tf.abs(F2)
            F1 = tf.concat([F1,F2],L-1)
        if i==0:
            output = F1
            continue
        output = tf.concat([output,F1],L-2)
    return output

# 추가적인 거리 측정 방식을 도입한 CND 함수 (실패함)
def CrossND_ver3 (input1, input2, stride = 3):
    # padding stride 미적용 일단 3으로 할 것
    L = len(input1.shape)
    if L==2: print("3차원 이상만 가능합니다.")
    elif L==3: paddings = tf.constant([[0, 0], [1, 1], [1, 1]])
    elif L==4: paddings = tf.constant([[0, 0], [0, 0], [1, 1], [1, 1]])
    else: print("미구현됨...")
        
    input2 = tf.pad(input2, paddings, "CONSTANT")
    for i in range(input1.shape[L-2]):
        if L==3: F1 = tf.subtract(tf.reshape(input1[:,i,0],[input1.shape[L-3],1,1]),input2[:,i:i+stride,:stride])
        elif L==4: F1 = tf.subtract(tf.reshape(input1[:,:,i,0],[-1,input1.shape[L-3],1,1]),input2[:,:,i:i+stride,:stride])
        F1 = tf.square(F1)
        for j in range(input1.shape[L-1]):
            if j == 0: continue
            if L==3: F2 = tf.subtract(tf.reshape(input1[:,i,j],[input1.shape[L-3],1,1]),input2[:,i:i+stride,j:j+stride])
            elif L==4: F2 = tf.subtract(tf.reshape(input1[:,:,i,j],[-1,input1.shape[L-3],1,1]),input2[:,:,i:i+stride,j:j+stride])
            F2 = tf.square(F2)
            F1 = tf.concat([F1,F2],L-1)
        if i==0:
            output = F1
            continue
        output = tf.concat([output,F1],L-2)
    return output

# 초기 CNN 계층 (지금은 쓰이지 않음)
def ConvNet_ori(x, in_channels, filter_count, stride = 1,filter_size = 3, mu = 0.5, sigma = 0.01, padding = 'SAME'):
    W = tf.get_variable("W",[filter_size,filter_size,in_channels,filter_count],initializer=tf.random_normal_initializer(0, sigma))
    b = tf.get_variable("b",[1,1,filter_count],initializer=tf.random_normal_initializer(mu, sigma))
    x = tf.nn.conv2d(x,W,strides=[1,stride,stride,1],padding=padding)
    x = tf.add(x,b)
    return tf.nn.relu(x)

# CNN 계층
def ConvNet(x, in_channels, filter_count, batch_prob, stride = 1,filter_size = 3, mu = 0.5, sigma = 0.01, padding = 'SAME'):
    # W = tf.Variable(tf.random_normal([filter_size,filter_size,in_channels,filter_count], stddev = sigma))
    # b = tf.Variable(tf.random_normal([1,1,filter_count], mean = mu, stddev = sigma))
    W = tf.get_variable("W",[filter_size,filter_size,in_channels,filter_count],initializer=tf.random_normal_initializer(0, sigma))
    #print(W)
    #b = tf.get_variable("b",[1,1,filter_count],initializer=tf.random_normal_initializer(mu, sigma))
    x = tf.nn.conv2d(x,W,strides=[1,stride,stride,1],padding=padding)
    #x = tf.add(x,b)
    x = tf.layers.batch_normalization(x, center=True, scale=True, training=batch_prob, name = "batch_norm")
    return tf.nn.relu(x)

# inception 계층
def inception2d_v2(x, in_channels, filter_count, batch_prob, mu=0, sigma=0.01):
    
    # 1x1
    one_filter_1 = tf.get_variable("one_filter_1",[1, 1, in_channels, filter_count[0]],initializer=tf.initializers.truncated_normal(mu,sigma))
    one_by_one = tf.nn.conv2d(x, one_filter_1, strides=[1, 1, 1, 1], padding='SAME')
    one_by_one = tf.layers.batch_normalization(one_by_one, center=True, scale=True, training=batch_prob, name = "batch_norm1")
    
    # 3x3
    one_filter_3 = tf.get_variable("one_filter_3",[1, 1, in_channels, int(filter_count[1]*(2/3))],initializer=tf.initializers.truncated_normal(mu,sigma))
    three_filter = tf.get_variable("three_filter",[3, 3, int(filter_count[1]*(2/3)), filter_count[1]],initializer=tf.initializers.truncated_normal(mu,sigma))
    three_by_three = tf.nn.conv2d(x, one_filter_3, strides=[1, 1, 1, 1], padding='SAME')
    three_by_three = tf.nn.conv2d(three_by_three, three_filter, strides=[1, 1, 1, 1], padding='SAME')
    three_by_three = tf.layers.batch_normalization(three_by_three, center=True, scale=True, training=batch_prob, name = "batch_norm2")
    
    # 5x5
    one_filter_5 = tf.get_variable("one_filter_5",[1, 1, in_channels, int(filter_count[2]/2)],initializer=tf.initializers.truncated_normal(mu,sigma))
    five_filter = tf.get_variable("five_filter",[5, 5, int(filter_count[2]/2), filter_count[2]],initializer=tf.initializers.truncated_normal(mu,sigma))
    five_by_five = tf.nn.conv2d(x, one_filter_5, strides=[1, 1, 1, 1], padding='SAME')
    five_by_five = tf.nn.conv2d(five_by_five, five_filter, strides=[1, 1, 1, 1], padding='SAME')
    five_by_five = tf.layers.batch_normalization(five_by_five, center=True, scale=True, training=batch_prob, name = "batch_norm3")
    
    # avg pooling
    one_filter_p = tf.get_variable("one_filter_p",[1, 1, in_channels, filter_count[3]],initializer=tf.initializers.truncated_normal(mu,sigma))
    pooling = tf.nn.avg_pool(x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
    pooling_by_one = tf.nn.conv2d(pooling, one_filter_p, strides=[1, 1, 1, 1], padding='SAME')
    
    x = tf.concat([one_by_one, three_by_three, five_by_five, pooling_by_one], axis=3)  # Concat in the 4th dim to stack
    
    return tf.nn.relu(x)

# PCB계층(비슷한 영역을 찾아내는 계층)의 추가 (실패로 인해 쓰이지 않음)
def PCB(x, stripes):
    M = x.shape[1]
    N = x.shape[2]
    T = x.shape[3]
    
    W = tf.get_variable("PCB_W",[1,1,T,stripes],initializer=tf.random_normal_initializer)
    
    Pi_f = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    Pi_f = tf.nn.softmax(Pi_f)
    
    new_x = x * tf.reshape(Pi_f[:,:,:,0],[-1,M,N,1])
    for i in range(1,stripes):
        new_x = tf.concat([new_x, x * tf.reshape(Pi_f[:,:,:,i],[-1,M,N,1])], axis=3)
    
    return new_x

def PCB_v2(x, stripes, RPP = True):
    M = x.shape[1]
    N = x.shape[2]
    T = x.shape[3]
    
    if RPP:
        W = tf.get_variable("PCB_W",[1,1,T,stripes],initializer=tf.random_normal_initializer)
    
        Pi_f = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
        Pi_f = tf.nn.softmax(Pi_f)
    
        new_x = x * tf.reshape(Pi_f[:,:,:,0],[-1,M,N,1])
        for i in range(1,stripes):
            new_x = tf.concat([new_x, x * tf.reshape(Pi_f[:,:,:,i],[-1,M,N,1])], axis=3)
        new_x = tf.nn.relu(new_x)
    else:
        s = int(int(M)/stripes)
        new_x = x[:,:s,:,:]
        for i in range(1,stripes):
            new_x = tf.concat([new_x,x[:,i*s:(i+1)*s,:,:]], axis=3)
    
    return new_x

def PCB_v3(x, stripes, batch_prob,RPP = True):
    M = x.shape[1]
    N = x.shape[2]
    T = x.shape[3]
    
    s = int(int(M)/stripes)
    
    if RPP:
        W = tf.get_variable("PCB_W",[1,1,T,stripes],initializer=tf.random_normal_initializer)
    
        Pi_f = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
        Pi_f = tf.layers.batch_normalization(Pi_f, center=True, scale=True, training=batch_prob, name = "batch_norm_1")
        Pi_f = tf.nn.softmax(Pi_f)
    
        new_x = x[:,:s,:,:] * tf.reshape(Pi_f[:,:s,:,0],[-1,s,N,1])
        for i in range(1,stripes):
            new_x = tf.concat([new_x, x[:,i*s:(i+1)*s,:,:] * tf.reshape(Pi_f[:,i*s:(i+1)*s,:,i],[-1,s,N,1])], axis=3)
        
        new_x = tf.layers.batch_normalization(new_x, center=True, scale=True, training=batch_prob, name = "batch_norm_2")
        new_x = tf.nn.relu(new_x)
    else:
        new_x = x[:,:s,:,:]
        for i in range(1,stripes):
            new_x = tf.concat([new_x,x[:,i*s:(i+1)*s,:,:]], axis=3)
    
    return new_x

def PCB_v4(x1, x2, y, batch_prob):
    M = x1.shape[1]
    N = x1.shape[2]
    T = x1.shape[3]
    
    new_x1 = tf.reduce_mean(x1, axis = 2)
    new_x1 = tf.reduce_mean(new_x1, axis = 1)
    new_x1 = tf.reshape(new_x1, [-1,T])
    
    new_x2 = tf.reduce_mean(x2, axis = 2)
    new_x2 = tf.reduce_mean(new_x2, axis = 1)
    new_x2 = tf.reshape(new_x2, [-1,T])
    
    with tf.variable_scope("PCB") as scope:
        W = tf.get_variable("W",[T,128],initializer=tf.random_normal_initializer(0,0.2))
        b = tf.get_variable("b",[128],initializer=tf.random_normal_initializer(0.5,0.01))
        
        new_x1 = tf.add(tf.matmul(new_x1,W),b)
        new_x1 = tf.layers.batch_normalization(new_x1, center=True, scale=True, training=batch_prob, name = "batch_norm")
        new_x1 = tf.nn.sigmoid(new_x1)
        scope.reuse_variables()
        new_x2 = tf.add(tf.matmul(new_x2,W),b)
        new_x2 = tf.layers.batch_normalization(new_x2, center=True, scale=True, training=batch_prob, name = "batch_norm")
        new_x2 = tf.nn.sigmoid(new_x2)
        
        cost = tf.reduce_mean(tf.square(new_x1 - new_x2), axis = 1)
        cost = tf.reduce_mean(y[:,1] * cost - y[:,0] * tf.log(cost))
    
    return cost

def PCB_FC_part(x, batch_prob, output = 1):
    T = x.shape[-1]
    if len(x.shape) > 2:
        T = T * x.shape[1] * x.shape[2]
    
    W1 = tf.get_variable("W1",[T,100],initializer=tf.random_normal_initializer(0,0.2))
    b1 = tf.get_variable("b1",[100],initializer=tf.random_normal_initializer(0.5,0.01))
    W2 = tf.get_variable("W2",[100,output],initializer=tf.random_normal_initializer(0,0.2))
    b2 = tf.get_variable("b2",[output],initializer=tf.random_normal_initializer(0.5,0.01))
    
    x = tf.reshape(x, [-1,T])
    x = tf.add(tf.matmul(x,W1),b1)
    x = tf.layers.batch_normalization(x, center=True, scale=True, training=batch_prob, name = "batch_norm1")
    x = tf.nn.sigmoid(x)
    x = tf.add(tf.matmul(x,W2),b2)
    x = tf.layers.batch_normalization(x, center=True, scale=True, training=batch_prob, name = "batch_norm2")
    #x = tf.nn.sigmoid(x)
    
    return x

def PCB_v5(new_x, x2, batch_prob):
    M = x2.shape[1]
    N = x2.shape[2]
    T = x2.shape[3]
    
    with tf.variable_scope("PCB") as scope:
        new_x2 = tf.reshape(x2[:,0,0,:] * PCB_FC_part(new_x[:,0,0,:],batch_prob),[-1,1,1,T])
        for i in range(M):
            for j in range(N):
                if i==0 and j ==0: continue
                scope.reuse_variables()
                part = tf.reshape(x2[:,i,j,:] * PCB_FC_part(new_x[:,i,j,:],batch_prob),[-1,1,1,T])
                new_x2 = tf.concat([new_x2, part], axis = 1)
        new_x2 = tf.reshape(new_x2,[-1,M,N,T])
        
    return new_x2
    
# 단순하게 특징점들을 3부분으로 나누는 함수
def Seperator(x, stripes):
    M = int(x.shape[1])
    T = int(x.shape[3])
    Part = []
    
    for i in range(stripes):
        s = int(T/stripes)
        chunk = x[:,:,:,i*s:(i+1)*s]
        chunk = tf.reduce_mean(chunk, axis = 2)#, keepdims=True)
        chunk = tf.reduce_mean(chunk, axis = 1)#, keepdims=True)
        Part.append(chunk)
            
    return Part

# 3부분으로 나뉜 특징점에 특정 가중치를 곱하여 내보내는 함수
def Seperator_v2(x):
    M = int(x.shape[1])
    s = int(M/3)
    
    new_x = x[:,:s,:,:] * 1.25
    new_x = tf.concat([new_x, x[:,s:2*s,:,:] * 1.0], axis = 1)
    new_x = tf.concat([new_x, x[:,2*s:,:,:] * 0.75], axis = 1)
    
    return tf.nn.relu(new_x)

# 나누는 과정을 편하게 하기위해 더 많은 기능을 추가하였다. (w : 추가할 가중치 리스트, part : 나눌 횟수)
def Seperator_v2_1(x,w = [1.5, 1.25, 1.0, 0.75, 0.5],part = 5):
    M = int(x.shape[1])
    s = int(M/part)
    
    new_x = x[:,:s,:,:] * w[0]
    for i in range(1,part-1):
        new_x = tf.concat([new_x, x[:,i*s:(i+1)*s,:,:] * w[i]], axis = 1)
    new_x = tf.concat([new_x, x[:,(part-1)*s:,:,:] * w[part-1]], axis = 1)
    
    return tf.nn.relu(new_x)

# 신경망이 최적의 가중치를 찾아내길 원했으나 실패함
def Seperator_v3(x):
    M = int(x.shape[1])
    s = int(M/3)
    
    b1 = tf.get_variable("b1",[1],initializer=tf.random_normal_initializer(0.5,0.01))
    b2 = tf.get_variable("b2",[1],initializer=tf.random_normal_initializer(0.5,0.01))
    b3 = tf.get_variable("b3",[1],initializer=tf.random_normal_initializer(0.5,0.01))
    
    new_x = x[:,:s,:,:] * b1
    new_x = tf.concat([new_x, x[:,s:2*s,:,:] * b2], axis = 1)
    new_x = tf.concat([new_x, x[:,2*s:,:,:] * b3], axis = 1)
    
    return new_x

# 각 이미지들이 서로의 특징점에 영향을 받아 최적의 가중치를 찾는 함수 (실패함)
def Seperator_v4(x1, x2, part, batch_prob):
    M = int(x1.shape[1])
    s = int(M/part)
    r = int(M%part)
    
    dist_x = tf.abs(x1 - x2)
    dist_x = tf.reduce_mean(dist_x, axis=[1,2])
    prob = PCB_FC_part(dist_x, batch_prob, output = part)
    prob = tf.nn.softmax(prob) * part
    
    output = tf.reduce_mean(prob, axis = 0)
    
    new_x1 = x1[:,:s,:,:] * prob[:,None,None,None,0]
    new_x2 = x2[:,:s,:,:] * prob[:,None,None,None,0]
    for i in range(1,part):
        new_x1 = tf.concat([new_x1, x1[:,i*s:(i+1)*s,:,:] * prob[:,None,None,None,i]], axis = 1)
        new_x2 = tf.concat([new_x2, x2[:,i*s:(i+1)*s,:,:] * prob[:,None,None,None,i]], axis = 1)
    new_x1 = tf.concat([new_x1, x1[:,part*s:,:,:] * prob[:,None,None,None,part-1]], axis = 1)
    new_x2 = tf.concat([new_x2, x2[:,part*s:,:,:] * prob[:,None,None,None,part-1]], axis = 1)
    
    return tf.nn.relu(new_x1), tf.nn.relu(new_x2), output

# 위의 함수에서 좀 더 발전된 형태
def Seperator_v5(x1, x2, part, batch_prob):
    M = int(x1.shape[1])
    s = int(M/part)
    r = int(M%part)
    
    dist_x = tf.abs(x1 - x2)
    
    for i in range(s):
        dist_x_part = tf.reduce_mean(dist_x[:,i*s:(i+1)*s,:,:], axis=[1,2])
        prob = PCB_FC_part(dist_x_part, batch_prob, output = 1)
        prob = tf.nn.sigmoid(prob) * part
    
    new_x1 = x1[:,:s,:,:] * prob[:,None,None,None,0]
    new_x2 = x2[:,:s,:,:] * prob[:,None,None,None,0]
    for i in range(1,part):
        new_x1 = tf.concat([new_x1, x1[:,i*s:(i+1)*s,:,:] * prob[:,None,None,None,i]], axis = 1)
        new_x2 = tf.concat([new_x2, x2[:,i*s:(i+1)*s,:,:] * prob[:,None,None,None,i]], axis = 1)
    new_x1 = tf.concat([new_x1, x1[:,part*s:,:,:] * prob[:,None,None,None,part-1]], axis = 1)
    new_x2 = tf.concat([new_x2, x2[:,part*s:,:,:] * prob[:,None,None,None,part-1]], axis = 1)
    
    return new_x1, new_x2

# 4계층으로 이루어진 완전 연결 계층
def FC_layer_ori(x, input_size, output_size = 2):
    W1 = tf.Variable(tf.random_normal([input_size,4096], stddev = 0.2))
    W2 = tf.Variable(tf.random_normal([4096,4096], stddev = 0.2))
    W3 = tf.Variable(tf.random_normal([4096,512], stddev = 0.2))
    W4 = tf.Variable(tf.random_normal([512,output_size], stddev = 0.2))

    b1 = tf.Variable(tf.random_normal([4096], mean = 0.5, stddev = 0.01))
    b2 = tf.Variable(tf.random_normal([4096], mean = 0.5, stddev = 0.01))
    b3 = tf.Variable(tf.random_normal([512], mean = 0.5, stddev = 0.01))
    b4 = tf.Variable(tf.random_normal([output_size], mean = 0.5, stddev = 0.01))
    
    x = tf.reshape(x, [-1,input_size])
    x = tf.add(tf.matmul(x,W1),b1)
    x = tf.nn.sigmoid(x)

    x = tf.add(tf.matmul(x,W2),b2)
    x = tf.nn.sigmoid(x)

    x = tf.add(tf.matmul(x,W3),b3)
    x = tf.nn.sigmoid(x)

    x = tf.add(tf.matmul(x,W4),b4)
    return x

# 4계층으로 이루어진 완전 연결 계층
def FC_layer(x, input_size, batch_prob, output_size = 2):
    W1 = tf.Variable(tf.random_normal([input_size,4096], stddev = 0.2))
    W2 = tf.Variable(tf.random_normal([4096,4096], stddev = 0.2))
    W3 = tf.Variable(tf.random_normal([4096,512], stddev = 0.2))
    W4 = tf.Variable(tf.random_normal([512,output_size], stddev = 0.2))

    b1 = tf.Variable(tf.random_normal([4096], mean = 0.5, stddev = 0.01))
    b2 = tf.Variable(tf.random_normal([4096], mean = 0.5, stddev = 0.01))
    b3 = tf.Variable(tf.random_normal([512], mean = 0.5, stddev = 0.01))
    b4 = tf.Variable(tf.random_normal([output_size], mean = 0.5, stddev = 0.01))
    
    x = tf.reshape(x, [-1,input_size])
    x = tf.add(tf.matmul(x,W1),b1)

    x = tf.layers.batch_normalization(x, center=True, scale=True, training=batch_prob)
    x = tf.nn.sigmoid(x)
    #print(x)

    x = tf.add(tf.matmul(x,W2),b2)

    x = tf.layers.batch_normalization(x, center=True, scale=True, training=batch_prob)
    x = tf.nn.sigmoid(x)
    #print(x)

    x = tf.add(tf.matmul(x,W3),b3)

    x = tf.layers.batch_normalization(x, center=True, scale=True, training=batch_prob)
    x = tf.nn.sigmoid(x)
    #print(x)

    x = tf.add(tf.matmul(x,W4),b4)
    #print(x)
    return x

# 재사용을 위해 만든 완전연결 계층
def FC_layer_v2(x, in_channels, batch_prob, output_size = 2):
    # W1 = tf.Variable(tf.random_normal([in_channels,512], stddev = 0.2))
    W1 = tf.get_variable("W1",[in_channels,512],initializer=tf.random_normal_initializer(0,0.2))
    # W2 = tf.Variable(tf.random_normal([512,512], stddev = 0.2))
    W2 = tf.get_variable("W2",[512,512],initializer=tf.random_normal_initializer(0,0.2))
    # W3 = tf.Variable(tf.random_normal([512,output_size], stddev = 0.2))
    W3 = tf.get_variable("W3",[512,output_size],initializer=tf.random_normal_initializer(0,0.2))

    # b1 = tf.Variable(tf.random_normal([512], mean = 0.5, stddev = 0.01))
    b1 = tf.get_variable("b1",[512],initializer=tf.random_normal_initializer(0.5,0.01))
    #b2 = tf.Variable(tf.random_normal([512], mean = 0.5, stddev = 0.01))
    b2 = tf.get_variable("b2",[512],initializer=tf.random_normal_initializer(0.5,0.01))
    #b3 = tf.Variable(tf.random_normal([output_size], mean = 0.5, stddev = 0.01))
    b3 = tf.get_variable("b3",[output_size],initializer=tf.random_normal_initializer(0.5,0.01))
    
    x = tf.reshape(x, [-1,in_channels])
    x = tf.add(tf.matmul(x,W1),b1)

    x = tf.layers.batch_normalization(x, center=True, scale=True, training=batch_prob)
    x = tf.nn.sigmoid(x)
    #print(x)

    x = tf.add(tf.matmul(x,W2),b2)

    x = tf.layers.batch_normalization(x, center=True, scale=True, training=batch_prob)
    x = tf.nn.sigmoid(x)
    #print(x)

    x = tf.add(tf.matmul(x,W3),b3)

    x = tf.layers.batch_normalization(x, center=True, scale=True, training=batch_prob)
    #print(x)

    return x

# 1계층의 완전 연결계층
def FC_layer_v3(x, in_channels, batch_prob, output_size = 2):
    W1 = tf.get_variable("W",[in_channels,output_size],initializer=tf.random_normal_initializer(0,0.2))
    b1 = tf.get_variable("b",[output_size],initializer=tf.random_normal_initializer(0.5,0.01))
    
    x = tf.reshape(x, [-1,in_channels])
    x = tf.add(tf.matmul(x,W1),b1)

    x = tf.layers.batch_normalization(x, center=True, scale=True, training=batch_prob)
    return x

# 재 학습시 초기화 되지 않은 변수들을 초기화 시켜주는 함수
def initialize_uninitialized(sess):
    global_vars          = tf.global_variables()
    is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    print([str(i.name) for i in not_initialized_vars]) # only for testing
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))