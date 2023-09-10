# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 18:57:16 2023

@author: Riley
"""
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.layers import Flatten, Dense, Concatenate, Reshape, LSTM
from tensorflow.keras.models import Sequential, Model
import scipy.io as sio
import tensorflow.keras
from tensorflow.keras.callbacks import EarlyStopping
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from tensorflow.keras import backend as K
import time
from sklearn.model_selection import StratifiedKFold
import scipy.signal as signal
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 不显示等级2以下的提示信息
import tensorflow as tf 
from  sklearn  import  preprocessing
import datetime
from tensorflow.python.keras.utils.vis_utils import plot_model

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import *
from tensorflow.keras.initializers import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.python.keras.layers import Layer



def smooth_labels(labels, factor=0.01):
    # smooth the labels
    labels *= (1 - factor)
    labels += (factor / labels.shape[1])

    # returned the smoothed labels
    return labels

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_s=tf.shape(source)[0]
    n_s=64 if n_s is None else n_s
    n_t= tf.shape(target)[0]
    n_t=64 if n_t is None else n_t
    n_samples =n_s+n_t
    total = tf.concat([source, target], axis=0)                                                      #   [None,n]
    total0 = tf.expand_dims(total,axis=0)               #   [1,b,n]
    total1 = tf.expand_dims(total,axis=1)               #   [b,1,n]
    L2_distance = tf.reduce_sum(((total0 - total1) ** 2),axis=2)     #   [b,b,n]=>[b,b]                                 #   [None,None,n]=>[128,128,1]
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = tf.reduce_sum(L2_distance) / tf.cast((n_samples ** 2 - n_samples),dtype=float)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [tf.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)  # /len(kernel_val)   #[b,b]

def MMD(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    n_s=tf.shape(source)[0]
    n_s=64 if n_s is None else n_s
    n_t= tf.shape(target)[0]
    n_t=64 if n_t is None else n_t
    XX = tf.reduce_sum(kernels[:n_s, :n_s])/tf.cast((n_s**2),dtype=float)
    YY = tf.reduce_sum(kernels[-n_t:, -n_t:])/tf.cast((n_t**2),dtype=float)
    XY = tf.reduce_sum(kernels[:n_s, -n_t:])/tf.cast((n_s*n_t),dtype=float)
    YX = tf.reduce_sum(kernels[-n_t:, :n_s])/tf.cast((n_s*n_t),dtype=float)
    loss = XX + YY - XY - YX
    return loss


num_classes = 2
batch_size = 128
img_rows, img_cols, num_chan = 8, 9, 4
flag = 'a'
t = 6

acc_list = []
std_list = []
all_acc = []
all1_acc = []
short_name = ['0']

# 45次实验分别进行10倍交叉验证，取平均
dataset_dir = "G:\\4情绪识别任务2\\4D-CRNN-master(4区)调整\\dreamer数据集\\with_base_0.5\\"
for i in range(len(short_name)):
    K.clear_session()
    print("\nprocessing: ", short_name[i], "......")
    file_path = os.path.join(dataset_dir, 'DE_num'+short_name[i])
    file = sio.loadmat(file_path)
    data = file['data'][:7452]
    y_v = file['valence_labels'][0][:7452]
    y_a = file['arousal_labels'][0][:7452]
    y_v = to_categorical(y_v, num_classes)
    y_a = to_categorical(y_a, num_classes)
    
    one_falx = data.transpose([0, 2, 3, 1])
    # one_falx = one_falx[:,:,:,2:4]
    one_falx = one_falx.reshape((-1, t, img_rows, img_cols, num_chan))
    one_y_v = np.empty([0,2])
    one_y_a = np.empty([0,2])
    for j in range(int(len(y_a)//t)):
        one_y_v = np.vstack((one_y_v, y_v[j*t]))
        one_y_a = np.vstack((one_y_a, y_a[j*t]))
    # print(one_y_v.shape)
    # print(one_y_a.shape)
    # print(one_falx.shape)

    if flag=='v':
        one_y = one_y_v
    else:
        one_y = one_y_a

    seed = 7
    np.random.seed(seed)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    # x_train, x_test, y_train, y_test = train_test_split(one_falx, one_y, test_size=0.1)
    cvscores = []
    start = time.time()
    # create model
    for train, test in kfold.split(one_falx, one_y.argmax(1)):
        img_size = (img_rows, img_cols, num_chan)

       # 建立网络共享层
        x1 = Conv2D(64, 5, activation = 'relu', padding = 'same', name= 'conv1')
        x2 = Conv2D(128, 4, activation = 'relu', padding = 'same', name = 'conv2')
        x3 = Conv2D(256, 4, activation = 'relu', padding = 'same', name = 'conv3')
        x4 = Conv2D(64, 1, activation = 'relu', padding = 'same', name = 'conv4')
        x5 = MaxPooling2D(2, 2)
        
        x6 = Flatten()
        x7 = Dense(512, activation = 'relu')
        x8 = Reshape((1, 512))


        input_1 = Input(shape = img_size)
        input_2 = Input(shape = img_size)
        input_3 = Input(shape = img_size)
        input_4 = Input(shape = img_size)
        input_5 = Input(shape = img_size)
        input_6 = Input(shape = img_size)


        base_network_1 = x8(x7(x6(x5(x4(x3(x2(x1(input_1))))))))
        base_network_2 = x8(x7(x6(x5(x4(x3(x2(x1(input_2))))))))
        base_network_3 = x8(x7(x6(x5(x4(x3(x2(x1(input_3))))))))
        base_network_4 = x8(x7(x6(x5(x4(x3(x2(x1(input_4))))))))
        base_network_5 = x8(x7(x6(x5(x4(x3(x2(x1(input_5))))))))
        base_network_6 = x8(x7(x6(x5(x4(x3(x2(x1(input_6))))))))

      
        
        #输入连接
        out_all1_1 = Concatenate(axis=1)([base_network_1, base_network_2, base_network_3])  #组合连接成shape=(None, 6, 512)
        out_all2_1 = Concatenate(axis=1)([base_network_4, base_network_5, base_network_6])  #组合连接成shape=(None, 6, 512)
        distance1=MMD(out_all1_1,out_all2_1)        
        out_all3_1 = Concatenate(axis = 1)(                            # 维度不变, 维度拼接，第一维度变为原来的6倍
                [base_network_1, base_network_2, base_network_3, base_network_4, base_network_5, base_network_6])

        # lstm layer
        lstm_layer1_1 = LSTM(128, name = 'lstm1')(out_all3_1)
        
        # dense layer
        out_layer1_1 = Dense(2, activation = 'softmax', name = 'out')(lstm_layer1_1)
        model1 = Model(inputs = [ input_1, input_2, input_3, input_4, input_5, input_6], outputs = out_layer1_1)  # 6个输入
        model1.add_loss(distance1)
    
        #输入连接
        out_all1_2 = Concatenate(axis=1)([base_network_1, base_network_2, base_network_6])  #组合连接成shape=(None, 6, 512)
        out_all2_2 = Concatenate(axis=1)([base_network_3, base_network_4, base_network_5])  #组合连接成shape=(None, 6, 512)
        distance2=MMD(out_all1_2,out_all2_2) 
        out_all3_2 = Concatenate(axis = 1)(                            # 维度不变, 维度拼接，第一维度变为原来的6倍
                [base_network_1, base_network_2, base_network_6, base_network_3, base_network_4, base_network_5])
        # lstm layer
        lstm_layer1_2 = LSTM(128, name = 'lstm1')(out_all3_2)
   
        # dense layer
        out_layer1_2 = Dense(2, activation = 'softmax', name = 'out')(lstm_layer1_2)
        model2 = Model(inputs = [ input_1, input_2, input_3, input_4, input_5, input_6], outputs = out_layer1_2)  # 6个输入
        model2.add_loss(distance2)         
        
        #输入连接
        out_all1_3 = Concatenate(axis=1)([base_network_6, base_network_5, base_network_4])  #组合连接成shape=(None, 6, 512)
        out_all2_3 = Concatenate(axis=1)([base_network_3, base_network_2, base_network_1])  #组合连接成shape=(None, 6, 512)
        distance3=MMD(out_all1_3,out_all2_3) 
        out_all3_3 = Concatenate(axis = 1)(                            # 维度不变, 维度拼接，第一维度变为原来的6倍
                [base_network_6, base_network_5, base_network_4, base_network_3, base_network_2, base_network_1])
        
   
        # lstm layer
        lstm_layer1_3 = LSTM(128, name = 'lstm1')(out_all3_3)

   
        # dense layer
        
        out_layer1_3 = Dense(2, activation = 'softmax', name = 'out')(lstm_layer1_3)
        model3 = Model(inputs = [ input_1, input_2, input_3, input_4, input_5, input_6], outputs = out_layer1_3)  # 6个输入
        model3.add_loss(distance3)        

        model1.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
                      optimizer=tensorflow.keras.optimizers.Adam(),
                      metrics=['accuracy'])

        model2.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
                      optimizer=tensorflow.keras.optimizers.Adam(),
                      metrics=['accuracy'])
        
        model3.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
                      optimizer=tensorflow.keras.optimizers.Adam(),
                      metrics=['accuracy'])
        
        
        MODEL_DIR = './DREAMER'+'nb'+str(i)
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        checkpoint_path1 = os.path.join(MODEL_DIR, "best_model1.h5")
        checkpoint_path2 = os.path.join(MODEL_DIR, "best_model2.h5")
        checkpoint_path3 = os.path.join(MODEL_DIR, "best_model3.h5")
        checkpoint_callback1 = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path1,           # 保存模型的路径
            save_best_only=True,             # 只保存最优的模型
            save_weights_only=False,         # 保存模型权重和结构
            monitor='val_accuracy',              # 监控的指标为验证集上的 loss
            verbose=0                        # 显示保存进度
            )
        checkpoint_callback2 = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path2,           # 保存模型的路径
            save_best_only=True,             # 只保存最优的模型
            save_weights_only=False,         # 保存模型权重和结构
            monitor='val_accuracy',              # 监控的指标为验证集上的 loss
            verbose=0                        # 显示保存进度
            )
        checkpoint_callback3 = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path3,           # 保存模型的路径
            save_best_only=True,             # 只保存最优的模型
            save_weights_only=False,         # 保存模型权重和结构
            monitor='val_accuracy',              # 监控的指标为验证集上的 loss
            verbose=0                         # 显示保存进度
            )
        for kkk in range(2):

            # Fit the model
            x_train1 = one_falx[train]   #(2703, 6, 8, 9, 5)
            y_train = one_y[train]      #(2703,3)
            
            x_test1 = one_falx[test]      #(675, 6, 8, 9, 5)
            y_test = one_y[test]         #(675, 3)
            x_test=[x_test1[:, 0], x_test1[:, 1], x_test1[:, 2], x_test1[:, 3], x_test1[:, 4], x_test1[:, 5]]       
            earlyStop = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=5, mode='max', verbose=1, restore_best_weights = True)
            model1.fit([x_train1[:, 0], x_train1[:, 1], x_train1[:, 2], x_train1[:, 3], x_train1[:, 4], x_train1[:, 5]], 
                          smooth_labels(y_train),callbacks=[checkpoint_callback1], epochs=50, batch_size=128, verbose=0,validation_data=(x_test, y_test))#取第二维度的6段

            # Fit the model
            x_train2 = one_falx[train] 
            y_train = one_y[train]      #(2703,3)    
            x_test2 = one_falx[test]
            y_test = one_y[test]
            x_test=[ x_test2[:, 0], x_test2[:, 1], x_test2[:, 2], x_test2[:, 3] ,x_test2[:, 4], x_test2[:, 5]]          
            model2.fit([x_train1[:, 0], x_train1[:, 1], x_train1[:, 2], x_train1[:, 3], x_train1[:, 4], x_train1[:, 5]], 
                          smooth_labels(y_train),callbacks=[checkpoint_callback2], epochs=50, batch_size=128, verbose=0,validation_data=(x_test, y_test))#取第二维度的6段


            x_train3 = one_falx[train]   #(2703, 6, 8, 9, 5)
            y_train = one_y[train]      #(2703,3)
            x_test3 = one_falx[test]      #(675, 6, 8, 9, 5)
            x_test=[ x_test3[:, 0], x_test3[:, 1], x_test3[:, 2], x_test3[:, 3],  x_test3[:, 4], x_test3[:, 5]]                  
            model3.fit([x_train1[:, 0], x_train1[:, 1], x_train1[:, 2], x_train1[:, 3], x_train1[:, 4], x_train1[:, 5]], 
                          smooth_labels(y_train),callbacks=[checkpoint_callback3], epochs=50, batch_size=128, verbose=0,validation_data=(x_test, y_test))#取第二维度的6段



        
        saved_model1=tf.keras.models.load_model(checkpoint_path1)
        saved_model2=tf.keras.models.load_model(checkpoint_path2)
        saved_model3=tf.keras.models.load_model(checkpoint_path3) 
               
        prescores1 = saved_model1.evaluate(x_test, y_test, verbose=0)
        prescores2 = saved_model2.evaluate(x_test, y_test, verbose=0)
        prescores3 = saved_model3.evaluate(x_test, y_test, verbose=0)
        
        # print("pre%.2f%%" % (prescores1[1] * 100)) # Accuracy  五次（五折）
        # print("pre%.2f%%" % (prescores2[1] * 100)) # Accuracy  五次（五折）
        # print("pre%.2f%%" % (prescores3[1] * 100)) # Accuracy  五次（五折）
        
        saved_model1.fit([x_train1[:, 0], x_train1[:, 1], x_train1[:, 2], x_train1[:, 3], x_train1[:, 4], x_train1[:, 5]], 
                          smooth_labels(y_train),callbacks=[earlyStop], epochs=50, batch_size=128, verbose=0,validation_data=(x_test, y_test))#取第二维度的6段
        saved_model2.fit([x_train1[:, 0], x_train1[:, 1], x_train1[:, 2], x_train1[:, 3], x_train1[:, 4], x_train1[:, 5]], 
                          smooth_labels(y_train),callbacks=[earlyStop], epochs=50, batch_size=128, verbose=0,validation_data=(x_test, y_test))#取第二维度的6段
        saved_model3.fit([x_train1[:, 0], x_train1[:, 1], x_train1[:, 2], x_train1[:, 3], x_train1[:, 4], x_train1[:, 5]], 
                          smooth_labels(y_train),callbacks=[earlyStop], epochs=50, batch_size=128, verbose=0,validation_data=(x_test, y_test))#取第二维度的6段
                             
            
        #evaluate the model
        scores1 = saved_model1.evaluate(x_test, y_test, verbose=0)
        scores2 = saved_model2.evaluate(x_test, y_test, verbose=0)
        scores3 = saved_model3.evaluate(x_test, y_test, verbose=0)
        # print("%.2f%%" % (scores1[1] * 100)) # Accuracy  五次（五折）
        # print("%.2f%%" % (scores2[1] * 100)) # Accuracy  五次（五折）
        # print("%.2f%%" % (scores3[1] * 100)) # Accuracy  五次（五折）
        # arr=[scores1[1],scores2[1],scores3[1],prescores1[1],prescores2[1],prescores3[1]]
        # _, score = max(enumerate(arr), key=lambda x: x[1])
        # arr1=[scores1[1],scores2[1],scores3[1]]
        # max_index, _ = max(enumerate(arr1), key=lambda x: x[1])
        # print("score",score)
        ddd1_=saved_model1([x_test1[0:330, 0], x_test1[0:330, 1], x_test1[0:330, 2], x_test1[0:330, 3], x_test1[0:330, 4], x_test1[0:330, 5]])  #修改
        ddd1__=saved_model1([x_test1[330:676, 0], x_test1[330:676, 1], x_test1[330:676, 2], x_test1[330:676, 3], x_test1[330:676, 4], x_test1[330:676, 5]])  #修改
        ddd1=tf.concat([ddd1_,ddd1__],0)
        ddd2_=saved_model2([x_test1[0:330, 0], x_test1[0:330, 1], x_test1[0:330, 2], x_test1[0:330, 3], x_test1[0:330, 4], x_test1[0:330, 5]])  #修改
        ddd2__=saved_model2([x_test1[330:676, 0], x_test1[330:676, 1], x_test1[330:676, 2], x_test1[330:676, 3], x_test1[330:676, 4], x_test1[330:676, 5]])  #修改
        ddd2=tf.concat([ddd2_,ddd2__],0)
        ddd3_=saved_model3([x_test1[0:330, 0], x_test1[0:330, 1], x_test1[0:330, 2], x_test1[0:330, 3], x_test1[0:330, 4], x_test1[0:330, 5]])  #修改
        ddd3__=saved_model3([x_test1[330:676, 0], x_test1[330:676, 1], x_test1[330:676, 2], x_test1[330:676, 3], x_test1[330:676, 4], x_test1[330:676, 5]])  #修改
        ddd3=tf.concat([ddd3_,ddd3__],0)        
        xuan1 = np.argmax(ddd1, axis=1)
        xuan2 = np.argmax(ddd2, axis=1)
        xuan3 = np.argmax(ddd3, axis=1)
        
        count=np.zeros([len(xuan1)])
        for kk in range(len(xuan1)):
            if xuan1[kk]==xuan2[kk]:
                count[kk]=xuan1[kk]
            elif xuan1[kk]==xuan3[kk]:
                count[kk]=xuan1[kk]
            elif xuan2[kk]==xuan3[kk]:
                count[kk]=xuan2[kk]
            else:
                count[kk]=xuan1[kk]
        
        temp = count - np.argmax(y_test, axis=1)
        score=(np.sum(count==np.argmax(y_test, axis=1)))/len(y_test)
        # # #--------------------用于混淆矩阵可注释-------------------------------------------------------

        
        # dd_name = f"ddd{max_index+1}" 
        # xuan = np.argmax(globals()[dd_name], axis=1)
        # value1=(np.sum(xuan==np.argmax(y_test, axis=1)))/len(y_test)
        # if score_hun is None:
        #     score_hun = value1
        #     np.save("G:\\4情绪识别任务2\\4D-CRNN-master(4区)调整\\SEED\\截稿程序6.11绘制混淆和RPF的数据\\"+'t'+str(t)+'nb'+str(nb)+state+"pred.npy",xuan)
        #     np.save("G:\\4情绪识别任务2\\4D-CRNN-master(4区)调整\\SEED\\截稿程序6.11绘制混淆和RPF的数据\\"+'t'+str(t)+'nb'+str(nb)+state+"test.npy",np.argmax(y_test, axis=1))
        # elif value1 > score_hun:
        #     score_hun = value1
        #     np.save("G:\\4情绪识别任务2\\4D-CRNN-master(4区)调整\\SEED\\截稿程序6.11绘制混淆和RPF的数据\\"+'t'+str(t)+'nb'+str(nb)+state+"pred.npy",xuan)
        #     np.save("G:\\4情绪识别任务2\\4D-CRNN-master(4区)调整\\SEED\\截稿程序6.11绘制混淆和RPF的数据\\"+'t'+str(t)+'nb'+str(nb)+state+"test.npy",np.argmax(y_test, axis=1))
        
        # # 如果当前值小于或等于之前保存的值，则忽略
        # else:
        #     pass
        #--------------------用于混淆矩阵可注释---------------------------------------------------
        
        all_acc.append(score * 100)
        all1_acc.append(score * 100)
    # print("all acc: {}".format(all_acc))
    print("mean acc: {}".format(np.mean(all_acc)))
    print("std acc: {}".format(np.std(all_acc)))
    acc_list.append(np.mean(all_acc))
    std_list.append(np.std(all_acc))
    print("进度： {}".format(short_name[i]))
    all_acc = []
    end = time.time()
    print("%.2f" % (end - start))
print('Acc_all: {}'.format(acc_list))
print('Std_all: {}'.format(std_list))
print('Acc_avg: {}'.format(np.mean(acc_list)))
print('Std_avg: {}'.format(np.mean(std_list)))