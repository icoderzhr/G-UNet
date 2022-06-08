# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 10:48:27 2020

@author: zoe
"""

import os 
import numpy as np
import cv2
import os.path
#import matplotlib.pyplot as plt
import skimage
#import skimage.measure
import scipy.io as scio

imag_rows=128
imag_cols=128

def create_train_data(imag_rows, imag_cols):
    train_data_path = os.path.join(os.getcwd(),'./DataSet_mat/wrap-phase')
    images = os.listdir(train_data_path) #按某种排列顺序列出所有train_data_path中数据的名字
    imgs=[]
    #循环训练数据集中的所有数据
    for image_name in images:
        img = scio.loadmat(os.path.join(train_data_path,image_name))['wrap_img']   #加载.mat文件 ，img 是一个128*128*1的三维数组
        imgs.append(img)
    np.asarray(imgs)  #多维数组
    np.save('x_train.npy',imgs)
    print('Data save to .npy files done...')


    #--------------------train_label----------------------
    label_path = os.path.join(os.getcwd(),'./DataSet_mat/train_label')
    imgs_label = []
    for label_name in os.listdir(label_path):
        label=scio.loadmat(os.path.join(label_path,label_name))['label'] 
        imgs_label.append(label)
    np.asarray(imgs_label)    
    np.save('y_train.npy',imgs_label)
    print('Label save to .npy files done...')     

def load_train_data():
       x_train = np.load('x_train.npy')
       y_trian = np.load('y_train.npy')
       return x_train ,y_trian


def preprocess(imgs, img_rows,img_cols):
    imgs_p = np.ndarray((imgs.shape[0],imgs.shape[1],img_rows,img_cols),dtype=np.uint8)#一个4维数组，数据类型为8位无符号整形
    for i in range(imgs.shape[0]):
        imgs_p[i, 0 ] = cv2.resize(imgs[i,0],(img_cols,img_rows),interpolation=cv2.INTER_CUBIC)
    return imgs_p   

def create_test_data(imag_rows, imag_cols):
    test_data_path = os.path.join(os.getcwd(), './DataSet_mat/test_data')#获得test data的路径
    images = os.listdir(test_data_path)#列出test data的路径文件夹下的所有文件
    total = len(images)#统计文件夹下的文件个数，即测试集个数
    #imgs = np.array((total, imag_rows, imag_cols), dtype=np.uint8)
    imgs = np.array((total,imag_rows, imag_cols,1), dtype=np.uint8)#定义一个四维数组
    #imgs_id = np.ndarray((total, ), dtype=np.int32)

    i = 0  
    imgs=[]
    for image_name in images: #循环训练数据集中的所有数据
        img = scio.loadmat(os.path.join(test_data_path, image_name))['wrap_img']
        img = np.array([img])
        #img = img.reshape(128, 128)
        img=img.reshape(128,128,1)
        imgs.append(img)
        if i % 10 == 0:
            print('Done: {0}/{1} images'.format(i, total)) #打印图片加载进度
        i += 1
    print('Loading done.')
    np.save('x_test.npy', imgs)
    print('Saving to .npy files done.')
#-------------------------test_lable----------------------------------

def load_test_data():
    x_test = np.load('x_test.npy')
    #y_test= np.load('y_test.npy')
    #imgs_id = np.load('imgs_id_test.npy')
    return x_test


if __name__ == '__main__':
    create_train_data(imag_rows, imag_cols)
    create_test_data(imag_rows, imag_cols)

            
    