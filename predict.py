# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 15:41:29 2020

@author: zoe
"""
import os
import cv2
from keras.models import load_model
import math
import numpy as np
import scipy.io as scio


x_test = np.load ('F:\\Projects\\G_UNet\\x_test.npy')
y_test = np.load('F:\\Projects\\G_UNet\\y_test.npy')

# 评估Ghost_UNet模型
b = load_model("F:\\Projects\\G_UNet\\2020.5.25_Ghost_UNetmaster\\GU_nores_200_16_1e-3_64-1024.h5")
score =b.evaluate(x_test, y_test, verbose=0)
print('GU_score: ',score)
print('GU_loss:', score[0])
print('GU_acc:', score[1])

# 预测Ghost_UNet模型
prediction = b.predict(x_test)
print(prediction.shape)
#print('pred_data: ', prediction )
pre = np.array(prediction)
np.save('Ghost_UNet_nores_200_16_1e-3_64-10248_pred.npy',pre)

imgs = np.load('Ghost_UNet_nores_200_16_1e-3_64-1024_pred.npy') #predicted results
imgs_Num=imgs.shape[0] #照片中的总数量
for i in range(imgs_Num):
        save_name = os.path.join(os.getcwd(), 'F:\\Projects\\G_UNet\\2020.5.25_Ghost_UNetmaster\\Ghost_UNet_nores_200_16_1e-3_64-1024_pred\\%d_predict.mat'%(i+1))
        imgs_mask_test_save = imgs[i,:,:,:]
        scio.savemat(save_name,{'GU_pics':imgs_mask_test_save})



#计算pred和label差值
def compute_delta(img1,img2):
    delta = abs((img1/1.-img2/1.))
    return delta

#计算pred和label均方误差和信噪比
def compute_mse_psnr(img1, img2):
    mse = np.mean((img1 / 1. - img2 / 1.) ** 2)
    if mse < 1.0e-10:
        return 100 * 1.0
    psnr = 10 * math.log10(255.0 * 255.0 / mse)
    return mse,psnr

#计算pred和label均方误差
def compute_mse(img1, img2):
    mse = np.mean((img1 / 1. - img2 / 1.) ** 2)
    if mse < 1.0e-10:
        return 100 * 1.0
    return mse

#计算pred和label信噪比
def compute_psnr(img1, img2):
    mse = np.mean((img1 / 1. - img2 / 1.) ** 2)
    if mse < 1.0e-10:
        return 100 * 1.0
    psnr = 10 * math.log10(255.0 * 255.0 / mse)
    return psnr

#结构相似度处理函数
def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

#计算pred和label结构相似度
def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

# 计算delta, mse, psnr, ssim各指标的值
# 读取laplace G-UNet UNet labels文件
ghostunet_path = 'F:\\Projects\\G_UNet\\2020.5.25_Ghost_UNetmaster\\Ghost_UNet_nores_200_16_1e-3_64-1024_pred\\'

labels_path = 'F:\\Projects\\G_UNet\DataSet_mat\\test_label\\'
f_nums = len(os.listdir(labels_path))
print(f_nums)
GU_list_delta = []
GU_list_mse = []
GU_list_psnr= []
GU_list_ssim= []

for i in range(1, f_nums+1):
    pred_ghostunet = np.asarray(scio.loadmat(os.path.join(ghostunet_path,"%d_predict.mat"%i))['GU_pics'])
    labels= np.asarray(scio.loadmat(os.path.join(labels_path,"img_ (%d).mat"%i))['label'])
    labels = labels.reshape(128, 128, 1)

    GU_delta = compute_delta(pred_ghostunet, labels)
    GU_mse, GU_psnr = compute_mse_psnr(pred_ghostunet, labels)
    GU_ssim = calculate_ssim(pred_ghostunet, labels)
    GU_list_delta.append(GU_delta)
    GU_list_mse.append(GU_mse)
    GU_list_psnr.append(GU_psnr)
    GU_list_ssim.append(GU_ssim)

#保存Ghost_UNet预测值和真实值的delta为.mat文件

delta = np.array(GU_list_delta)
print(delta.shape)
np.save('Ghost_UNet_nores_200_16_1e-3_64-1024_delta_abs.npy',delta)

imgs = np.load('Ghost_UNet_nores_200_16_1e-3_64-1024_delta_abs.npy') #predicted results

imgs_Num=imgs.shape[0] #照片中的总数量
for i in range(imgs_Num):
        save_name = os.path.join(os.getcwd(), 'F:\\Projects\\G_UNet\\2020.5.25_Ghost_UNetmaster\\Ghost_UNet_nores_200_16_1e-3_64-1024_delta_abs\%d_delta.mat'%(i+1))
        imgs_mask_test_save = imgs[i,:,:,:]
        scio.savemat(save_name,{'GU_delta':imgs_mask_test_save})


#打印各个指标的输出值
print('GU_delta: ',GU_list_delta)
#print(len(GU_list_delta))
print('GU_mse: ',GU_list_mse)
#print(len(GU_list_mse))
print('GU_psnr: ',GU_list_psnr)
#print(len(GU_list_psnr))
print('GU_ssim: ',GU_list_ssim)
#print(len(GU_list_ssim))



#打印各个指标的平均值
print("GU_delta_mean:", np.mean(GU_delta))
print("GU_mse_mean:", np.mean(GU_list_mse))
print("GU_psnr_mean:", np.mean(GU_list_psnr))
print("GU_ssim_mean:", np.mean(GU_list_ssim))

