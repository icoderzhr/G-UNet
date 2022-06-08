from GUNet_model import Ghost_UNet
from keras import losses
from keras.optimizers import Adam
import numpy as np
from keras.callbacks import ModelCheckpoint
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import datetime
import os
import keras.backend as K

model = Ghost_UNet()
# = model.GhostNet()
model.compile(optimizer = Adam(lr = 1e-4), loss= losses.mean_squared_error, metrics = ['accuracy'])
#下载训练数据
imgs_wrap = np.load('F:\\Projects\\G_UNet\\x_train.npy')
imgs_unwrap = np.load('F:\\Projects\\G_UNet\\y_train.npy')

#下载测试数据
x_test = np.load ('F:\\Projects\\G_UNet\\x_test.npy')
y_test = np.load('F:\\Projects\\G_UNet\\y_test.npy')

def rand_indx(n, batch_size):
    return np.random.randint(0, n, size=batch_size)

def make_train_batch(imgs_wrap, imgs_unwrap, batch_size, img_col, img_row):
    imgs_num = imgs_wrap.shape[0]
    indx = rand_indx(imgs_num, batch_size)  # indx返回的一个随机整型数
    imgs_wrap = imgs_wrap
    x_train = imgs_wrap[indx]
    imgs_unwrap = imgs_unwrap[indx]
    x_train = x_train.reshape(batch_size, img_col, img_row, 1)
    imgs_unwrap = imgs_unwrap.reshape(batch_size, img_col, img_row, 1)
    return x_train, imgs_unwrap

# 计算训练时间函数
def subtime(date1, date2):
    date1 = datetime.datetime.strptime(date1, "%Y-%m-%d %H:%M:%S")
    date2 = datetime.datetime.strptime(date2, "%Y-%m-%d %H:%M:%S")
    return date2 - date1

# 训练函数
def train(epochs, batch_size):
    train_results = []
    train_psnr = []
    train_SSIM = []
    startdate = datetime.datetime.now() # 获取当前时间
    startdate = startdate.strftime("%Y-%m-%d %H:%M:%S")
    for epoch in range(epochs):
        wrap_batch, unwrap_batch = make_train_batch(imgs_wrap, imgs_unwrap, batch_size, img_col=128, img_row=128)
        batch_results = model.train_on_batch(wrap_batch, unwrap_batch)
        #psnr = compute_psnr(wrap_batch, unwrap_batch)
        train_results.append(batch_results)
        #train_psnr.append(psnr)
        #train_SSIM.append(ssim)
        if epoch % 100 == 0 and epoch != 0:
            lr = K.get_value(model.optimizer.lr)
            lr = lr * 0.1
            K.set_value(model.optimizer.lr,lr)
        if epoch % 10 == 0:
            print('train_epoch:  ', epoch)
            print('train_loss:  ', train_results[-1])#-1表示读取列表中倒数第一个元素
            #print('train_psnr: ', train_psnr[-1])
            #print('train_SSIM: ', train_SSIM[-1])
    enddate = datetime.datetime.now()  # 获取当前时间
    enddate = enddate.strftime("%Y-%m-%d %H:%M:%S")  # 当前时间转换为指定字符串格式

# 计算训练时长
    print('start date: ', startdate)
    print('end date: ', enddate)
    print('Time: ', subtime(startdate, enddate))  # enddate > startdate

# 保存模型和权重
    save_weight_dir = os.path.join(os.getcwd(), 'GU_weight_200_16_1e-4_64-1024.h5')
    model.save_weights(save_weight_dir)
    save_model_dir = os.path.join(os.getcwd(), 'GU_200_16_1e-4_64-1024.h5')
    model.save(save_model_dir)

# 模型评估
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

# loss-epochs图
    epochs = range(1, len(train_results) + 1)
    results = np.asarray(train_results)
    loss = results[:, 0]
    print('loss: ',loss)
    #print(len(loss))
    #print(len(train_results))
    #print(type(train_results))
    #print(type(loss))
    #print(train_results)
    plt.plot(epochs, loss, 'b', label = 'Ghost_UNet')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.figure()
# loss-epochs图
    acc = results[:, 1]
    print('acc: ',acc)
    plt.plot(epochs, acc, 'r', label='Ghost_UNet')
    plt.title('Training Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.legend()
    plt.show()






if __name__ == '__main__':
    train(200,16)
    #model_checkpoint = ModelCheckpoint('GU_200_16_1e-3_32-512.h5', monitor='loss', verbose=1, save_best_only=True)

