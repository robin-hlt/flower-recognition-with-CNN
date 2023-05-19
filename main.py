import glob
import os
import cv2
import keras
import tensorflow as tf
from tensorflow.keras import optimizers,layers
from tensorflow.keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
path = './flower_photos/'
w = 100
h = 100
c = 3
def read_img(path):
   cate=[path+x for x in os.listdir(path) if os.path.isdir(path+x)]
   imgs=[]
   labels=[]
   for idx,folder in enumerate(cate):
        for im in glob.glob(folder+'/*.jpg'):
           img=cv2.imread(im)
           img=cv2.resize(img,(w,h))
           imgs.append(img)
           labels.append(idx)
   return np.asarray(imgs,np.float32),np.asarray(labels,np.int32)
data,label=read_img(path)
print("shape of data:",data.shape)
print("shape of label:",label.shape)
seed = 785
np.random.seed(seed)
x_train, x_val, y_train, y_val = train_test_split(data, label, test_size=0.20,random_state=seed)
x_train = x_train / 255
x_val = x_val / 255
flower_dict = {0:'dasiy',1:'dandelion',2:'roses',3:'sunflowers',4:'tulips'}
#构建CNN模型
model = Sequential([
   layers.Conv2D(32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu),
   layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
   layers.Dropout(0.25),
   layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
   layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
   layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
   layers.Dropout(0.25),
   layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
   layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
   layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
   layers.Dropout(0.25),
   layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
   layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
   layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
   layers.Dropout(0.25),
   layers.Flatten(),
   layers.Dense(512, activation=tf.nn.relu),
   layers.Dense(256, activation=tf.nn.relu),
   layers.Dense(5, activation='softmax')
   ])
opt = optimizers.Adam(lr=0.0001)
model.compile(optimizer=opt,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
#训练模型
model.fit(x_train, y_train, epochs=20, validation_data=(x_val,y_val),batch_size=200, verbose=2)
model.summary()
#测试阶段
path = './TestImages/'
imgs=[] 
for im in glob.glob(path+'/*.jpg'):
#print('reading the images:%s'%(im))
# 遍历图像的同时，打印每张图片的“路径+名称”信息
    img=cv2.imread(im) 
    img=cv2.resize(img,(w,h))
    imgs.append(img)
imgs = np.asarray(imgs,np.float32)
print("shape of data:",imgs.shape)
#将图像导入模型进行预测
prediction = model.predict_classes(imgs)
#绘制预测图像
for i,t in enumerate(os.listdir(path)):
    print("第",i+1,"朵花预测:"+flower_dict[prediction[i]])
    img = plt.imread(path+t)#此处修改使图片与预测结果顺序对应输出
    plt.imshow(img)
