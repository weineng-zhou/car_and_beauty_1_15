# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 10:31:34 2018

@author: weineng.zhou
"""

import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
print(tf.__version__)

currentroot = os.getcwd()

###############################################################################
# 汽车图片
filedir = currentroot + '/data/cars'

file_list1 = [] 
for root, dirs, files in os.walk(filedir): 
    for file in files: 
        if os.path.splitext(file)[1] == ".jpg":
            file_list1.append(os.path.join(root, file))

if not os.path.exists(currentroot + '/data/cars/cars_128'):
    os.makedirs(currentroot + '/data/cars/cars_128')
    
#批量改变图片像素
for filepath in file_list1:
    try:
        im = Image.open(filepath) 
        new_im =im.resize((128, 128))
        new_im.save(currentroot + '/data/cars/cars_128/' + filepath[filepath.rfind('\\')+1:])
        print('图片' + filepath[filepath.rfind('\\')+1:] + '像素转换完成')
    except OSError as e:
        print(e.args)
        
# 重新建立新图像列表
filedir = currentroot + '/data/cars/cars_128'

file_list_1=[] 
for root, dirs, files in os.walk(filedir): 
    for file in files: 
        if os.path.splitext(file)[1] == ".jpg":
            file_list_1.append(os.path.join(root, file))
                     
len(file_list_1)
###############################################################################

# 美女图片
filedir = currentroot + '/data/beauty'
os.listdir(filedir)

file_list2 = [] 
for root, dirs, files in os.walk(filedir): 
    for file in files: 
        if os.path.splitext(file)[1] == ".jpg":
            file_list2.append(os.path.join(root, file))

if not os.path.exists(currentroot + '/data/beauty/beauty_128'):
    os.makedirs(currentroot + '/data/beauty/beauty_128')
    
#批量改变图片像素
for filepath in file_list2:
    try:
        im = Image.open(filepath) 
        new_im =im.resize((128, 128))
        new_im.save(currentroot + '/data/beauty/beauty_128/' + filepath[filepath.rfind('\\')+1:])
        print('图片' + filepath[filepath.rfind('\\')+1:] + '像素转换完成')
    except OSError as e:
        print(e.args)
        
# 重新建立新图像列表
filedir = currentroot + '/data/beauty/beauty_128'
file_list_2 = [] 
for root, dirs, files in os.walk(filedir): 
    for file in files: 
        if os.path.splitext(file)[1] == ".jpg":
            file_list_2.append(os.path.join(root, file))
                     
len(file_list_2)

##############################################################################  
#合并列表          
len(file_list_1)
len(file_list_2)
file_list_all = file_list_1 + file_list_2
len(file_list_all)

##############################################################################            

M = []
for filename in file_list_all:
    im = Image.open(filename)
    width,height = im.size
    im_L = im.convert("L") 
    Core = im_L.getdata()
    arr1 = np.array(Core,dtype='float32') / 255.0
    arr1.shape
    list_img = arr1.tolist()
    M.extend(list_img)

X = np.array(M).reshape(len(file_list_all),width,height)
X.shape


class_names = ['汽车', '美女']
               
#用字典储存图像信息
dict_label = {0:'汽车', 1:'美女'}


#用列表输入标签，0表示汽车，1表示美女
label = [0]*len(file_list_1) + [1]*len(file_list_2)
y = np.array(label)

#按照4:1的比例将数据划分训练集和测试集
train_images, test_images, train_labels, test_labels = train_test_split(
        X, y, test_size=0.2, random_state=0)

###############################################################################
plt.figure()
plt.imshow(train_images[10])
plt.colorbar()
plt.grid(False)

###############################################################################
#显示来自训练集的前25个图像，并在每个图像下面显示类名。
#验证数据的格式是否正确，准备构建神经网络
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
        
###############################################################################
#第一个输入层有128个节点(或神经元)。
#第二个(也是最后一个)层是2个节点的softmax层————返回一个2个概率分数的数组，其和为1。
#每个节点包含一个分数，表示当前图像属于两个类别的概率
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(128, 128)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])
###############################################################################
model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

###############################################################################
# prediction
predictions = model.predict(test_images)
predictions[0]
np.argmax(predictions[0])
dict_label[np.argmax(predictions[0])]

###############################################################################
# 定义画图函数
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = '#00bc57'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)
###############################################################################
def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(len(class_names)), predictions_array,
                     color="#FF7F0E", width=0.2)
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
 
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('#00bc57')
###############################################################################

          
# 让我们看看第1张图片，预测标签和真实标签
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)


###############################################################################

# 让我们看看第12张图片，预测标签和真实标签
i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)


###############################################################################
#绘制预测标签和真实标签以及预测概率柱状图
#正确的预测用绿色表示，错误的预测用红色表示
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
  
###############################################################################
  
#最后，利用训练后的模型对单个图像进行预测。
#从测试数据集中获取第15个图像
img = test_images[14]
plt.imshow(img, cmap=plt.cm.binary)
###############################################################################
# 将图像添加到唯一的成员批处理中.
img = (np.expand_dims(img,0))
print(img.shape)
# 预测图像:
predictions_single = model.predict(img)
print(predictions_single)
###############################################################################
plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(2), class_names, rotation=45)
###############################################################################
np.argmax(predictions_single[0])
dict_label[np.argmax(predictions_single[0])]

###############################################################################

#从外部获取未知图像
filedir = currentroot + '/data/pred'
file_list_pred = [] 
for root, dirs, files in os.walk(filedir): 
    for file in files: 
        if os.path.splitext(file)[1] == ".jpg":
            file_list_pred.append(os.path.join(root, file))

if not os.path.exists(currentroot + '/data/pred/pred_128'):
    os.makedirs(currentroot + '/data/pred/pred_128')
    
#批量改变未知图片像素
for filepath in file_list_pred:
    try:
        im = Image.open(filepath) 
        new_im =im.resize((128, 128))
        new_im.save(currentroot + '/data/pred/pred_128/' + filepath[filepath.rfind('\\')+1:])
        print('未知图片' + filepath +'像素转换完成')
    except OSError as e:
        print(e.args)
        

# 获取标准化图片列表
filedir = currentroot + '/data/pred/pred_128'
file_list_pred_128 = [] 
for root, dirs, files in os.walk(filedir): 
    for file in files: 
        if os.path.splitext(file)[1] == ".jpg":
            file_list_pred_128.append(os.path.join(root, file))
            
###############################################################################

# 对于一个图像  
im = Image.open(file_list_pred_128[0])
width,height = im.size
im_L= im.convert("L") 
Core = im_L.getdata()
arr1 = np.array(Core,dtype='float32')/255.0
arr1.shape
list_img = arr1.tolist()
img = np.array(list_img).reshape(width,height)
pred_labels = np.array([0])
print(img.shape)
plt.imshow(img, cmap=plt.cm.binary)
# 将图像添加到唯一成员的批处理文件中.
img = (np.expand_dims(img,0))
print(img.shape)


###############################################################################
# 预测图像概率:
predictions_single = model.predict(img)
print(predictions_single)
###############################################################################
plot_value_array(0, predictions_single, pred_labels)
_ = plt.xticks(range(2), class_names, rotation=45)
###############################################################################
np.argmax(predictions_single[0])
dict_label[np.argmax(predictions_single[0])]
###############################################################################

def img2label(filename):
    im = Image.open(filename)
    width,height = im.size
    im_L= im.convert("L") 
    Core = im_L.getdata()
    arr1 = np.array(Core,dtype='float32')/255.0
    list_img = arr1.tolist()
    img = np.array(list_img).reshape(width,height)
    img = (np.expand_dims(img,0))
    predictions_single = model.predict(img)
    return np.argmax(predictions_single[0])

# 得到多个图像的二分类字典编号
pred_labels = [img2label(filename) for filename in file_list_pred_128]

# 字典编号翻译成对应的标签
for i, num in enumerate(pred_labels):
    print('第'+str(i+1)+'张图像识别为: '+dict_label[num])

# 可视化多个图像
M = []
for filename in file_list_pred_128:
    im = Image.open(filename)
    width,height = im.size
    im_L = im.convert("L") 
    Core = im_L.getdata()
    arr1 = np.array(Core,dtype='float')/255.0
    list_img = arr1.tolist()
    M.extend(list_img)
    
pred_images = np.array(M).reshape(len(file_list_pred_128),width,height)

plt.figure(figsize=(12,12))
for i, img in enumerate(pred_images):
    plt.subplot(len(file_list_pred_128)/2, len(file_list_pred_128)/2, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(img, cmap=plt.cm.binary)
    plt.xlabel(class_names[pred_labels[i]])
