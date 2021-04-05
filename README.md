# tensorflow_test


## 一个基于tensorflow、kears的卷积神经网络模型来区分猫狗图片，用猫狗各12500张图片来进行训练，最新训练的模型准确率达92%（该模型储存到model.h5中），后续继续进行改进


文件结构


  data

      cats_and_dogs_small
  
         test
          
              cats
          
              dogs
     
          train
              
              cats
              
              dogs
      
          validation
          
              cats
              
              dogs
      
  
  
      test1
  
  
      train
  
  
train是原始数据集，有猫和狗的图片各12500张，test1是测试数据集。用loaddata.py可以建立cats_and_dogs_small文件夹，
从原始数据train中生成训练数据集train，测试数据test，验证数据validation，测试数据集没用到可忽略。


kear_dc来进行模型训练，kear_test在test1取100张图片进行分类

卷积网络结构四次卷积四次池化两次全连接，结构如下
***

Layer (type)  |                 Output Shape       |         Param   
=================================================================
conv2d (Conv2D) |               (None, 148, 148, 32) |       896       
_________________________________________________________________
max_pooling2d (MaxPooling2D) |  (None, 74, 74, 32)    |      0         
_________________________________________________________________
conv2d_1 (Conv2D)         |     (None, 72, 72, 64)   |       18496     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 |  (None, 36, 36, 64)   |       0         
_________________________________________________________________
conv2d_2 (Conv2D)      |        (None, 34, 34, 128)   |      73856     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2  | (None, 17, 17, 128)     |    0         
_________________________________________________________________
conv2d_3 (Conv2D)        |      (None, 15, 15, 128)   |      147584    
_________________________________________________________________
max_pooling2d_3 (MaxPooling2  | (None, 7, 7, 128)    |       0         
_________________________________________________________________
flatten (Flatten)      |        (None, 6272)      |          0         
_________________________________________________________________
dropout (Dropout)      |        (None, 6272)        |        0         
_________________________________________________________________
dense (Dense)           |       (None, 512)          |       3211776   
_________________________________________________________________
dense_1 (Dense)         |       (None, 1)            |       513       
=================================================================
Total params: 3,453,121
Trainable params: 3,453,121
Non-trainable params: 0
_________________________________________________________________
***
