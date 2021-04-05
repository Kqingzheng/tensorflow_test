from keras.preprocessing import image
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os, shutil
# 加载训练好的模型
model = load_model('model.hdf5')
# 从测试集中建立对应文件夹来读取数据
original_dataset_dir = 'test1'
base_dir = 'cats_and_dogs_small_test'
if not os.path.exists(base_dir):
    os.makedirs(base_dir)
# Directory with our training cat pictures(猫的测试图像目录)
train_cats_dir = os.path.join(base_dir, 'cats')
if not os.path.exists(train_cats_dir):
    os.mkdir(train_cats_dir)

# Directory with our training dog pictures(狗的测试图像目录)
train_dogs_dir = os.path.join(base_dir, 'dogs')
if not os.path.exists(train_dogs_dir):
    os.mkdir(train_dogs_dir)


fnames = ['{}.jpg'.format(i) for i in range(1, 101)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    # dst = os.path.join(train_cats_dir, fname)
    # shutil.copyfile(src, dst)
    # 查看单张图片结果(load_img中路径名称需修改)
    path = f"./test1/{fname}"
    img = image.load_img(path, target_size=(150, 150))
    # 显示图片
    # npimg=np.array(img)
    # plt.imshow(npimg)
    # plt.show()


    x = image.img_to_array(img) / 255.
    x = x.reshape((1,) + x.shape)


    xx = model.predict_classes(x)
    yy = model.predict(x)

    if xx[0][0]==0:
        print(f"{fname}识别结果为：猫")
        dst = os.path.join(train_cats_dir, fname)
        shutil.copyfile(src, dst)
    else:
        print(f"{fname}识别结果为：狗")
        dst = os.path.join(train_dogs_dir, fname)
        shutil.copyfile(src, dst)
