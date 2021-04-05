from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras import models
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
import matplotlib.pyplot as plt

def model():
    model = models.Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    return model


#  训练数据增强
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   #  表示图像随机旋转的角度范围
                                   rotation_range=40,
                                   #  图像在水平方向上平移的范围
                                   width_shift_range=0.2,
                                   #  图像在垂直方向上平移的范围
                                   height_shift_range=0.2,
                                   #  随机错切变换的角度
                                   shear_range=0.2,
                                   #  图像随机缩放的范围
                                   zoom_range=0.2,
                                   #  随机将一半图像水平翻转
                                   horizontal_flip=True)
#  验证数据不能增强
validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_dir = r"./cats_and_dogs_small/train/"
validation_dir = r"./cats_and_dogs_small/validation/"

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    #  将所有图像的大小调整为150×150
                                                    target_size=(150, 150),
                                                    #  批量大小
                                                    batch_size=32,
                                                    class_mode='binary')
validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                              target_size=(150, 150),
                                                              batch_size=32,
                                                              class_mode='binary')



#  初始化模型
model = model()
#  用于配置训练模型（优化器、目标函数、模型评估标准）
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#  查看各个层的信息
model.summary()
#  回调函数，在每个训练期之后保存模型
model_checkpoint = ModelCheckpoint('model.hdf5',  # 保存模型的路径
                                   monitor='loss',  # 被监测的数据
                                   verbose=1,  # 日志显示模式:0=>安静模式,1=>进度条,2=>每轮一行
                                   save_best_only=True)  # 若为True,最佳模型就不会被覆盖
#  用history接收返回值用于画loss/acc曲线
history = model.fit_generator(train_generator,
                              steps_per_epoch=313,
                              epochs=100,
                              callbacks=[model_checkpoint],
                              validation_data=validation_generator,
                              validation_steps=50)

# print(history.history)
#
# print(history.epoch)
#
# print(history.history['val_loss'])
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()