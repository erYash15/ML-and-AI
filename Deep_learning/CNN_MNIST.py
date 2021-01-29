# Credits: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py


from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

print(K.image_data_format())

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

print("Data Formating")

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 #padding = 'same',
                 input_shape=input_shape)) 

# input_shape = (28, 28, 1)
# kernel_size: 3*3=9
# training parameter: 1*32*9 + 32
model.add(Conv2D(64, (3, 3), activation='relu'))
# training parameter: 32*64*9 + 64
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
# training parameter: 10816 * 128 +128 = 1384576
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
# training parameter: 128 * 10 + 10 = 1290

model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


##########################################################
#=========================================================
##########################################################

# OUTPUT:

# Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
# 11493376/11490434 [==============================] - 0s 0us/step
# (60000, 28, 28)
# (60000,)
# (10000, 28, 28)
# (10000,)
# channels_last
# Data Formating
# (60000, 28, 28, 1)
# (60000,)
# (10000, 28, 28, 1)
# (10000,)
# x_train shape: (60000, 28, 28, 1)
# 60000 train samples
# 10000 test samples

# Model: "sequential_1"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# conv2d_2 (Conv2D)            (None, 26, 26, 32)        320       
# _________________________________________________________________
# conv2d_3 (Conv2D)            (None, 24, 24, 64)        18496     
# _________________________________________________________________
# max_pooling2d_1 (MaxPooling2 (None, 12, 12, 64)        0         
# _________________________________________________________________
# dropout_2 (Dropout)          (None, 12, 12, 64)        0         
# _________________________________________________________________
# flatten_1 (Flatten)          (None, 9216)              0         
# _________________________________________________________________
# dense_2 (Dense)              (None, 128)               1179776   
# _________________________________________________________________
# dropout_3 (Dropout)          (None, 128)               0         
# _________________________________________________________________
# dense_3 (Dense)              (None, 10)                1290      
# =================================================================
# Total params: 1,199,882
# Trainable params: 1,199,882
# Non-trainable params: 0
# _________________________________________________________________
# Epoch 1/12
# 469/469 [==============================] - 4s 8ms/step - loss: 2.2975 - accuracy: 0.1131 - val_loss: 2.2554 - val_accuracy: 0.3661
# Epoch 2/12
# 469/469 [==============================] - 4s 8ms/step - loss: 2.2510 - accuracy: 0.2169 - val_loss: 2.1978 - val_accuracy: 0.5325
# Epoch 3/12
# 469/469 [==============================] - 4s 8ms/step - loss: 2.1941 - accuracy: 0.3206 - val_loss: 2.1233 - val_accuracy: 0.6172
# Epoch 4/12
# 469/469 [==============================] - 4s 8ms/step - loss: 2.1208 - accuracy: 0.4102 - val_loss: 2.0229 - val_accuracy: 0.6744
# Epoch 5/12
# 469/469 [==============================] - 4s 8ms/step - loss: 2.0202 - accuracy: 0.4789 - val_loss: 1.8872 - val_accuracy: 0.7155
# Epoch 6/12
# 469/469 [==============================] - 4s 8ms/step - loss: 1.8870 - accuracy: 0.5305 - val_loss: 1.7114 - val_accuracy: 0.7551
# Epoch 7/12
# 469/469 [==============================] - 4s 8ms/step - loss: 1.7248 - accuracy: 0.5794 - val_loss: 1.5051 - val_accuracy: 0.7793
# Epoch 8/12
# 469/469 [==============================] - 4s 8ms/step - loss: 1.5529 - accuracy: 0.6162 - val_loss: 1.2936 - val_accuracy: 0.7985
# Epoch 9/12
# 469/469 [==============================] - 4s 8ms/step - loss: 1.3714 - accuracy: 0.6482 - val_loss: 1.1040 - val_accuracy: 0.8140
# Epoch 10/12
# 469/469 [==============================] - 4s 8ms/step - loss: 1.2248 - accuracy: 0.6737 - val_loss: 0.9526 - val_accuracy: 0.8245
# Epoch 11/12
# 469/469 [==============================] - 4s 8ms/step - loss: 1.1072 - accuracy: 0.6922 - val_loss: 0.8363 - val_accuracy: 0.8324
# Epoch 12/12
# 469/469 [==============================] - 4s 8ms/step - loss: 1.0154 - accuracy: 0.7060 - val_loss: 0.7503 - val_accuracy: 0.8399
# Test loss: 0.7503002285957336
# Test accuracy: 0.839900016784668

##########################################################
#=========================================================
##########################################################