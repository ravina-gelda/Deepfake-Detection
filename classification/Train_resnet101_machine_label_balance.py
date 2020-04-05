import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
from os.path import join
import cv2
import dlib
import csv
import json
import seaborn as sns
from PIL import Image as pil_image
from PIL import Image
from tqdm import tqdm
import pandas as pd
import numpy as np
import detect_from_video
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import InputLayer, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from collections import Counter
input_shape = (64, 64, 3)

train_dir_file = '../augmented_image_file_label/'
train_dir_machine = '../augmented_image_machine_label/'
#sess = tf.compat.v1.Session(config=tf.ConfigProto(device_count = {'GPU': 0}))
original_data = [f for f in os.listdir('../processed_image') if f.endswith('.png')]
data_file = [f for f in os.listdir(train_dir_file) if f.endswith('.png')]
data_machine = [f for f in os.listdir(train_dir_machine) if f.endswith('.png')]
X = []

true_label = pd.read_csv('final_true_label.csv')
y_true_label = true_label['File_Label']
generated_label = pd.read_csv('final_machine_label.csv')
y_generated_label = generated_label['Machine_Label']
a1 = dict(Counter(y_true_label))
a2 = dict(Counter(y_generated_label))
print(a1)
print(a2)
true_size = a1[0] * 2
machine_size = a2[0] * 2
print(true_size)
print(machine_size)
y_true_label = y_true_label[:true_size]
print(len(y_true_label))
y_generated_label = y_generated_label[:machine_size]
#for idx in range(1, 35476):
#    im = Image.open('../processed_image/'+'_'+str(idx)+'.png')
#
#    im = im.resize((64, 64))
#    im = np.array(im)
#    cv2.imwrite('../processed_little_image/'+'_'+str(idx)+'.png', im)
count = 0
for img in data_machine:
    count += 1
    if count < machine_size+1:
        X.append(img_to_array(load_img(train_dir_machine+img)).flatten() / 255.0)
X = np.array(X)
print(X.shape)
X = X.reshape(-1, 64, 64, 3)
print()
y_original_generated_label = y_generated_label
y_generated_label = to_categorical(y_generated_label, 2)
#Train-Test split
X_train, X_val, Y_train, Y_val = train_test_split(X, y_generated_label, test_size = 0.2,shuffle=True, random_state=5)

print(X_train.shape)
print(X_val.shape)
print(Y_train.shape)
print(Y_val.shape)

image_size = 64
from keras.applications.resnet import ResNet101
ResNet101_conv = ResNet101(include_top=False, weights='imagenet', input_shape=(image_size, image_size, 3))
for layer in ResNet101_conv.layers[:-4]:
    layer.trainable = False
for layer in ResNet101_conv.layers:
    print(layer, layer.trainable)
from keras import models
from keras import layers
from keras import optimizers
model = models.Sequential()
# Add the vgg convolutional base model
model.add(ResNet101_conv)
# Add new layers
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(2, activation='softmax'))
model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

model.summary()
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
EPOCHS = 20
BATCH_SIZE = 256
history1 = model.fit(X_train, Y_train, batch_size = BATCH_SIZE, epochs = EPOCHS, validation_data = (X_val, Y_val),  callbacks = [early_stopping], verbose = 1)

acc_train = history1.history['acc']
loss_train = history1.history['loss']
acc_val = history1.history['val_acc']
loss_val = history1.history['val_loss']
writer = csv.writer(open('results/Resnet101_train_with_generated_label.csv','w'))
writer.writerow(['Epoch', 'Train_Acc','Val_Acc', 'Train_Loss', 'Val_Loss'])
for i in range(len(acc_val)):
    writer.writerow([i, acc_train[i], acc_val[i], loss_train[i], loss_val[i]])

model.save('trained_models/Resnet101_train_with_generated_label.h5')


