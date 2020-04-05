import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
original_data = [f for f in os.listdir('../processed_image') if f.endswith('.png')]
data_file = [f for f in os.listdir(train_dir_file) if f.endswith('.png')]
data_machine = [f for f in os.listdir(train_dir_machine) if f.endswith('.png')]
X = []
#whole_label = pd.read_csv('idx_label.csv')
#k = len(original_data)
#print(k)
#y_true_label = whole_label['File_Label']
#y_true_label = y_true_label.tolist()
#y_generated_label = whole_label['Machine_Label']
#y_generated_label = y_generated_label.tolist()
#
#for i in range(k, len(data_file)):
#    y_true_label.append(1)
#for i in range(k, len(data_machine)):
#    y_generated_label.append(1)
#
#writer1 = csv.writer(open('final_true_label.csv','w'))
#writer1.writerow(['Index','File_Label'])
#for i in range(len(y_true_label)):
#    writer1.writerow([i, y_true_label[i]])
#
#writer2 = csv.writer(open('final_machine_label.csv','w'))
#writer2.writerow(['Index','Machine_Label'])
#for i in range(len(y_generated_label)):
#    writer2.writerow([i, y_generated_label[i]])

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
for img in data_file:
    count += 1
    if count < true_size+1:
        X.append(img_to_array(load_img(train_dir_file+img)).flatten() / 255.0)
X = np.array(X)
print(X.shape)
X = X.reshape(-1, 64, 64, 3)
print()
y_original_true_label = y_true_label
y_true_label = to_categorical(y_true_label, 2)
#Train-Test split
X_train, X_val, Y_train, Y_val = train_test_split(X, y_true_label, test_size = 0.2,shuffle=True, random_state=5)

print(X_train.shape)
print(X_val.shape)
print(Y_train.shape)
print(Y_val.shape)
# Train on InceptionResnetV2
#googleNet_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=input_shape)
#googleNet_model.trainable = True
#model = Sequential()
#model.add(googleNet_model)
#model.add(GlobalAveragePooling2D())
#model.add(Dense(units=2, activation='softmax'))
#model.compile(loss='binary_crossentropy',
#              optimizer=optimizers.Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
#              metrics=['accuracy'])

image_size = 64
from keras.applications.resnet50 import ResNet50
resnet_conv = ResNet50(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
for layer in resnet_conv.layers[:-4]:
    layer.trainable = False
for layer in resnet_conv.layers:
    print(layer, layer.trainable)
from keras import models
from keras import layers
from keras import optimizers
model = models.Sequential()
# Add the vgg convolutional base model
model.add(resnet_conv)
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
writer = csv.writer(open('results/Resnet50_train_with_true_label.csv','w'))
writer.writerow(['Epoch', 'Train_Acc','Val_Acc', 'Train_Loss', 'Val_Loss'])
for i in range(len(acc_val)):
    writer.writerow([i, acc_train[i], acc_val[i], loss_train[i], loss_val[i]])

model.save('trained_models/Resnet50_train_with_true_label.h5')


