# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 15:53:50 2021

@author: Lucas Gava
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



os.chdir('C:/Users/Lucas Gava/Desktop/IA/Possiveis Datas/DataMelanoma')
if os.path.isdir('train/AUG') is False:
    os.makedirs('train/AUG')
    os.makedirs('train/ISIC')
    os.makedirs('valid/AUG')
    os.makedirs('valid/ISIC')
    os.makedirs('test/AUG')
    os.makedirs('test/ISIC')
    
    for i in random.sample(glob.glob('ISIC*'), 500):
        shutil.move(i, 'train/ISIC')
    for i in random.sample(glob.glob('AUG*'), 500):
        shutil.move(i, 'train/AUG')
    for i in random.sample(glob.glob('ISIC*'), 100):
        shutil.move(i, 'valid/ISIC')
    for i in random.sample(glob.glob('AUG*'), 100):
        shutil.move(i, 'valid/AUG')
    for i in random.sample(glob.glob('ISIC*'), 15):
        shutil.move(i, 'test/ISIC')
    for i in random.sample(glob.glob('AUG*'), 15):
        shutil.move(i, 'test/AUG')
        
        
        
train_path = 'C:/Users/Lucas Gava/Desktop/IA/Possiveis Datas/DataMelanoma/train'
valid_path = 'C:/Users/Lucas Gava/Desktop/IA/Possiveis Datas/DataMelanoma/valid'
test_path = 'C:/Users/Lucas Gava/Desktop/IA/Possiveis Datas/DataMelanoma/test'



train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=train_path, target_size=(224,224), classes=['ISIC', 'AUG'], batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=valid_path, target_size=(224,224), classes=['ISIC', 'AUG'], batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=test_path, target_size=(224,224), classes=['ISIC', 'AUG'], batch_size=10, shuffle=False)
    
    
    
imgs, labels = next(train_batches)

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
    
plotImages(imgs)
print(labels)


model = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding = 'same', input_shape=(224,224,3)),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Flatten(),
    Dense(units=2, activation='softmax')
])


model.summary()

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

evo = model.fit(x=train_batches,
    steps_per_epoch=len(train_batches),
    validation_data=valid_batches,
    validation_steps=len(valid_batches),
    epochs=30,
    verbose=2
)


test_batches.classes


predictions = model.predict(x=test_batches, steps=len(test_batches), verbose=0)

np.round(predictions)

# cm = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1))

# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)

#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')

#     print(cm)

#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, cm[i, j],
#             horizontalalignment="center",
#             color="white" if cm[i, j] > thresh else "black")

#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
    
    
# test_batches.class_indices

# cm_plot_labels = ['Não Melanoma','Melanoma']
# plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')


plt.plot(evo.history['accuracy'])
plt.plot(evo.history['val_accuracy'])
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['treinio','validação'],loc='upper left')
plt.show()










