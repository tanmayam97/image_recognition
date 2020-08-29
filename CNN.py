# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 19:06:26 2020

@author: Tanmay Ambatkar
"""


import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras import layers
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
plt.style.use('fivethirtyeight')

# Load the data

from keras.datasets import cifar10

# loading the data and downloading it

(x_train,y_train),(x_test,y_test) = cifar10.load_data()

# Look at the data types of the variables

print(type(x_train))
print(type(y_train))
print(type(x_test))
print(type(y_test))

# get the shape of the arrays

print('x_train shape',x_train.shape)
print('y_train shape',y_train.shape)
print('x_test shape',x_test.shape)
print('y_test shape',y_test.shape)

# take a look at the first image as array

index = 10
x_train[index]

# show the imagew as a picture

img = plt.imshow(x_train[index])

# get the image label

print("The image label is:",y_train[index])

# Get the image classification

classification = ['aeroplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# print the image class

print('The image class is:',classification[y_train[index][0]])

# convert the label into a set of 10 numbers to input into the neural network

y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

# print the new labels

print(y_train_one_hot)

# print the new label of the image / pic above

print('The one hot new label is:',y_train_one_hot[index])


# Normalize the pixels to be values between 0 and 1

x_train = x_train/255
x_test = x_test/255

x_train[index]


# create the models

model = Sequential()

# Add the first layer Convulution Layer
model.add(Conv2D(32,(5,5), activation='relu', input_shape=(32,32,3)))

# add the pooling layer
model.add(MaxPooling2D(pool_size=(2,2)))

# Add Second Convulution layer
model.add(Conv2D(32,(5,5), activation='relu'))

# Add another pooling layer
model.add(MaxPooling2D(pool_size=(2,2)))

# Add Flatening layer
model.add(Flatten())

# Add a layer with 1000 neurons
model.add(Dense(1000, activation = 'relu'))

# Add a Dropout layer
model.add(Dropout(0.5))

# Add another layer with 500 neurons
model.add(Dense(500, activation = 'relu'))

# Add another Dropout layer
model.add(Dropout(0.5))

# Add another layer with 500 neurons
model.add(Dense(250, activation = 'relu'))

# Add another layer with 10 neurons
model.add(Dense(10, activation = 'softmax'))

# compile the model

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# train the model

hist = model.fit(x_train,y_train_one_hot,batch_size=256,epochs=10,validation_split=0.2)

# Evaluate the model using the test data set

model.evaluate(x_test,y_test_one_hot)[1]

# visualize the models accuracy

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Val'], loc = 'upper left')
plt.show()

# visualize the models loss

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Val'], loc = 'upper right')
plt.show()

# Test the model with example

new_image = mpimg.imread(r"C:\Users\Tanmay Ambatkar\Downloads\cat.jpg")
img = plt.imshow(new_image)

# resize the image
from skimage.transform import resize
resized_image = resize(new_image, (32,32,3));
img = plt.imshow(resized_image)

# Lets Predict the image

predictions = model.predict(np.array([resized_image]))
predictions

# sort the predictions

list_index = [0,1,2,3,4,5,6,7,8,9]
x = predictions

for i in range(10):
    for j in range(10):
        if x[0][list_index[i]] > x[0][list_index[j]]:
            temp = list_index[i]
            list_index[i] = list_index[j]
            list_index[j] = temp
# show the sorted label
print(list_index)

# print the first 5 predictions
for i in range(5):
    print(classification[list_index[i]] , ':' , round(predictions[0][list_index[i]] * 100, 2), '%')
            