# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 10:46:37 2020

@author: uni tech
"""

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten,Activation,  Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import cv2
import os
from imutils import paths
import numpy as np
import random
from sklearn.model_selection import train_test_split




training_data =  []


path = "C:/Users/uni tech/Desktop/spyderr/datasets/SOCOFing/Real"


for img in os.listdir(path):
    list_of_strings=[]
    img_path = os.path.join(path,img)
    img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (80, 80))

    
    new_name=os.path.split(img_path)[-1]
    new_name2 = new_name[:-4]
   
    for x in new_name2:
        list_of_strings.append(x)
 
    
    if "M" in list_of_strings:
        training_data.append([new_array, 0])
        
    elif "F" in list_of_strings:
        training_data.append([new_array, 1])
   
        

        
random.shuffle(training_data)


for sample in training_data:
    print(sample[1])
    

X=[]
y=[]

for features, labels in training_data:
    X.append(features)
    y.append(labels)
    

X = np.array(X).reshape(-1, 80, 80, 1)
X = X / 255

# Defining callbacks function    
early_stoppings = EarlyStopping(monitor='val_loss',
                                patience = 3,
                                verbose = 1,
                                restore_best_weights = True)   


# Defining the model and adding layers to it
model = Sequential()
model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())
model.add(Dense(128))
model.add(Activation("relu"))

model.add(Dense(64))
model.add(Activation("relu"))


model.add(Dense(1))
model.add(Activation("sigmoid"))


print(model.summary())


# model compilation
# adam = Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(optimizer = "adam", loss="binary_crossentropy",  metrics=['accuracy'])

# # Model training
model.fit(X, y ,batch_size=100 ,epochs = 5, validation_split=0.1 , callbacks= [early_stoppings])

# from keras.models import load_model 
model.save('fingerprint_recog.h5')



    
    
        

      