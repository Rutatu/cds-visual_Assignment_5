#!/usr/bin/env python

''' ---------------- About the script ----------------

Assignment 5: CNNs on cultural image data

This script builds a deep learning model using convolutional neural networks which classify Impressionism paintings by their respective artists. It uses LeNet architecture for CNN.

Preprocessing of the data involves resizing the images, getting the images and labels into an array for the model. As an output, this script produces a visualization showing loss/accuracy of the model during training and the classification report.


Example:    
    $ python cnn-artists.py


'''




"""---------------- Importing libraries ----------------
"""

# data tools
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
import pandas as pd

# Import pathlib
from pathlib import Path

# sklearn tools
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

# tf tools
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, 
                                     MaxPooling2D, 
                                     Activation, 
                                     Flatten, 
                                     Dense)
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K

from tensorflow.python.keras.preprocessing.image_dataset import image_dataset_from_directory





"""---------------- Functions ----------------
"""

def plot_model(H, epochs):
    '''
    visualize model´s performance: training and validation loss, 
    training and validation accuracy
    
    '''
    plt.style.use("fivethirtyeight")
    plt.figure()
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.tight_layout()
    #plt.show()
    plt.savefig('LeNet CNN model performance.png')


  
    
def get_data(data_dir):
    '''
    loads training and validation data/images in the RGB format,
    reshapes images and converts them into numpy array  
    
    '''
    # define labels
    labels = ["Cezanne", "Degas", "Gauguin", "Hassam", "Matisse", "Monet", "Pissarro", "Renoir", "Sargent", "VanGogh"]
    # defining desirable image size
    img_size = 256
    # an empty list to store data
    data = [] 
    # a loop through each folder according to labels
    for label in labels: 
        # defining folder path
        path = os.path.join(data_dir, label)
        # assigning an index to a label/folder
        class_num = labels.index(label)
        # a loop through each image in each folder in the path
        for img in os.listdir(path):   # returns a list containing the names of the entries in the directory given by path
            try:
                # reading an image into an object called 'image array'
                img_arr = cv2.imread(os.path.join(path, img))[...,::-1] # convert BGR to RGB format, cause openCV opens it in BGR (blue, green, red) format
                # reshaping an image to preferred size
                resized_arr = cv2.resize(img_arr, (img_size, img_size))
                # appending the resized image and label index to a list
                data.append([resized_arr, class_num])
            # makes sure that the loop has gone through all the files    
            except Exception as e:
                print(e)
    # return a numpy array             
    return np.array(data, dtype=object)





"""---------------- Main script ----------------
"""


def main():

    """------ Loading data and preprocessing ------
    """


    # getting training and validation data
    print("[INFO] loading and preprocessing training and validation data ...")
    train = get_data('data/Impressionist_Classifier_data/training')
    val = get_data('data/Impressionist_Classifier_data/validation')
    
    
    
    
    """------ Preparing training and validations sets ------
    """
       
    # empty lists for training and validation images and labels
    x_train = []
    y_train = []
    x_val = []
    y_val = []

    # appending features (images as numpy arrays) and labels to the empty lists for further processing
    for feature, label in train:
        x_train.append(feature)
        y_train.append(label)

    for feature, label in val:
        x_val.append(feature)
        y_val.append(label)

    # normalizing the data (rescaling RGB channel values from 0-255 to 0-1)
    x_train = np.array(x_train) / 255
    x_val = np.array(x_val) / 255

    # integers to one-hot vectors
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_val = lb.fit_transform(y_val)
    
    
    
    
    """------ Defining and training LeNet CNN model ------
    """
    
    # LeNet architecture:
    ## >INPUT => CONV => ReLU => MAXPOOL => CONV => ReLU => MAXPOOL => FC => ReLU => FC
    
    # define model
    model = Sequential()

    # first set of CONV => RELU => POOL
    model.add(Conv2D(32, (3, 3), 
                     padding="same", 
                     input_shape=(256, 256, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), 
                           strides=(2, 2)))

    # second set of CONV => RELU => POOL
    model.add(Conv2D(50, (5, 5), 
                     padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), 
                           strides=(2, 2)))

    # FC => RELU
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation("relu"))

    # softmax classifier
    model.add(Dense(10))
    model.add(Activation("softmax"))
    
    
    # compile model
    opt = SGD(lr=0.01)
    model.compile(loss="categorical_crossentropy",
                  optimizer=opt,
                  metrics=["accuracy"])
    
    print(model.summary())
    
    
    
    # train model
    print("[INFO] training LeNet CNN model ...")
    H = model.fit(x_train, y_train, 
                  validation_data=(x_val, y_val), 
                  batch_size=32,
                  epochs=20,
                  verbose=1)
    
    
    
    
    """------ LeNet CNN model output ------
    """
    
    
    # ploting and saving model´s performance graph
    plot_model(H,20)
    
    # classification report
    labels = ["Cezanne", "Degas", "Gauguin", "Hassam", "Matisse", "Monet", "Pissarro", "Renoir", "Sargent", "VanGogh"]
    predictions = model.predict(x_val, batch_size=32)
    print(classification_report(y_val.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=labels))
    
    # turning into dataframe and saving report
    report = pd.DataFrame(classification_report(y_val.argmax(axis=1), predictions.argmax(axis=1), target_names=labels, output_dict = True)).transpose()
    report.to_csv('Impressionist_classifier_report.csv')

    
    
    
    print("Script was executed successfully! Have a nice day")

    
         
# Define behaviour when called from command line
if __name__=="__main__":
    main()

