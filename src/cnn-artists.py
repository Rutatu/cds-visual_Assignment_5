#!/usr/bin/env python



''' ---------------- About the script ----------------

Assignment 5: CNNs on cultural image data

This script builds a deep learning model using convolutional neural networks which classify Impressionism paintings by their respective artists. It uses LeNet architecture for CNN.

Preprocessing of the data involves resizing the images, getting the images and labels into an array for the model. As an output, this script produces a visualization showing loss/accuracy of the model during training and the classification report.


Arguments:
    
    -trd,    --train_data:         Directory of training data
    -vald,   --val_data:           Directory of validation data
    -optim,  --optimizer:          Method to update the weight parameters to minimize the loss function. Choose between SGD and Adam.
    -lr,     --learning_rate:      The amount that the weights are updated during training. Default = 0.01
    -ep,     --epochs:             Defines how many times the learning algorithm will work through the entire training dataset. Default = 50




Example:    
    
    with default values:
        $ python cnn-artists.py -trd ../data/Impressionist_Classifier_data/training -vald ../data/Impressionist_Classifier_data/validation -optim SGD
        
    with optional arguments:
        $ python cnn-artists.py -trd ../data/Impressionist_Classifier_data/training -vald ../data/Impressionist_Classifier_data/validation - optim Adam -lr 0.002 -ep 100


'''






"""---------------- Importing libraries ----------------
"""

# data tools
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
import pandas as pd

sys.path.append(os.path.join(".."))

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
                                     Dense,
                                     Dropout)
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import backend as K
from tensorflow.keras.layers import LeakyReLU

from tensorflow.python.keras.preprocessing.image_dataset import image_dataset_from_directory
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import plot_model

# Command-line interface
import argparse




"""---------------- Main script ----------------
"""


def main():
    
    """------ Argparse parameters ------
    """
    # Instantiating the ArgumentParser  object as parser 
    parser = argparse.ArgumentParser(description = "[INFO] Classify Impressionists paintings and print out performance accuracy report")
    
    # Adding optional (with defaults) and required arguments
    parser.add_argument("-trd", "--train_data", required=True, help = "Directory of training data")
    parser.add_argument("-vald", "--val_data", required=True, help = "Directory of validation data")
    parser.add_argument("-optim", "--optimizer", required = False, default = SGD, help = "Method to update the weight parameters to minimize the loss function. Choose between SGD and Adam.")
    parser.add_argument("-lr", "--learning_rate", required = False, default = 0.01, type = float, help = "The amount that the weights are updated during training. Default = 0.01")
    parser.add_argument("-ep", "--epochs", required=False, default = 50, help = "Defines how many times the learning algorithm will work through the entire training dataset. Default = 50")
    
                                       
    # Parsing the arguments
    args = vars(parser.parse_args())
    
    # Saving parameters as variables
    trd = args["train_data"] # training data dir
    vald = args["val_data"] # validation data dir
    optim = args["optimizer"] # optimizer
    lr = args["learning_rate"] # learning rate
    ep = int(args["epochs"]) # epochs
   
    
     
    
    

    """------ Loading data and preprocessing ------
    """


    # getting training and validation data
    print("[INFO] loading and preprocessing training and validation data ...")
    train = get_data(os.path.join(trd))
    val = get_data(os.path.join(vald))
 
    
    #Create ouput folder, if it doesn´t exist already, for saving the classification report, performance graph and model´s architecture 
    if not os.path.exists("../output"):
        os.makedirs("../output")
    
    
    
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
    model.add(Conv2D(64, (3, 3), 
                     padding="same", 
                     input_shape=(256, 256, 3)))
    model.add(Activation("relu"))
    ##model.add(LeakyReLU(alpha = 0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), 
                           strides=(2, 2)))
    ##model.add(Dropout(0.3))

    # second set of CONV => RELU => POOL
    model.add(Conv2D(128, (5, 5), 
                     padding="same"))
    model.add(Activation("relu"))
    ##model.add(LeakyReLU(alpha = 0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), 
                           strides=(2, 2)))
    
    # FC => RELU
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation("relu"))
    
    # adding a dropout layer
    model.add(Dropout(0.5))

    ##model.add(LeakyReLU(alpha = 0.1))

    # softmax classifier
    model.add(Dense(10))
    model.add(Activation("softmax"))
    
    # ploting and saving model´s architecture
    plot_model(model, to_file='../output/Model´s_architecture.png',
               show_shapes=True,
               show_dtype=True,
               show_layer_names=True)
    
    # Printing that model´s architecture graph has been saved
    print(f"\n[INFO] Model´s architecture graph has been saved")
    
    
        
    #  ModelCheckpoint and EarlyStopping to monitor model performance
    ## checkpoint = ModelCheckpoint("LeNet_1.h5", 
                                  ## monitor='val_accuracy', 
                                  ## verbose=1, 
                                  ## save_best_only=True, 
                                  ## save_weights_only=False, 
                                  ## mode='auto', period=1)
    ## early = EarlyStopping(monitor='val_accuracy', min_delta=0, 
                         ## patience=25, verbose=1, mode='auto')
    
    
 

    """------ Optimizer choice ------
    """
    
    if optim == "SGD":
        opt = SGD(lr=lr)

        
        # compile model
        model.compile(loss="categorical_crossentropy",
                      optimizer=opt,
                      metrics=["accuracy"])
    
        print(model.summary())  
    
    
        # train model
        print("[INFO] training LeNet CNN model ...")
        H = model.fit(x_train, y_train, 
                      validation_data=(x_val, y_val), 
                      batch_size=32,
                      epochs=ep,
                      verbose=1)
                      #callbacks=[checkpoint,early])
    
    
    
    elif optim == "Adam":
        opt = Adam(lr=lr)
    

        # compile model
        model.compile(loss="categorical_crossentropy",
                      optimizer=opt,
                      metrics=["accuracy"])
    
        print(model.summary())
        
        # train model
        print("[INFO] training LeNet CNN model ...")
        H = model.fit(x_train, y_train, 
                      validation_data=(x_val, y_val), 
                      batch_size=32,
                      epochs=ep,
                      verbose=1)
                      #callbacks=[checkpoint,early])
            
    else:
        print("Not a valid optimizer. Choose between 'SGD' and 'Adam'.")
        
        
    
    """------ LeNet CNN model output ------
    """
    
    # finding out the number of epochs the model has run through
    # epochs = len(H.history['val_accuracy'])
    
   
    # ploting and saving model´s performance graph
    plot_history(H,ep)
    
    # Printing that performance graph has been saved
    print(f"\n[INFO] Model´s performance graph has been saved")
    
        
    # Extracting the labels
    labels = os.listdir(os.path.join(trd))
    # Classification report
    predictions = model.predict(x_val, batch_size=32)
    print(classification_report(y_val.argmax(axis=1),
                                predictions.argmax(axis=1),
                                target_names=labels))
    
    
    
    # defining full filepath to save .csv file 
    outfile = os.path.join("../", "output", "Impressionist_classifier_report.csv")
    
    # turning report into dataframe and saving as .csv
    report = pd.DataFrame(classification_report(y_val.argmax(axis=1), predictions.argmax(axis=1), target_names=labels, output_dict = True)).transpose()
    report.to_csv(outfile)
    print(f"\n[INFO] Classification report has been saved")

    
    print("\nScript was executed successfully! Have a nice day")
        
        
    
    
    
    
    
    
"""---------------- Functions ----------------
"""

# this function was developed for use in class and has been adapted for this project
def plot_history(H, epochs):
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
    plt.savefig('../output/LeNet_CNN_model´s_performance.png')


    

# this function was taken from Gautam (2020) and adapted for this project
def get_data(data_dir):
    '''
    loads training and validation data/images in the RGB format,
    reshapes images and converts them into a numpy array  
    
    '''
    # Extracting the labels
    labels = os.listdir(os.path.join(data_dir))
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

    
"""
"""
    
    
    
         
# Define behaviour when called from command line
if __name__=="__main__":
    main()

