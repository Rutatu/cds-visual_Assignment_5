# cds-visual_Assignment_5

***Assignment for visual analytics class at Aarhus University.***

***2021-04-13***


# CNNs on cultural image data: Multi-class classification of impressionist painters

## About the script

This script builds a deep learning model using convolutional neural networks which classify Impressionism paintings by their respective artists. It uses LeNet architecture for CNN. Why might we want to do this? Well, consider the scenario where we have found a new, never-before-seen painting which is claimed to be the artist Renoir. An accurate predictive model could be useful here for art historians and archivists!

## Methods
The problem of the task relates to classifying art paintings. To address this problem, I have used LeNet architecture to build a CNN model using a neural network framework TensorFlow 2.4. LeNet architecture used in this assignment consists of 6 layers: two Convolutional Layers (CONV), two Max Pooling layers (MAXPOOL), two Fully Connected Layers (FC). It uses activation function ReLU except for the output or classification layer (FC 2) with 10 possible classes, which uses softmax activation function.

LeNet architecture: INPUT => CONV 1=> ReLU => MAXPOOL 1=> CONV 2=> ReLU => MAXPOOL 2 => FC 1=> ReLU => FC 2

Depiction of the full model architecture can be found in folder called ***'out'***.

 


## Repository contents

| File | Description |
| --- | --- |
| git status | List all new or modified files |
| git diff | Show file differences that haven't been staged |



## Data

Paintings of 10 Impressionist painters, namely: Camille Pisarro, Childe Hassam, Claude Monet, Edgar Degas, Henri Matisse, John Singer-Sargent, Paul Cezanne, Paul Gauguin, Pierre-Auguste Renoir, Vincent van Gogh. It consists of 400 training images and 100 validation images respectively for each of the 10 artists.

Data used for this assignment can be found here: https://www.kaggle.com/delayedkarma/impressionist-classifier-data

__Data structure__

Before executing the code make sure that the images are located in the following path: ***'data/Impressionist_Classifier_data'***

***'Impressionist_Classifier_data'*** folder should contain two folders: training and validation, each of which contains ten folders labeled by an artist name.
The code should work on any other similar image data structured this way, however the model parameters and preprocessing might require readjustments based on different data.


__Data preprocessing__

Image preprocessing involved reshaping all images to a preferred size of 256x256, converting images from BGR to RGB format, because openCV opened it in BGR (blue, green, red) format, extracting the labels of the images from the subfolders directory name.



## Intructions to run the code

The code was tested on an HP computer with Windows 10 operating system. It was executed on Jupyter worker02.

__Code parameters__


| Parameter | Description |
| --- | --- |
| train_data | Directory of training data |
| val_data | Directory of validation data |
| learning_rate | Learning rate. Default = 0.01 |
| optimizer | Optimizer. Default = SGD |
| epochs | Number of epochs. Default = 50 |


__Steps__

Set-up:
```
#1 Open terminal on worker02 or locally
#2 Navigate to the environment where you want to clone this repository
#3 Clone the repository
$ git clone https://github.com/Rutatu/cds-visual_Assignment_5.git 

#4 Navigate to the newly cloned repo
$ cd cds-visual_Assignment_5

#5 Create virtual environment with its dependencies and activate it
$ bash cnn_venv.sh
$ source ./cnn_venv/bin/activate

``` 

Run the code:

```
#6 Navigate to the directory of the script
$ cd src

#7 Run the code with default parameters
$ python cnn-artists.py -trd data/Impressionist_Classifier_data/training -vald data/Impressionist_Classifier_data/validation

#8 Run the code with self-chosen parameters
$ python cnn-artists.py -trd data/Impressionist_Classifier_data/training -vald data/Impressionist_Classifier_data/validation -lr 0.002 -opt Adam -ep 100

#9 To remove the newly created virtual environment
$ bash kill_cnn.sh

 ```


I hope it worked!


__References__

https://www.analyticsvidhya.com/blog/2020/10/create-image-classification-model-python-keras/
