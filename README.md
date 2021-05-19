# cds-visual_Assignment_5

***Class assignment for visual analytics class at Aarhus University.***

***2021-04-13***


# CNNs on cultural image data: Multi-class classification of impressionist painters

## About the script

This assignment is Class Assignment 5. This script builds a deep learning model using convolutional neural networks which classify Impressionism paintings by their respective artists. It uses LeNet architecture for CNN. Why might we want to do this? Well, consider the scenario where we have found a new, never-before-seen painting which is claimed to be the artist Renoir. An accurate predictive model could be useful here for art historians and archivists!

## Methods
The problem of the task relates to classifying art paintings. To address this problem, I have used LeNet architecture to build a CNN model using a neural network framework TensorFlow 2.4. LeNet architecture used in this assignment consists of 6 layers: two Convolutional Layers (CONV), two Max Pooling layers (MAXPOOL), two Fully Connected Layers (FC). It uses activation function ReLU except for the output or classification layer (FC 2) with 10 possible classes, which uses softmax activation function. One regularization technique - a dropout with a rate of 50%, was employed after a dense fully connected layer (FC1), in order to improve the model´s fit


LeNet architecture: INPUT => CONV 1=> ReLU => MAXPOOL 1=> CONV 2=> ReLU => MAXPOOL 2 => FC 1=> ReLU => FC 2

Depiction of the full model´s architecture can be found in folder called ***'output'***.

 


## Repository contents

| File | Description |
| --- | --- |
| output | Folder containing files produced by the script |
| output/Impressionist_classifier_report.csv | Classification metrics of the model |
| output/LeNet_CNN_model´s_performance.png | Model´s performance graph |
| output/Model´s_architecture.png | Depiction of CNN model´s architecture used |
| src | Folder containing the script |
| src/cnn_artists.py | The script |
| README.md | Description of the assignment and the instructions |
| cnn_venv.sh | bash file for creating a virtual environmment  |
| kill_cnn.sh | bash file for removing a virtual environment |
| requirements.txt | list of python packages required to run the script |



## Data

Paintings of 10 Impressionist painters, namely: Camille Pisarro, Childe Hassam, Claude Monet, Edgar Degas, Henri Matisse, John Singer-Sargent, Paul Cezanne, Paul Gauguin, Pierre-Auguste Renoir, Vincent van Gogh. It consists of 400 training images and 100 validation images respectively for each of the 10 artists.

Data used for this assignment can be found here: https://www.kaggle.com/delayedkarma/impressionist-classifier-data

__Data structure__

Before executing the code make sure that the images are located in the following path: ***'data/Impressionist_Classifier_data'***

***'Impressionist_Classifier_data'*** folder should contain two folders: training and validation, each of which contains ten folders labeled by an artist name.
The code should work on any other similar image data structured this way, however the model parameters and preprocessing might require readjustments based on different data.


__Data preprocessing__

Image preprocessing involved reshaping all images to a preferred size of 256x256, converting images from BGR to RGB format, because openCV opened it in BGR (blue, green, red) format, extracting the labels of the images from the subfolders directory names.



## Intructions to run the code

The code was tested on an HP computer with Windows 10 operating system. It was executed on Jupyter worker02.

__Code parameters__


| Parameter | Description |
| --- | --- |
| train_data  (trd) | Directory of training data |
| val_data (vald) | Directory of validation data |
| optimizer (optim) | Optimizer. Choose betweeen SGD and Adam |
| learning_rate (lr) | Learning rate. Default = 0.01 |
| epochs (ep) | Number of epochs. Default = 50 |


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
$ python cnn-artists.py -trd data/Impressionist_Classifier_data/training -vald data/Impressionist_Classifier_data/validation -optim SGD

#8 Run the code with self-chosen parameters
$ python cnn-artists.py -trd data/Impressionist_Classifier_data/training -vald data/Impressionist_Classifier_data/validation -optim Adam -lr 0.002 -ep 100

#9 To remove the newly created virtual environment
$ bash kill_cnn.sh

#10 To find out possible optional arguments for the script
$ python cnn-artists.py --help


 ```

I hope it worked!


## Results

The current CNN model achieved a weighted average accuracy of 44% for correctly classifying artists according to their paintings.

Performance graph with training and validation curves reveals that the model was able to learn (training loss curve flattened out after a significant decrease at around 20th epoch). However, while the validation loss curve showed a decrease at the very beginning, it started growing after the 10th epoch and was fluctuating throughout. It ended up creating a huge gap between training and validation loss curves, which may indicate a poor generalizability and an overfit of the model. More data or different regularization methods might be needed to improve the accuracy.




## References

Brownlee, J. (2019, February 27). How to use Learning Curves to Diagnose Machine Learning Model Performance. Machine Learning Mastery https://machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/

Gautam, T. (2020, October 16). Create your Own Image Classification Model using Python and Keras. Analytics Vidhya https://www.analyticsvidhya.com/blog/2020/10/create-image-classification-model-python-keras/

