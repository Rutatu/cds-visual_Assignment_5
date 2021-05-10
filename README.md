# cds-visual_Assignment_5

***Assignment for visual analytics class at Aarhus University.***

***2021-04-13***


# CNNs on cultural image data: Multi-class classification of impressionist painters

### About the script

This script builds a deep learning model using convolutional neural networks which classify Impressionism paintings by their respective artists. It uses LeNet architecture for CNN. Why might we want to do this? Well, consider the scenario where we have found a new, never-before-seen painting which is claimed to be the artist Renoir. An accurate predictive model could be useful here for art historians and archivists!

### Data

Paintings of 10 Impressionist painters, namely: Camille Pisarro, Childe Hassam, Claude Monet, Edgar Degas, Henri Matisse, John Singer-Sargent, Paul Cezanne, Paul Gauguin, Pierre-Auguste Renoir, Vincent van Gogh. It consists of 400 training images and 100 validation images respectively for each of the 10 artists.

Data used for this assignment can be found here: https://www.kaggle.com/delayedkarma/impressionist-classifier-data


### Data preprocessing and methods





### Intructions to run the code


___Data structure___

Before executing the code make sure that the images are located in the following path: ***'data/Impressionist_Classifier_data'***

***'Impressionist_Classifier_data'*** folder should contain two folders: training and validation, each of which contains ten folders labeled by an artist name.
The code should work on any other similar image data structured this way, however the model parameters and preprocessing might require readjustments based on different data.



___Steps:___

- Open terminal on worker02 or locally
- Navigate to the environment where you want to clone this repository
- Clone the repository:
```Ruby
$ git clone https://github.com/Rutatu/cds-visual_Assignment_5.git 
``` 

- Navigate to the newly cloned repo:
```Ruby
$ cd cds-visual_Assignment_5
 ```

- Create virtual environment with its dependencies and activate it:
```Ruby
$ bash cnn_venv.sh
$ source ./cnn_venv/bin/activate
 ```

- Navigate to the directory of the script:
```Ruby
$ cd src
 ```
- Run the code:
```Ruby
$ python cnn-artists.py -trd data/Impressionist_Classifier_data/training -vald data/Impressionist_Classifier_data/validation
 ```

- To remove the newly created virtual environment:
``` Ruby
bash kill_cnn.sh
``` 




I hope it worked!


__References__

https://www.analyticsvidhya.com/blog/2020/10/create-image-classification-model-python-keras/
