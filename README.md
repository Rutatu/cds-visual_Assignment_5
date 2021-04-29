# cds-visual_Assignment_5

***Assignment for visual analytics class at Aarhus University.***

***2021-04-13***


# CNNs on cultural image data: Multi-class classification of impressionist painters

## About the script

This script builds a deep learning model using convolutional neural networks which classify Impressionism paintings by their respective artists. It uses LeNet architecture for CNN. Why might we want to do this? Well, consider the scenario where we have found a new, never-before-seen painting which is claimed to be the artist Renoir. An accurate predictive model could be useful here for art historians and archivists!

Data used for this assignment can be found here: https://www.kaggle.com/delayedkarma/impressionist-classifier-data




__Purpose__

This assignment is designed to test that you have an understanding of:

- how to build and train deep convolutional neural networks;
- how to preprocess and prepare image data for use in these models;
- how to work with complex, cultural image data, rather than toy datasets


## Intructions to run the code

Before executing the code make sure that the images are located in the following path: ***'data/Impressionist_Classifier_data'***

***'Impressionist_Classifier_data'*** folder should contain two folders: training and validation, which contain ten folders labeled by an artist name.

__Steps:__

- Open terminal on worker02
- Navigate to the environment where you want to clone this repository
- Clone the repository:
``` Ruby
git clone https://github.com/Rutatu/cds-visual_Assignment_5.git 
``` 
- Navigate to the newly cloned repo
- Script can be executed either from the bash file or .py file depending on whether you need/want to create a virtual environment:
``` Ruby
# Run the code from bash file (it will install requirements.txt file and execute the code)
bash cnn_venv.sh

# Run the code from .py file 
python cnn-artists.py
``` 

- Remove the newly created virtual environment (if you did so):
``` Ruby
bash kill_cnn.sh
``` 




I hope it worked!


__References__

https://www.analyticsvidhya.com/blog/2020/10/create-image-classification-model-python-keras/
