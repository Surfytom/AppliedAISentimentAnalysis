This folder contains .pynb files and .py files.

All .py files can be run to show the code process of my project.

However I recommend running the .pynb files in google colab as the models take up alot of ram
and the dataset embeddings can take up alot of space aswell.

If you want to load the kaggle dataset please download it from this link: https://www.kaggle.com/datasets/kazanova/sentiment140
Place this dataset within the Dataset folder and name the file training
I was not able to include the dataset in the assignment folder as it was 230 mb

I was also not able to include the training word embeddings as they took up too much space.
These are not needed to load and use already trained models

Training the models takes around 30 minutes depending on the epochs and the system the code is runnning on.

Model Training shows the training process for the tranformer model
ModelEvaluation shows the code to generate the graphs shown in the documentation

VADEREvaluation shows the evaluation process for the VADER model

XGBoost training shows the training process for the XGBoost model along with results

text pre processing shows the code I ran to pre-process the tweets

dataset contains the csv downloaded from the kaggle site

datasetBothModels shows the dataset produced by the tranformer models

The models folder contains all the models trained and used for the evalution in the documentation

IF USING COLAB PLEASE CHANGE THE PATH TO THE CURRENT FOLDER PATH IN YOUR DRIVE

You need all libraries listed below to run the code:

nltk
numpy
pandas
sklearn
glob
maplotlib
transformers
pytorch
regex
xgboost
os