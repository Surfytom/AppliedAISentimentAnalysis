# Binary Sentiment Analysis Model Training and Evaluation

This repository contains most the files handed in for my Applied AI module assignment for my masters in artificial intelligence course. I was not able to commit my model weights and dataset used [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140) as the files were too big.

This project contained training 3 distinct models on the tast of binary sentiment analysis and evaluating their performance. The three models I trained were a simple binary classifying MLP trained on embeddings generated from a trained encoder section of a tranformer (I trained and compared DISTILLBERT and MINILM), VADER (a rule based system) and XGBoost (a gradient boosting solution).

There are .py versions of .ipynb files as .py was required to hand in but I prefered working on .ipynb files.

## File Breakdown

### [DataGeneration.ipynb](https://github.com/Surfytom/AppliedAISentimentAnalysis/blob/main/DataGeneration.ipynb) and [TextPreProcessing.py](https://github.com/Surfytom/AppliedAISentimentAnalysis/blob/main/textPreProcessing.py)

These two files were responsible for pre-processing and sampling an appropriate amount of data from the original dataset as it consisted of an overwhelming 1.6 million tweets. My pre-processing techniques included removing whitespace, translating slang words and removing things such as usernames and urls that would not be useful for the model to learn from. I did this using Pandas, NLTK and Regex.

### [ModelTraining.ipynb](https://github.com/Surfytom/Assignment/blob/main/ModelTraining.ipynb)

This file showcases the training code for the two tranformer embedding based binary classifiers. This involved fetching the specific data generated by the [DataGeneration.ipynb](https://github.com/Surfytom/AppliedAISentimentAnalysis/blob/main/DataGeneration.ipynb) file which imported and ran inference on a portion of the dataset using the two embedding models. The binary classifiers are trained using a normal MLP structure and Binary Cross Entropy loss function.

### [ModelEvaluation.ipynb](https://github.com/Surfytom/AppliedAISentimentAnalysis/blob/main/ModelTraining.ipynb)

This file runs through a few graphs displaying the performance of the two binary classifyer models. I mainly focus on the loss of the models to evaluate performance. Please read the [Final Project PDF Paper]() which has a section on hyper-parameter tuning and evaluation of the models.

### Other Files

#### [XGBoostTraining.ipynb](https://github.com/Surfytom/AppliedAISentimentAnalysis/blob/main/XGBoostTraining.ipynb)

This was used to train the XGBoost model by importing the xgboost library and using their ibuilt functions.

#### [VADEREvaluation.py](https://github.com/Surfytom/AppliedAISentimentAnalysis/blob/main/VADEREvaluation.py)

This script runs the NLTK built in VADER sentiment analyser and evaluates its performance on the dataset.
