import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
import re # removes links
import numpy as np
from textPreProcessing import DataProcessor

nltk.download('wordnet')
nltk.download('vader_lexicon')

dataframe = pd.read_csv("./Dataset/training.csv",
                        encoding='ISO-8859-1',
                        names=['target','ids','date','flag','user','tweet'])

dataframe['target'] = np.where(dataframe['target'] == 4, 1, 0)

indexes = np.load("./datasetBothModels/testIndexes.npy")
dataframe = dataframe.loc[indexes]

cleaner = DataProcessor()

dataframe["tweet"] = cleaner.CleanTextData(dataframe["tweet"])

testData = dataframe["tweet"].values.tolist()
testLabels = dataframe["target"].values.tolist()

vader = SentimentIntensityAnalyzer()

predictArray = []
for tweet in testData:
  score = vader.polarity_scores(tweet)

  if score['compound'] < 0.0:
    predicted = 0.0
  else:
    predicted = 1.0
  
  predictArray.append(predicted)

testLabels = np.array(testLabels)
predictArray = np.array(testLabels)

correct = (testLabels == predictArray).sum()

print(f"VADER Accuracy on test dataset: {correct}/{len(testLabels)} | {(correct / len(testLabels)) * 100}%")