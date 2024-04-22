# -*- coding: utf-8 -*-
"""ModelTraining.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1nQmdPe_iVn82VosmrU1JN5Ob9xJmbn8g
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
import pandas as pd
import re
import torch
import transformers
import os

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('vader_lexicon')

# This is the model class used to outline the classifier models architecture
class Model(torch.nn.Module):

  def __init__(self, miniLMInput=False):
    super().__init__()

    # Changes the input size for the distill bert and mini lm embeddings
    inputSize = 768

    if miniLMInput:
      inputSize = 384

    # Defines the size and amount of linear layers the model has
    self.linear1 = torch.nn.Linear(inputSize, 128)
    self.linear2 = torch.nn.Linear(128, 64)
    self.linear3 = torch.nn.Linear(64, 32)
    # self.linear4 = torch.nn.Linear(256, 128)
    # self.linear5 = torch.nn.Linear(128, 30)
    # self.linear6 = torch.nn.Linear(30, 10)
    self.output = torch.nn.Linear(32, 1)

    self.debug = False

  def forward(self, x):
    # Runs through each layer with a forward pass and return output
    x = self.linear1(x)
    x = torch.nn.functional.leaky_relu(x)

    if self.debug:
      print(f"L1: {x}")

    x = self.linear2(x)
    x = torch.nn.functional.leaky_relu(x)

    if self.debug:
      print(f"L2: {x}")

    #drop = torch.nn.Dropout(p=0.7)
    #x = drop(self.linear3(x))
    x = self.linear3(x)
    x = torch.nn.functional.leaky_relu(x)

    if self.debug:
      print(f"L3: {x}")

    # drop = torch.nn.Dropout(p=0.3)
    # x = drop(self.linear4(x))
    # x = torch.nn.functional.relu(x)

    # drop = torch.nn.Dropout(p=0.5)
    # x = drop(self.linear5(x))
    # x = torch.nn.functional.relu(x)

    # drop = torch.nn.Dropout(p=0.7)
    # x = drop(self.linear6(x))
    # x = torch.nn.functional.relu(x)

    x = self.output(x)

    if self.debug:
      print(f"x: {x}")

    return torch.nn.functional.sigmoid(x)

  def setDebug(self, value):
    # sets debug to value
    self.debug = value

"""##MINILM Training"""

# Gets mini lm test batches from files
paths = glob.glob("./datasetBothModels/miniLM/testbatches/*.npy")
testBatches = [path for path in paths if "labels" not in path]
testLabels = [path for path in paths if "labels" in path]

# Gets mini lm training batches from files
paths = glob.glob("./datasetBothModels/miniLM/batches/*.npy")
batches = [path for path in paths if "labels" not in path]
labels = [path for path in paths if "labels" in path]

# Sets the params for the model
params = {
    "learningRate": 0.001,
    "optimizer": "Adam"
}

# Create the classifier model and sends it to the gpu for training
miniLMModel = Model(miniLMInput=True).cuda()

# Defines the loss function (Binary Cross Entropy)
criterion = torch.nn.BCELoss()

# Defines the loss optimizer
optimizer = torch.optim.Adam(miniLMModel.parameters(), lr = params["learningRate"])

def testModel():
  # This function runs the models on a validation dataset to assess acurracy

  miniLMModel.eval()

  predicted = []
  truth = []

  # Runs through test batches and labels
  for batchPath, labelPath in zip(testBatches, testLabels):

    # Loads the bathes into numpy arrays
    x2 = np.load(batchPath)

    y2 = np.load(labelPath)
    y2 = y2.reshape(-1, 1)

    # Splits the minilm batches into smaller batches of 256
    miniX = int(x2.shape[0] / 4)

    for j in range(4):
      # Gets the 256 batches in seperate arrays
      idx = j*miniX
      y = torch.tensor(y2[idx:idx + miniX]).float()
      y = y.cuda()

      x = torch.tensor(x2[idx:idx + miniX])
      x = x.cuda()

      # Runs the models in evaluation mode on these batches
      with torch.no_grad():
        y_pred = miniLMModel(x)

      y = y.cpu()
      y_pred = y_pred.cpu()

      # Converts results to clamp to 0 or 1 to compare to the truth values
      y_pred = torch.where(y_pred <= 0.5,  0, 1)
      predicted = torch.cat((torch.tensor(predicted), y_pred))
      truth = torch.cat((torch.tensor(truth), y))

  # Calculates accuracy
  acc = (truth == predicted).sum().float()/len(truth)

  return acc

epochs = 50

predicted = []
truth = []
epochLosses = []

for i in range(epochs):

  # Dynamic loss if
  # if len(losses) > 50:
  #   if losses[-1] <= 0.15:
  #     optimizer = torch.optim.Adam(miniLMModel.parameters(), lr = (params["learningRate"] / 10))

  losses = []
  miniLMModel.train()
  for batchPath, labelPath in zip(batches, labels):

    # print(f"batch: {batchPath[-17:]}")
    # print(f"batch: {labelPath[-17:]}")

    x2 = np.load(batchPath)
    #x = x / np.linalg.norm(x)

    y2 = np.load(labelPath)
    y2 = y2.reshape(-1, 1)
    miniX = int(x2.shape[0] / 4)

    for j in range(4):

      idx = j*miniX
      y = torch.tensor(y2[idx:idx + miniX]).float()
      y = y.cuda()

      x = torch.tensor(x2[idx:idx + miniX])
      x = x.cuda()

      # Runs model of training batch
      y_pred = miniLMModel(x)

      #print(f"{x.size()} | {y.size()}")
      #print(f"yPred {y_pred}")
      #print(f"y {y}")
      #print(y_pred)
      #print(y)
      #print(torch.max(y_pred, 1)[1])
      # print(f"{y_pred.size()} | {y.size()}")
      # print(f"{y_pred.type()}) | {y.type()}")

      #print(y_pred.device)
      #print(y.device)

      # Calculates loss based on difference between prediction and actual truth value
      loss = criterion(y_pred, y)

      # Appends loss value for evaluation
      losses.append(loss.cpu().detach().numpy())

      #print(f"{y_pred.size()} | {y.size()}")
      #print(f"{y_pred[y_pred == 1]} \n {y[:10]}")
      #print(f"{y_pred[y_pred == 0]}")
      #print(f"loss: {loss}")
      #print((y==y_pred))
      #print(predicted)
      #print(y_pred)

      # Takes optimizer step in loss space (updates model weights)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      y = y.cpu()
      y_pred = y_pred.cpu()

      # Clamps y hat values to 0 or 1 for comparison with truth values
      y_pred = torch.where(y_pred <= 0.5,  0, 1)
      predicted = torch.cat((torch.tensor(predicted), y_pred))
      truth = torch.cat((torch.tensor(truth), y))
      # print(f"y_pred: {y_pred.shape}")
      # print(f"y: {y.shape}")
      # print(f"predicted: {predicted.shape}")
      # print(f"truth: {truth.shape}")

  # Calculates training accuracy
  acc = (truth == predicted).sum().float()/len(truth)

  # Validation test
  valAcc = testModel()

  # Appends epochs average loss for evaluation
  epochLosses.append(np.array(losses).mean())

  #losses = torch.cat((torch.tensor(losses), torch.tensor(loss.cpu().item())))

  print(f"Epoch {i} | Loss {loss.item()} | Accuracy {acc} | Validation Accuracy {valAcc}")

# Saves model with unique name to load later and evaluate
modelName = f"MINILM-{round(acc.item(), 3)}-{round(valAcc.item(), 3)}-{round(epochLosses[-1].item(), 3)}-{epochs}-{params['learningRate']}-{params['optimizer']}"

folderPath = f"./models/{modelName}/"

if not os.path.exists(folderPath):
    os.makedirs(folderPath)

torch.save(miniLMModel, f"{folderPath}{modelName}.pt")
np.save(f"{folderPath}/losses.npy", np.array(epochLosses))

"""##DistillBert Training"""

# This code is the same as the mini lm code

paths = glob.glob("./datasetBothModels/distillBert/testbatches/*.npy")
testBatches = [path for path in paths if "labels" not in path]
testLabels = [path for path in paths if "labels" in path]

paths = glob.glob("./datasetBothModels/distillBert/batches/*.npy")
batches = [path for path in paths if "labels" not in path]
labels = [path for path in paths if "labels" in path]

params = {
    "learningRate": 0.01,
    "optimizer": "Adamax"
}

distillModel = Model().cuda()
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adamax(distillModel.parameters(), lr = params["learningRate"])

def testModelBert():

  distillModel.eval()

  predicted = []
  truth = []

  for batchPath, labelPath in zip(testBatches, testLabels):

    x2 = np.load(batchPath)

    y2 = np.load(labelPath)
    y2 = y2.reshape(-1, 1)

    y = torch.tensor(y2).float()
    y = y.cuda()

    x = torch.tensor(x2)
    x = x.cuda()

    with torch.no_grad():
      y_pred = distillModel(x)

    y = y.cpu()
    y_pred = y_pred.cpu()

    y_pred = torch.where(y_pred <= 0.5,  0, 1)
    predicted = torch.cat((torch.tensor(predicted), y_pred))
    truth = torch.cat((torch.tensor(truth), y))

  acc = (truth == predicted).sum().float()/len(truth)

  return acc

epochs = 50

predicted = []
truth = []
epochLosses = []

for i in range(epochs):

  # if len(losses) > 50:
  #   if losses[-1] <= 0.15:
  #     optimizer = torch.optim.Adam(distillModel.parameters(), lr = (params["learningRate"] / 10))

  losses = []
  distillModel.train()
  for batchPath, labelPath in zip(batches, labels):
    # Batches are no longer split into 4 as distill bert was saved in batches of 256 already

    # print(f"batch: {batchPath[-17:]}")
    # print(f"batch: {labelPath[-17:]}")

    x2 = np.load(batchPath)
    #x = x / np.linalg.norm(x)

    y2 = np.load(labelPath)
    y2 = y2.reshape(-1, 1)

    y = torch.tensor(y2).float()
    y = y.cuda()

    x = torch.tensor(x2)
    x = x.cuda()

    y_pred = distillModel(x)

    #print(f"{x.size()} | {y.size()}")
    #print(f"yPred {y_pred}")
    #print(f"y {y}")
    #print(y_pred)
    #print(y)
    #print(torch.max(y_pred, 1)[1])
    # print(f"{y_pred.size()} | {y.size()}")
    # print(f"{y_pred.type()}) | {y.type()}")

    #print(y_pred.device)
    #print(y.device)

    loss = criterion(y_pred, y)

    losses.append(loss.cpu().detach().numpy())

    #print(f"{y_pred.size()} | {y.size()}")
    #print(f"{y_pred[y_pred == 1]} \n {y[:10]}")
    #print(f"{y_pred[y_pred == 0]}")
    #print(f"loss: {loss}")
    #print((y==y_pred))
    #print(predicted)
    #print(y_pred)

    #loss = loss.cpu()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    y = y.cpu()
    y_pred = y_pred.cpu()

    y_pred = torch.where(y_pred <= 0.5,  0, 1)
    predicted = torch.cat((torch.tensor(predicted), y_pred))
    truth = torch.cat((torch.tensor(truth), y))
    # print(f"y_pred: {y_pred.shape}")
    # print(f"y: {y.shape}")
    # print(f"predicted: {predicted.shape}")
    # print(f"truth: {truth.shape}")

  acc = (truth == predicted).sum().float()/len(truth)

  valAcc = testModelBert()

  epochLosses.append(np.array(losses).mean())

  #losses = torch.cat((torch.tensor(losses), torch.tensor(loss.cpu().item())))

  print(f"Epoch {i} | Loss {loss.item()} | Accuracy {acc} | Validation Accuracy {valAcc}")

modelName = f"DISTILLBERT-{round(acc.item(), 3)}-{round(valAcc.item(), 3)}-{round(epochLosses[-1].item(), 3)}-{epochs}-{params['learningRate']}-{params['optimizer']}"

folderPath = f"./models/{modelName}/"

if not os.path.exists(folderPath):
    os.makedirs(folderPath)

torch.save(distillModel, f"{folderPath}{modelName}.pt")
np.save(f"{folderPath}/losses.npy", np.array(epochLosses))