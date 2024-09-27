# Code written as part of NFRP-22-41-55 (1st March 2023 - 30 April 2024)
# =======================================================================

import sklearn
import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot as py

################################################################################################################
##################################################### LR GPU ###################################################

# Implementation of LR with Encoding and Normalization 
# Training of the model on Dataset1 ( 2008 - 2020)
# Train (75 %), Test (25 %)
################################################################################################################
import cudf
import cuml
import time
import cupy as cp
import pandasas pd
from cuml.linear_model import LogisticRegression
from cuml.datasets import make_classification
from cuml.metrics import accuracy_score, confusion_matrix
from cuml.model_selection import train_test_split
from cuml.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import chi2

df = cudf.read_csv(r"kronodroid-dataset-2008-to-2020.csv")
df.head()

df.shape
df.drop(columns=['Package'], inplace=True)
df.drop(columns=['MalFamily'], inplace=True)
df.drop(columns=['Categories'], inplace=True)
df.drop(columns=['sha256'], inplace=True)
df.drop(columns=['Scanners'], inplace=True)
df.drop(columns=['Detection_Ratio'], inplace=True)
df.drop(columns=['TimesSubmitted'], inplace=True)
#data.drop(columns=['EarliestModDate'], inplace=True)   #already deleted
#data.drop(columns=['HighestModDate'], inplace=True)
df.drop(columns=['Highest-date'], inplace=True)
df.drop(columns=['Year'], inplace=True)
df.drop(columns=['year_month'], inplace=True)
df.drop(columns=['NrContactedIps'], inplace=True)

#Encoding
df2 = cudf.get_dummies(df)
df2.head()

X = df2.drop('Malware', axis=1)  # Replace 'target_column' with the name of your target variable
y = df2['Malware']

# SPLIT
train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size = 0.25,
                                                                            random_state = 42)

# NORMALIZATION
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(train_features)
X_test_scaled = scaler.transform(test_features)

# LR model
start_time = time.time()
logreg = LogisticRegression(max_iter = 1000)
logreg.fit(X_train_scaled, train_labels)
end_time = time.time()

# Prediction and accuracy
test_pred = logreg.predict(X_test_scaled)
accuracy = accuracy_score(test_labels, test_pred)
print(f"Accuracy: {accuracy}")
print("Accuracy for model : %.2f" % (accuracy_score(test_labels, test_pred) * 100))
training_time = end_time - start_time
print(f"Training Time: {training_time} seconds")



##############################################################################################################
##############################################################################################################


