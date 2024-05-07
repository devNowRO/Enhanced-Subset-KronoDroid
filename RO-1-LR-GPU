import sklearn
import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot as py
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression


################################################################################################################
##################################################### LR GPU ###################################################

# Implementation of LR with Encoding and Normalization (Time stamps not included in encoding)
# With Feature Selection
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

df = cudf.read_csv(r"kronodroid-combined-2008-to-2020.csv")
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
#df.drop(columns=['year_month'], inplace=True) #kept timestamp
# We have kept the column year_month

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
logreg = LogisticRegression(max_iter = 200)
logreg.fit(X_train_scaled, train_labels)
end_time = time.time()

# Prediction and accuracy
test_pred = logreg.predict(X_test_scaled)
accuracy = accuracy_score(test_labels, test_pred)
print(f"Accuracy: {accuracy}")
print("Accuracy for model : %.2f" % (accuracy_score(test_labels, test_pred) * 100))
training_time = end_time - start_time
print(f"Training Time: {training_time} seconds")

########## LR implementation on GPU ################
########## with FEATURE SELECTION ##################
dff = cudf.DataFrame(df2) #df2 encoded dataframe
pd_data = dff.to_pandas()
pd_data = pd.DataFrame(pd_data)

pd1 = pd_data.drop("Malware", axis = 1)
pd2 = pd_data["Malware"]

from sklearn.model_selection import train_test_split
trf, tsf, trl, tsl = train_test_split(pd1, pd2, test_size = 0.25, random_state = 42)

attribute_selector = SelectKBest(score_func= chi2, k=100)
X_train_selected = attribute_selector.fit_transform(trf, trl)
X_test_selected = attribute_selector.transform(tsf)

# Create a Min-Max scaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)

# LR model
start_time = time.time()
LR_model = LogisticRegression(penalty = "l1", max_iter = 1000)
LR_model.fit(X_train_scaled,trl)
end_time = time.time()

# Prediction and accuracy
test_pred = LR_model.predict(X_test_scaled)
accuracy = accuracy_score(tsl, test_pred)
print(f"Accuracy: {accuracy}")
print("Accuracy for model : %.2f" % (accuracy_score(tsl, test_pred) * 100))
training_time = end_time - start_time
print(f"Training Time: {training_time} seconds")


# EMBEDDED METHOD

dff = cudf.DataFrame(df2) #df2 encoded dataframe
X = dff.drop("Malware", axis = 1)
y = dff["Malware"]
X = X.astype('float32')
y = y.astype('float32')

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#  Regularization 
logreg = LogisticRegression( penalty ='l1', max_iter= 1000)
# Convert cuDF objects (scaled data)  to NumPy arrays
X_train_np = X_train_scaled.to_pandas().values
y_train_np = y_train.to_pandas().values
X_test_np = X_test_scaled.to_pandas().values

# Using SelectFromModel to select best 100 features
from sklearn.feature_selection import SelectFromModel
attribute_selector = SelectFromModel(logreg, threshold = "median",  max_features= 100)
X_train_selected2 = attribute_selector.fit_transform(X_train_np, y_train_np)
X_test_selected2 = attribute_selector.transform(X_test_np)

# Display selected features
selected_features = attribute_selector.get_support()
# Get the indices of selected features
selected_feature_indices = [i for i, selected in enumerate(selected_features) if selected]
print("Selected Features:")
print(selected_feature_indices)
print("Number of Selected Features:", len(selected_feature_indices))

# LR model
start_time = time.time()
#Logistic Regression Model on selected data
LR_model3 = LogisticRegression( max_iter = 1000)
LR_model3.fit(X_train_selected2,y_train)
end_time = time.time()

# Prediction and accuracy
test_pred3 = LR_model3.predict(X_test_selected2)
accuracy = accuracy_score(y_test, test_pred3)
print(f"Accuracy: {accuracy}")
print("Accuracy for model : %.2f" % (accuracy_score(y_test, test_pred3) * 100))
training_time = end_time - start_time
print(f"Training Time: {training_time} seconds")

##############################################################################################################
##############################################################################################################


