# Code written as part of NFRP-22-41-55 (1st March 2023 - 30 April 2024)
# =======================================================================

import sklearn
import os
import json
import pickle
import random
import time
import numpy as np
import pandas as pd
import sklearn.metrics
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import BatchNormalization

seed_value = 42
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
random.seed(seed_value)



################################################################################################################################
################################################################################################################################
# CONCEPT DRIFT 
# Implementation of NN with Encoding and Normalization (Time stamps not included in encoding)
# With and without Feature Selection
################################################################################################################################
################################################################################################################################

train_data = pd.read_csv(r'D:\NFRP\train_dataset_2008_2018.csv')
train_data.head()
train_data.shape

train_data.drop(columns=['Package'], inplace=True)
train_data.drop(columns=['MalFamily'], inplace=True)
train_data.drop(columns=['Categories'], inplace=True)
train_data.drop(columns=['sha256'], inplace=True)
train_data.drop(columns=['Scanners'], inplace=True)
train_data.drop(columns=['Detection_Ratio'], inplace=True)
train_data.drop(columns=['TimesSubmitted'], inplace=True)
#data.drop(columns=['EarliestModDate'], inplace=True)   #already deleted
#data.drop(columns=['HighestModDate'], inplace=True)
train_data.drop(columns=['Highest-date'], inplace=True)
train_data.drop(columns=['Year'], inplace=True)
train_data.drop(columns=['year_month'], inplace=True)
train_data.drop(columns=['NrContactedIps'], inplace=True)


# ENCODING
D1 =  pd.get_dummies(train_data)
D1.head()

X = D1.drop('Malware', axis=1)  # Replace 'target_column' with the name of your target variable
y = D1['Malware']

train_f_D1 = X # Train features
train_l_D1 = y # Train labels

# NORMALIZE
scaler = MinMaxScaler()
train_f_D1_scaled = scaler.fit_transform(train_f_D1)

##########################
#######  TEST DATASET 2019
##########################
test_data = pd.read_csv(r'D:\NFRP\test_dataset_2019.csv')
test_data.head()

test_data.drop(columns=['Package'], inplace=True)
test_data.drop(columns=['MalFamily'], inplace=True)
test_data.drop(columns=['Categories'], inplace=True)
test_data.drop(columns=['sha256'], inplace=True)
test_data.drop(columns=['Scanners'], inplace=True)
test_data.drop(columns=['Detection_Ratio'], inplace=True)
test_data.drop(columns=['TimesSubmitted'], inplace=True)
#data.drop(columns=['EarliestModDate'], inplace=True)   #already deleted
#data.drop(columns=['HighestModDate'], inplace=True)
test_data.drop(columns=['Highest-date'], inplace=True)
test_data.drop(columns=['Year'], inplace=True)
test_data.drop(columns=['year_month'], inplace=True)
test_data.drop(columns=['NrContactedIps'], inplace=True)

#  ENCODING
D2 =  pd.get_dummies(test_data)
D2.head()

X2 = D2.drop('Malware', axis=1)  
y2 = D2['Malware']

test_f_D2 = X2
test_l_D2 = y2

# NORMALIZE
scaler2 = MinMaxScaler()
test_f_D2_scaled = scaler.fit_transform(test_f_D2)

##########################
#######  TEST DATASET 2020
##########################
test_data2 = pd.read_csv(r'D:\NFRP\test_dataset_2020.csv')
test_data2.head()

test_data2.drop(columns=['Package'], inplace=True)
test_data2.drop(columns=['MalFamily'], inplace=True)
test_data2.drop(columns=['Categories'], inplace=True)
test_data2.drop(columns=['sha256'], inplace=True)
test_data2.drop(columns=['Scanners'], inplace=True)
test_data2.drop(columns=['Detection_Ratio'], inplace=True)
test_data2.drop(columns=['TimesSubmitted'], inplace=True)
#data.drop(columns=['EarliestModDate'], inplace=True)   #already deleted
#data.drop(columns=['HighestModDate'], inplace=True)
test_data2.drop(columns=['Highest-date'], inplace=True)
test_data2.drop(columns=['Year'], inplace=True)
test_data2.drop(columns=['year_month'], inplace=True)
test_data2.drop(columns=['NrContactedIps'], inplace=True)

#  ENCODING
D3 =  pd.get_dummies(test_data2)
D3.head()

X3 = D3.drop('Malware', axis=1)  # Replace 'target_column' with the name of your target variable
y3 = D3['Malware']

test_f_D3 = X3
test_l_D3 = y3

# NORMALIZE
scaler3 = MinMaxScaler()
test_f_D3_scaled = scaler.fit_transform(test_f_D3)

# NN model architecture
model = Sequential()
model.add(Dense(10, input_dim = train_f_D1_scaled.shape[1], activation='relu', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Dense(7, activation='relu', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))

# Compile the model
start_time = time.time()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Train the model
epochs = 10
batch_size = 32
model.fit(train_f_D1_scaled,train_l_D1, epochs=epochs, batch_size=batch_size, verbose = 2)
training_time = time.time() - start_time
print(f"Training Time: {training_time} seconds")
model.save("krono_DL_model_2.keras")

model_2 = load_model("krono_DL_model_2.keras")
file_size_bytes = os.path.getsize('krono_DL_model_2.keras')
file_size_kb = file_size_bytes / 1024
file_size_mb = file_size_kb / 1024
print(f"Saved model file size: {file_size_kb:.2f} KB / {file_size_mb:.2f} MB")

#####   TEST SET 2019
# Prediction and accuracy
pstart_time = time.time()
pred = model.predict(test_f_D2_scaled)
prediction_time = time.time() - pstart_time
binary_predictions = (pred > 0.5).astype(int)
accuracy = accuracy_score(test_l_D2,binary_predictions)
print("Accuracy:", accuracy)
print("Accuracy for model : %.2f" % (accuracy_score(test_l_D2,binary_predictions) * 100))

#Evaluation of model 2019
# confusion matrix, classification report, ROC curve
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
con_matrix = confusion_matrix(test_l_D2, binary_predictions)
report = classification_report(test_l_D2, binary_predictions)
fpr, tpr, thresholds = roc_curve(test_l_D2, binary_predictions)
roc_auc = auc(fpr, tpr)
print("Confusion Matrix:\n", con_matrix)
print("\nClassification Report:\n", report)
print("\nFalse positive rate:\n", fpr)
print("\nROC AUC:", roc_auc)
tn, fp, fn, tp = con_matrix.ravel()
fpr = fp / (fp + tn)
print(f"False Positive Rate: {fpr:.4f}")
print(f"Prediction Time: {prediction_time} seconds")

#####   TEST SET 2020
# Prediction and accuracy
pstart_time = time.time()
pred = model.predict(test_f_D3_scaled)
prediction_time = time.time() - pstart_time

binary_pred = (pred > 0.5).astype(int)
accuracy = accuracy_score(test_l_D3,binary_pred)
print("Accuracy:", accuracy)
print("Accuracy for model : %.2f" % (accuracy_score(test_l_D3,binary_pred) * 100))

# confusion matrix, classification report, ROC curve
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
con_matrix = confusion_matrix(test_l_D3, binary_pred)
report = classification_report(test_l_D3, binary_pred)
fpr, tpr, thresholds = roc_curve(test_l_D3, binary_pred)
roc_auc = auc(fpr, tpr)
print("Confusion Matrix:\n", con_matrix)
print("\nClassification Report:\n", report)
print("\nFalse positive rate:\n", fpr)
print("\nROC AUC:", roc_auc)
tn, fp, fn, tp = con_matrix.ravel()
fpr = fp / (fp + tn)
print(f"False Positive Rate: {fpr:.4f}")
print(f"Prediction Time: {prediction_time} seconds")

##################################################################################################