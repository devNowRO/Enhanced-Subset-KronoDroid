# Code written as part of NFRP-22-41-55 (1st March 2023 - 30 April 2024)
# =======================================================================

import sklearn
import os
import json
import numpy as np
import pandas as pd
import pickle
import sklearn.metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from xgboost import plot_importance
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz


################################################################################################################################
################################################################################################################################
# CONCEPT DRIFT
# Implementation of RF with Encoding and Normalization 
# No Feature Selection
# Training of the model on Dataset1 ( 2008 - 2018)
# Testing of the model on Dataset2 ( 2019)
# Testing of the odel on Dataset3 (2020)
################################################################################################################################
################################################################################################################################

#############  TRAIN DATASET (2008 -2018)
train_data = pd.read_csv(r'D:\NFRP\train_dataset_2008_2018.csv')
train_data.head()

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
train_f_D1 = X #features
train_l_D1 = y #target label

# Scaling
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
X2 = D2.drop('Malware', axis=1)  # Replace 'target_column' with the name of your target variable
y2 = D2['Malware']
test_f_D2 = X2 # Features
test_l_D2 = y2 # Labels
# Scaling
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
test_f_D3 = X3 # Features 
test_l_D3 = y3 # Labels
# Scaling
scaler3 = MinMaxScaler()
test_f_D3_scaled = scaler.fit_transform(test_f_D3)

####### Random forest (fit data on 2008-2018)
rf_1 = RandomForestClassifier(n_estimators = 55)
start_time = time.time()
rf_1.fit(train_f_D1_scaled, train_l_D1)
training_time = time.time() - start_time
# Model size
joblib.dump(rf_1, 'rf_1_model.pkl')
file_size_bytes = os.path.getsize('rf_1_model.pkl')
file_size_kb = file_size_bytes / 1024
file_size_mb = file_size_kb / 1024
print(f"Saved model file size: {file_size_kb:.2f} KB / {file_size_mb:.2f} MB")

# Prediction and accuracy ( Test set 2019)
pstart_time = time.time()
y_pred_19 = rf_1.predict(test_f_D2_scaled)
prediction_time = time.time() - pstart_time
accuracy = accuracy_score(test_l_D2, y_pred_19)
print(f"Accuracy: {accuracy}")
print("Accuracy for model : %.2f" % (accuracy_score(test_l_D2, y_pred_19) * 100))
print(f"Training Time: {training_time:.4f} seconds")
print(f"Prediction Time: {prediction_time:.4f} seconds")

# Confusion matrix, classification report, ROC curve
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
con_matrix = confusion_matrix(test_l_D2, y_pred_19)
report = classification_report(test_l_D2, y_pred_19)
fpr, tpr, thresholds = roc_curve(test_l_D2, y_pred_19)
roc_auc = auc(fpr, tpr)
print("Confusion Matrix:\n", con_matrix)
print("\nClassification Report:\n", report)
print("\nFalse positive rate:\n", fpr)
print("\nROC AUC:", roc_auc)
tn, fp, fn, tp = con_matrix.ravel()
fpr = fp / (fp + tn)
print(f"False Positive Rate: {fpr:.4f}")

# Prediction and accuracy ( Test set 2020)
pstart_time_20 = time.time()
y_pred_20 = rf_1.predict(test_f_D3_scaled)
prediction_time_20 = time.time() - pstart_time_20
accuracy_20 = accuracy_score(test_l_D3, y_pred_20)
print(f"Accuracy: {accuracy_20}")
print("Accuracy for model : %.2f" % (accuracy_score(test_l_D3, y_pred_20) * 100))
print(f"Training Time: {training_time:.4f} seconds")
print(f"Prediction Time: {prediction_time_20:.4f} seconds")
# Confusion matrix, classification report, ROC curve
con_matrix2 = confusion_matrix(test_l_D3, y_pred_20)
report2 = classification_report(test_l_D3, y_pred_20)
fpr2, tpr2, thresholds2 = roc_curve(test_l_D3, y_pred_20)
roc_auc2 = auc(fpr2, tpr2)
print("Confusion Matrix:\n", con_matrix2)
print("\nClassification Report:\n", report2)
print("\nFalse positive rate:\n", fpr2)
print("\nROC AUC:", roc_auc2)
tn, fp, fn, tp = con_matrix2.ravel()
fprr = fp / (fp + tn)
print(f"False Positive Rate: {fprr:.4f}")

################################################################################################################################
################################################################################################################################
