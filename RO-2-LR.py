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
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression


################################################################################################################################
################################################################################################################################
# CONCEPT DRIFT
# Implementation of LR with Encoding and Normalization (Time stamps not included in encoding)
# No Feature Selection
# Training of the model on Dataset1 ( 2008 - 2018)
# Testing of the model on Dataset2 ( 2019)
# Testing of the model on Dataset3 (2020)
################################################################################################################################
################################################################################################################################

####### TRAIN DATASET (2008-2018)
train_data = pd.read_csv(r'D:\NFRP\combined_mal_ben_shuffled.csv')
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

#ENCODING
D1 =  pd.get_dummies(train_data)
D1.head()
X = D1.drop('Malware', axis=1)  
y = D1['Malware']

# Seperating features and target variable
train_f_D1 = X # Train features
train_l_D1 = y # Train labels

# NORMALIZE
scaler = MinMaxScaler()
train_f_D1_scaled = scaler.fit_transform(train_f_D1)

##########################
#######  TEST DATASET 2019
##########################
test_data = pd.read_csv(r'D:\NFRP\Test-set-2019.csv')
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

#  ENCODING
D2 =  pd.get_dummies(test_data)
X2 = D2.drop('Malware', axis=1)  # Replace 'target_column' with the name of your target variable
y2 = D2['Malware']
test_f_D2 = X2 #Test feature
test_l_D2 = y2 #Test labels

# Scaling
scaler2 = MinMaxScaler()
test_f_D2_scaled = scaler.fit_transform(test_f_D2)


##########################
#######  TEST DATASET 2020
##########################
test_data2 = pd.read_csv(r'D:\NFRP\Test-set-2020.csv')
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

#  ENCODING
D3 =  pd.get_dummies(test_data2)
D3.head()
X3 = D3.drop('Malware', axis=1)  # Replace 'target_column' with the name of your target variable
y3 = D3['Malware']

test_f_D3 = X3 # Test features
test_l_D3 = y3 # Test labels
# Scaling
scaler3 = MinMaxScaler()
test_f_D3_scaled = scaler.fit_transform(test_f_D3)

###### Logistic Regression
# Train the model
logreg = LogisticRegression(solver = 'lbfgs', max_iter = 500)
start_time = time.time()
logreg.fit(train_f_D1_scaled, train_l_D1)
training_time = time.time() - start_time

# Computing size of trained model
import joblib
import os
joblib.dump(logreg, 'lr_1_model.pkl')
file_size_bytes = os.path.getsize('lr_1_model.pkl')
file_size_kb = file_size_bytes / 1024
file_size_mb = file_size_kb / 1024
print(f"Saved model file size: {file_size_kb:.3f} KB / {file_size_mb:.2f} MB")

# Prediction and accuracy ( Test set 2019)
pstart_time = time.time()
pred = logreg.predict(test_f_D2_scaled)
prediction_time = time.time() - pstart_time
accuracy = accuracy_score(test_l_D2, pred)
print(f"Accuracy: {accuracy}")
print("Accuracy for model : %.2f" % (accuracy_score(test_l_D2, pred) * 100))
print(f"Training Time: {training_time:.4f} seconds")
print(f"Prediction Time: {prediction_time:.4f} seconds")

# Confusion matrix, classification report, ROC curve
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
con_matrix = confusion_matrix(test_l_D2, pred)
report = classification_report(test_l_D2, pred)
fpr, tpr, thresholds = roc_curve(test_l_D2, pred)
roc_auc = auc(fpr, tpr)
print("Confusion Matrix:\n", con_matrix)
print("\nClassification Report:\n", report)
print("\nFalse positive rate:\n", fpr)
print("\nROC AUC:", roc_auc)
tn, fp, fn, tp = con_matrix.ravel()
fpr = fp / (fp + tn)
print(f"False Positive Rate: {fpr:.4f}")

# Prediction and accuracy ( Test set 2020)
p20_start_time = time.time()
pred_2020 = logreg.predict(test_f_D3_scaled)
p20_prediction_time = time.time() - p20_start_time
accuracy_2020 = accuracy_score(test_l_D3, pred_2020)
print(f"Accuracy: {accuracy_2020}")
print("Accuracy for model : %.2f" % (accuracy_score(test_l_D3, pred_2020) * 100))
print(f"Training Time: {training_time:.4f} seconds")
print(f"Prediction Time: {p20_prediction_time:.4f} seconds")


################################################################################################################################
################################################################################################################################
# CONCEPT DRIFT 
# Implementation of LR with Encoding and Normalization (Time stamps not included in encoding)
# With Feature Selection
################################################################################################################################
################################################################################################################################

################  Regularization  ######################################
# LR concept drift
logreg = LogisticRegression(penalty='l1', solver='liblinear')
attribute_selector = SelectFromModel(logreg, max_features= 80)
X_train_selected_D1 = attribute_selector.fit_transform(train_f_D1_scaled, train_l_D1)
X_test_selected_19 = attribute_selector.transform(test_f_D2_scaled)
X_test_selected_20 = attribute_selector.transform(test_f_D3_scaled)

# Training on the selected features
LR_model2 = LogisticRegression(penalty='l1', solver='liblinear')
start_time = time.time()
LR_model2.fit(X_train_selected_D1, train_l_D1)
training_time = time.time() - start_time

joblib.dump(LR_model2, 'lr_2_model.pkl')
file_size_bytes = os.path.getsize('lr_2_model.pkl')
file_size_kb = file_size_bytes / 1024
file_size_mb = file_size_kb / 1024
print(f"Saved model file size: {file_size_kb:.2f} KB / {file_size_mb:.2f} MB")

# Prediction and accuracy 2019 (with FS)
p19_start_time = time.time()
test_pred_19 = LR_model2.predict(X_test_selected_19)
prediction_time_19 = time.time() - p19_start_time
accuracy = accuracy_score(test_l_D2, test_pred_19)
print(f"Accuracy: {accuracy}")
print("Accuracy for model 2019 : %.2f" % (accuracy_score(test_l_D2, test_pred_19) * 100))

# Prediction and accuracy 2020 (with FS)
p20_start_time = time.time()
test_pred_20 = LR_model2.predict(X_test_selected_20)
prediction_time_20 = time.time() - p20_start_time
accuracy_20 = accuracy_score(test_l_D3, test_pred_20)
print(f"Accuracy: {accuracy_20}")
print("Accuracy for model  2020: %.2f" % (accuracy_score(test_l_D3, test_pred_20) * 100))

print(f"Training Time: {training_time:.4f} seconds")
print(f"Prediction Time 2019: {prediction_time_19:.4f} seconds")
print(f"Prediction Time 2020: {prediction_time_20:.4f} seconds")

#Evaluation of model2019
# confusion matrix, classification report, ROC curve
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
con_matrix = confusion_matrix(test_l_D2, test_pred_19)
report = classification_report(test_l_D2, test_pred_19)
fpr, tpr, thresholds = roc_curve(test_l_D2, test_pred_19)
roc_auc = auc(fpr, tpr)
print("Confusion Matrix:\n", con_matrix)
print("\nClassification Report:\n", report)
print("\nFalse positive rate:\n", fpr)
print("\nROC AUC:", roc_auc)
tn, fp, fn, tp = con_matrix.ravel()
fpr = fp / (fp + tn)
print(f"False Positive Rate: {fpr:.4f}")

#Evaluation of model2020
#confusion matrix, classification report, ROC curve
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
con_matrix = confusion_matrix(test_l_D3, test_pred_20)
report = classification_report(test_l_D3, test_pred_20)
fpr, tpr, thresholds = roc_curve(test_l_D3, test_pred_20)
roc_auc = auc(fpr, tpr)
print("Confusion Matrix:\n", con_matrix)
print("\nClassification Report:\n", report)
print("\nFalse positive rate:\n", fpr)
print("\nROC AUC:", roc_auc)
tn, fp, fn, tp = con_matrix.ravel()
fpr = fp / (fp + tn)
print(f"False Positive Rate: {fpr:.4f}")


######### LR concept drift wth and without FS completed above

