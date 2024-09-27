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
# Implementation of LR with Encoding and Normalization 
# No Feature Selection
# Training of the model on Dataset1 ( 2008 - 2020)
# Train (75 %), Test (25 %)
################################################################################################################################
################################################################################################################################


############ LR training and testing on data from 2008 - 2020   ####################################################
df = pd.read_csv(r"D:\NFRP\kronodroid-dataset-2008-to-2020.csv")
df.head()
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

# ENCODING
df2 =  pd.get_dummies(df)
df2.head()
X = df2.drop('Malware', axis=1)  
y = df2['Malware']
# Trin test split (75 %, 25 %)
train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size = 0.25, random_state = 42)
# Scaling                                                                            
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(train_features)
X_test_scaled = scaler.transform(test_features)

###### LR model
import time
logreg = LogisticRegression(solver = 'lbfgs', max_iter = 500)
start_time = time.time()
logreg.fit(X_train_scaled, train_labels)
training_time = time.time() - start_time
# Prediction and accuracy 
pstart_time = time.time()
test_pred = logreg.predict(X_test_scaled)
prediction_time = time.time() - pstart_time
accuracy = accuracy_score(test_labels, test_pred)
print(f"Accuracy: {accuracy}")
print("Accuracy for model : %.2f" % (accuracy_score(test_labels, test_pred) * 100))
print(f"Training Time: {training_time:.4f} seconds")
print(f"Prediction Time: {prediction_time:.4f} seconds")

joblib.dump(logreg, 'linear_regression_model.pkl')
file_size_bytes = os.path.getsize('linear_regression_model.pkl')
file_size_kb = file_size_bytes / 1024
file_size_mb = file_size_kb / 1024
print(f"Saved model file size: {file_size_kb:.2f} KB / {file_size_mb:.2f} MB")

con_matrix = confusion_matrix(test_labels, test_pred)
report = classification_report(test_labels, test_pred)
fpr, tpr, thresholds = roc_curve(test_labels, test_pred)
roc_auc = auc(fpr, tpr)
print("Confusion Matrix:\n", con_matrix)
print("\nClassification Report:\n", report)
print("\nFalse positive rate:\n", fpr)
print("\nROC AUC:", roc_auc)

# Heatmap
import seaborn as sns
sns.heatmap(con_matrix, annot=True, fmt='d')
sns.heatmap(con_matrix/np.sum(con_matrix), annot=True, 
            fmt='.2%', cmap='Blues')
group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ["{0:0.0f}".format(value) for value in
                con_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     con_matrix.flatten()/np.sum(con_matrix)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(con_matrix, annot=labels, fmt="", cmap='Blues')
tn, fp, fn, tp = con_matrix.ravel()
fpr = fp / (fp + tn)
print(f"False Positive Rate: {fpr:.4f}")



################################################################################################################################
################################################################################################################################
# Implementation of LR with Encoding and Normalization 
# Training of the model on Dataset1 ( 2008 - 2020)
# Train (75 %), Test (25 %)
################################################################################################################################
################################################################################################################################

data_3 = pd.read_csv(r'D:\NFRP\kronodroid-dataset-2008-to-2020.csv', low_memory=False)
data_3.iloc[:, 480:486]
data_3.drop(columns=['Package'], inplace=True)
data_3.drop(columns=['MalFamily'], inplace=True)
data_3.drop(columns=['Categories'], inplace=True)
data_3.drop(columns=['sha256'], inplace=True)
data_3.drop(columns=['Scanners'], inplace=True)
data_3.drop(columns=['Detection_Ratio'], inplace=True)
data_3.drop(columns=['TimesSubmitted'], inplace=True)
#data.drop(columns=['EarliestModDate'], inplace=True)   #already deleted
#data.drop(columns=['HighestModDate'], inplace=True)
data_3.drop(columns=['Highest-date'], inplace=True)
data_3.drop(columns=['Year'], inplace=True)
data_3.drop(columns=['year_month'], inplace=True)
data_3.drop(columns=['NrContactedIps'], inplace=True)
#  ENCODING
data_4 =  pd.get_dummies(data_3)
data_4.head()
X = data_4.drop('Malware', axis=1)  # Replace 'target_column' with the name of your target variable
y = data_4['Malware']
# NORMALIZATION
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


# FEATURE SELECTION
# Variance Threshold method 
from sklearn.feature_selection import VarianceThreshold
threshold = 0.0117
vt = VarianceThreshold(threshold=threshold)
X_selected = vt.fit_transform(X_scaled)
selected_feature_mask  = vt.get_support()
num_selected_features = selected_feature_mask.sum()
print("Number of selected features:", num_selected_features)

# Inspect selected features
selected_feature_indices = vt.get_support(indices=True)
selected_feature_names = X.columns[selected_feature_indices]
print("Selected features:", selected_feature_names)

X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
# TRain model
logreg = LogisticRegression(solver = 'lbfgs', max_iter = 500)
start_time = time.time()
logreg.fit(X_train, y_train)
training_time = time.time() - start_time
# Model size
model_filename = 'lr_model_kronodroid_1_vt.pkl'
joblib.dump(logreg, model_filename)
model_file_size_bytes = os.path.getsize(model_filename)
print(f"Model file size (bytes): {model_file_size_bytes}")
model_file_size_megabytes = (model_file_size_bytes / (1024 * 1024))  # Convert bytes to megabytes
print(f"Model file size (MB): {model_file_size_megabytes:} MB")

# Prediction and accuracy
pstart_time = time.time()
test_pred = logreg.predict(X_test)
prediction_time = time.time() - pstart_time
accuracy = accuracy_score(y_test, test_pred)
print(f"Accuracy: {accuracy}")
print("Accuracy for model : %.2f" % (accuracy_score(y_test, test_pred) * 100))
print(f"Training Time: {training_time:.4f} seconds")
print(f"Prediction Time: {prediction_time:.4f} seconds")

# Confusion matrix, ROC, FPR
con_matrix = confusion_matrix(y_test, test_pred)
report = classification_report(y_test, test_pred)
fpr, tpr, thresholds = roc_curve(y_test, test_pred)
roc_auc = auc(fpr, tpr)
print("Confusion Matrix:\n", con_matrix)
print("\nClassification Report:\n", report)
print("\nFalse positive rate:\n", fpr)
print("\nROC AUC:", roc_auc)
tn, fp, fn, tp = con_matrix.ravel()
fpr = fp / (fp + tn)
print(f"False Positive Rate: {fpr:.4f}")

# FEATURE SELECTION
# Embedded Method ( Regularization)
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(penalty='l1', solver='liblinear')
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# SelectFromModel
attribute_selector = SelectFromModel(logreg, max_features= 88)
X_train_selected2 = attribute_selector.fit_transform(X_train, y_train)
X_test_selected2 = attribute_selector.transform(X_test)

# Training on the selected features
LR_model2 = LogisticRegression(penalty='l1', solver='liblinear')
start_time = time.time()
LR_model2.fit(X_train_selected2, y_train)
training_time = time.time() - start_time

# Prediction and accuracy
pstart_time = time.time()
test_pred3 = LR_model2.predict(X_test_selected2)
prediction_time = time.time() - pstart_time
accuracy = accuracy_score(y_test, test_pred3)
print(f"Accuracy: {accuracy}")
print("Accuracy for model : %.2f" % (accuracy_score(y_test, test_pred3) * 100))
print(f"Training Time: {training_time:.4f} seconds")
print(f"Prediction Time: {prediction_time:.4f} seconds")

# model size
model_filename = 'lr_model_kronodroid_1.pkl'
joblib.dump(LR_model2, model_filename)
model_file_size_bytes = os.path.getsize(model_filename)
print(f"Model file size (bytes): {model_file_size_bytes}")
model_file_size_megabytes = (model_file_size_bytes / (1024 * 1024))  # Convert bytes to megabytes
print(f"Model file size (MB): {model_file_size_megabytes:} MB")

# Confusion matrix, classification report, ROC curve
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
con_matrix = confusion_matrix(y_test, test_pred3)
report = classification_report(y_test, test_pred3)
fpr, tpr, thresholds = roc_curve(y_test, test_pred3)
roc_auc = auc(fpr, tpr)
print("Confusion Matrix:\n", con_matrix)
print("\nClassification Report:\n", report)
print("\nFalse positive rate:\n", fpr)
print("\nROC AUC:", roc_auc)
tn, fp, fn, tp = con_matrix.ravel()
fpr = fp / (fp + tn)
print(f"False Positive Rate: {fpr:.4f}")

####################################################################################################################################
