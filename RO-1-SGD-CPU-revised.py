# Code written as part of NFRP-22-41-55 (1st March 2023 - 30 April 2024)
# =======================================================================

from matplotlib import pyplot as py
from sklearn.model_selection import train_test_split
import sklearn
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

################################################################################################################################
################################################################################################################################
################################################################################################################################
############ SGD training and testing on data from 2008 - 2020


##### Model Train and Test(2008 - 2020)
data_3 = pd.read_csv(r'D:\NFRP\kronodroid-dataset-2008-to-2020.csv', low_memory=False)
data_3.head()

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
# TRAIN TEST SPLIT
X = data_4.drop('Malware', axis=1)  # Replace 'target_column' with the name of your target variable
y = data_4['Malware']
# NORMALIZATION

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
### NO FS
# Split
from sklearn.model_selection import train_test_split
X_train_nofs, X_test_nofs, y_train_nofs, y_test_nofs = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

import time
import joblib
import os
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
start_time = time.time()
sgd_clf.fit(X_train_nofs, y_train_nofs)
training_time = time.time() - start_time

# Save the trained model to a file
model_filename = 'sgd_model_kronodroid_nofs_ro1.pkl'
joblib.dump(sgd_clf, model_filename)
# Get the size of the saved model
model_file_size_bytes = os.path.getsize(model_filename)
print(f"Model file size (bytes): {model_file_size_bytes}")
model_file_size_megabytes = (model_file_size_bytes / (1024 * 1024))  # Convert bytes to megabytes
print(f"Model file size (MB): {model_file_size_megabytes:} MB")

# Prediction and accuracy
from sklearn.metrics import accuracy_score
pstart_time = time.time()
test_pred = sgd_clf.predict(X_test_nofs)
prediction_time = time.time() - pstart_time
accuracy = accuracy_score(y_test_nofs, test_pred)
print(f"Accuracy: {accuracy}")
print("Accuracy for model : %.2f" % (accuracy_score(y_test_nofs, test_pred) * 100))
# Continue with model evaluation or predictions

print(f"Training Time: {training_time:.4f} seconds")
print(f"Prediction Time: {prediction_time:.4f} seconds")


# confusion matrix, classification report, ROC curve
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
con_matrix = confusion_matrix(y_test_nofs, test_pred)
report = classification_report(y_test_nofs, test_pred)
fpr, tpr, thresholds = roc_curve(y_test_nofs, test_pred)
roc_auc = auc(fpr, tpr)
print("Confusion Matrix:\n", con_matrix)
print("\nClassification Report:\n", report)
print("\nFalse positive rate:\n", fpr)
print("\nROC AUC:", roc_auc)
tn, fp, fn, tp = con_matrix.ravel()
# Calculate False Positive Rate (FPR)
fpr = fp / (fp + tn)
print(f"False Positive Rate: {fpr:.4f}")


################################################################## With FS RO-1
##  FEATURE SELECTION
# Variance Threshold method 
from sklearn.feature_selection import VarianceThreshold

threshold = 0.0117
# Create a VarianceThreshold instance and fit it to your data
vt = VarianceThreshold(threshold=threshold)
X_selected = vt.fit_transform(X_scaled)
selected_feature_mask  = vt.get_support()
num_selected_features = selected_feature_mask.sum()
print("Number of selected features:", num_selected_features)

# Inspect selected features
selected_feature_indices = vt.get_support(indices=True)
selected_feature_names = X.columns[selected_feature_indices]
print("Selected features:", selected_feature_names)

# Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

sgd_clf = SGDClassifier(random_state=42)
start_time = time.time()
sgd_clf.fit(X_train, y_train)
training_time = time.time() - start_time

# Save the trained model to a file
model_filename = 'sgd_model_kronodroid_vt_ro1.pkl'
joblib.dump(sgd_clf, model_filename)
# Get the size of the saved model
model_file_size_bytes = os.path.getsize(model_filename)
print(f"Model file size (bytes): {model_file_size_bytes}")
model_file_size_megabytes = (model_file_size_bytes / (1024 * 1024))  # Convert bytes to megabytes
print(f"Model file size (MB): {model_file_size_megabytes:} MB")

# Prediction and accuracy
from sklearn.metrics import accuracy_score
pstart_time = time.time()
test_pred = sgd_clf.predict(X_test)
prediction_time = time.time() - pstart_time
accuracy = accuracy_score(y_test, test_pred)
print(f"Accuracy: {accuracy}")
print("Accuracy for model : %.2f" % (accuracy_score(y_test, test_pred) * 100))
# Continue with model evaluation or predictions
print(f"Training Time: {training_time:.4f} seconds")
print(f"Prediction Time: {prediction_time:.4f} seconds")


# confusion matrix, classification report, ROC curve
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
con_matrix = confusion_matrix(y_test, test_pred)
report = classification_report(y_test, test_pred)
fpr, tpr, thresholds = roc_curve(y_test, test_pred)
roc_auc = auc(fpr, tpr)
print("Confusion Matrix:\n", con_matrix)
print("\nClassification Report:\n", report)
print("\nFalse positive rate:\n", fpr)
print("\nROC AUC:", roc_auc)
tn, fp, fn, tp = con_matrix.ravel()
# Calculate False Positive Rate (FPR)
fpr = fp / (fp + tn)
print(f"False Positive Rate: {fpr:.4f}")

##### Feature Selection FILTER METHOD ( mutual_info_classif)

########################################################################
################ EMBEDDED METHODS #####################################
################  Regularization  ######################################
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
sgd_clf2 = SGDClassifier(random_state=42)
# Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Using SelectFromModel to select best 10 features
# Use selection on Scaled data
attribute_selector = SelectFromModel(sgd_clf2, max_features= 88)
X_train_selected2 = attribute_selector.fit_transform(X_train, y_train)
X_test_selected2 = attribute_selector.transform(X_test)

# Training on the selected features

from sklearn.linear_model import SGDClassifier
sgd_clf22 = SGDClassifier(random_state=42,penalty='l1' )
start_time2 = time.time()
sgd_clf22.fit(X_train, y_train)
training_time2 = time.time() - start_time2

import joblib
import os
# Save the trained model to a file
model_filename = 'sgd_model_kronodroid_reg_ro1.pkl'
joblib.dump(sgd_clf22, model_filename)
# Get the size of the saved model
model_file_size_bytes = os.path.getsize(model_filename)
print(f"Model file size (bytes): {model_file_size_bytes}")
model_file_size_megabytes = (model_file_size_bytes / (1024 * 1024))  # Convert bytes to megabytes
print(f"Model file size (MB): {model_file_size_megabytes:} MB")


# Prediction and accuracy
from sklearn.metrics import accuracy_score
pstart_time2 = time.time()
test_pred = sgd_clf22.predict(X_test)
prediction_time = time.time() - pstart_time2
accuracy1 = accuracy_score(y_test, test_pred)
print(f"Accuracy: {accuracy1}")
print("Accuracy for model : %.2f" % (accuracy_score(y_test, test_pred) * 100))
# Continue with model evaluation or predictions
print(f"Training Time: {training_time2:.4f} seconds")
print(f"Prediction Time: {prediction_time:.4f} seconds")

#Evaluation of model2019
# confusion matrix, classification report, ROC curve
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
con_matrix = confusion_matrix(y_test, test_pred)
report = classification_report(y_test, test_pred)
fpr, tpr, thresholds = roc_curve(y_test, test_pred)
roc_auc = auc(fpr, tpr)
print("Confusion Matrix:\n", con_matrix)
print("\nClassification Report:\n", report)
print("\nFalse positive rate:\n", fpr)
print("\nROC AUC:", roc_auc)
tn, fp, fn, tp = con_matrix.ravel()
# Calculate False Positive Rate (FPR)
fpr = fp / (fp + tn)
print(f"False Positive Rate: {fpr:.4f}")

