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
# Implementation of RF with Encoding and Normalization (Time stamps not included in encoding)
# No Feature Selection
# Training of the model on Dataset1 ( 2008 - 2020)
# Train (75 %), Test (25 %)
################################################################################################################################
################################################################################################################################

############ RF training and testing on data from 2008 - 2020
df = pd.read_csv(r"D:\NFRP\kronodroid-combined-2008-to-2020.csv")
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
# We have kept the column year_month

# ENCODING
df2 =  pd.get_dummies(df)
df2.head()
X = df2.drop('Malware', axis=1)  # Replace 'target_column' with the name of your target variable
y = df2['Malware']

# Split
train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size = 0.25, 
                                                                            random_state = 42)
# NORMALIZATION
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(train_features)
X_test_scaled = scaler.transform(test_features)
# Model training
rf = RandomForestClassifier(n_estimators=50)
start_time = time.time()
rf.fit(X_train_scaled, train_labels)
training_time = time.time() - start_time
# Prediction
pstart_time = time.time()
y_pred = rf.predict(X_test_scaled)
prediction_time = time.time() - pstart_time
# Accuracy
accuracy = accuracy_score(test_labels, y_pred)
print(f"Accuracy: {accuracy}")
print("Accuracy for model : %.2f" % (accuracy_score(test_labels, y_pred) * 100))
print(f"Training Time: {training_time:.4f} seconds")
print(f"Prediction Time: {prediction_time:.4f} seconds")

################################################################################################################################
################################################################################################################################
# Implementation of RF with Encoding and Normalization (Time stamps not included in encoding)
# WITH Feature Selection
# Training of the model on Dataset1 ( 2008 - 2020)
# Train (75 %), Test (25 %)
################################################################################################################################
################################################################################################################################


# FEATURE SELECTION
# Varuance threshold
from sklearn.feature_selection import VarianceThreshold
threshold = 0.017
vt_selector = VarianceThreshold(threshold)
X_train_selected = vt_selector.fit_transform(X_train_scaled)
X_test_selected= vt_selector.transform(X_test_scaled)
selected_feature  = vt_selector.get_support()
num_sel_f = selected_feature.sum()
print("Number of selected features:", num_sel_f)
# Model training on selected features
rf_classifier = RandomForestClassifier(n_estimators=55, random_state=42)
start_vt_time = time.time()
rf_classifier.fit(X_train_selected, train_labels)
train_time = time.time() - start_vt_time
# Predictions
pstart_time = time.time()
y_pred_all = rf_classifier.predict(X_test_selected)
pred_time = time.time() - pstart_time
# Accuracy
accuracy4 = accuracy_score(test_labels, y_pred_all)
print(f"Accuracy: {accuracy4}")
print("Accuracy for model 2019 : %.2f" % (accuracy_score(test_labels, y_pred_all) * 100))
print(f"Training Time: {train_time:.4f} seconds")
print(f"Prediction Time: {pred_time:.4f} seconds")
# Model size
joblib.dump(rf_classifier, 'rf_vt_model.pkl')
file_size_bytes = os.path.getsize('rf_vt_model.pkl')
file_size_kb = file_size_bytes / 1024
file_size_mb = file_size_kb / 1024
print(f"Saved model file size: {file_size_kb:.2f} KB / {file_size_mb:.2f} MB")

# Confusion matrix, classification report, ROC curve
con_matrix = confusion_matrix(test_labels, y_pred_all)
report = classification_report(test_labels, y_pred_all)
fpr, tpr, thresholds = roc_curve(test_labels, y_pred_all)
roc_auc = auc(fpr, tpr)
print("Confusion Matrix:\n", con_matrix)
print("\nClassification Report:\n", report)
print("\nFalse positive rate:\n", fpr)
print("\nROC AUC:", roc_auc)



# FEATURE SELECTION
# SelectFromModel
from sklearn.feature_selection import SelectFromModel
sel = SelectFromModel(RandomForestClassifier(n_estimators = 55), max_features=78)
sel.fit(X_train_scaled, train_labels)
train_f_selected = sel.transform(X_train_scaled)
test_f_selected = sel.transform(X_test_scaled)
selected_features_m = sel.get_support()
# Count the number of selected features (features with 'True' in the mask)
num_selected_f = sum(selected_features_m)
print(f"Number of selected features: {num_selected_f}")
# Create and train a RandomForestClassifier using the selected features
rf_classifier_sel = RandomForestClassifier(n_estimators=55)
start_time_sel = time.time()
rf_classifier_sel.fit(train_f_selected, train_labels)
train_time_sel = time.time() - start_time_sel
# Make predictions 
p_time_sel = time.time()
predictions_sel = rf_classifier_sel.predict(test_f_selected)
pred_time_sel = time.time() - p_time_sel
accuracy_sel = accuracy_score(test_labels, predictions_sel)
print(f"Accuracy: {accuracy_sel}")
print("Accuracy for model : %.2f" % (accuracy_score(test_labels, predictions_sel) * 100))
print(f"Training Time: {train_time_sel:.4f} seconds")
print(f"Prediction Time: {pred_time_sel:.4f} seconds")
# Model size
joblib.dump(rf_classifier_sel, 'rf_sel_model.pkl')
file_size_bytes = os.path.getsize('rf_sel_model.pkl')
file_size_kb = file_size_bytes / 1024
file_size_mb = file_size_kb / 1024
print(f"Saved model file size: {file_size_kb:.2f} KB / {file_size_mb:.2f} MB")
# confusion matrix, classification report, ROC curve
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
con_matrix = confusion_matrix(test_labels, predictions_sel)
report = classification_report(test_labels, predictions_sel)
fpr, tpr, thresholds = roc_curve(test_labels, predictions_sel)
roc_auc = auc(fpr, tpr)
print("Confusion Matrix:\n", con_matrix)
print("\nClassification Report:\n", report)
print("\nFalse positive rate:\n", fpr)
print("\nROC AUC:", roc_auc)

################### 2008-2020 with and without FS complete (SelectFromModel, Variance threshold above)


