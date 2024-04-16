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
# LR training and testing on data from 2008 - 2018  (75%, 25% )
################################################################################################################################
################################################################################################################################

data = pd.read_csv(r'D:\NFRP\combined_mal_ben_shuffled.csv', low_memory=False)
data.head()
data.iloc[:, 480:486]
data.drop(columns=['Package'], inplace=True)
data.drop(columns=['MalFamily'], inplace=True)
data.drop(columns=['Categories'], inplace=True)
data.drop(columns=['sha256'], inplace=True)
data.drop(columns=['Scanners'], inplace=True)
data.drop(columns=['Detection_Ratio'], inplace=True)
data.drop(columns=['TimesSubmitted'], inplace=True)
#data.drop(columns=['EarliestModDate'], inplace=True)   #already deleted
#data.drop(columns=['HighestModDate'], inplace=True)
data.drop(columns=['Highest-date'], inplace=True)
data.drop(columns=['Year'], inplace=True)
# We have kept the column year_month

# Encoding
data2 =  pd.get_dummies(data)
data2.head()
# Target and features
X = data2.drop('Malware', axis=1)  
y = data2['Malware']
# Train test split
train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size = 0.25, 
                                                                            random_state = 42)
# Scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(train_features)
X_test_scaled = scaler.transform(test_features)

# RF Model
import time
rf1 = RandomForestClassifier(n_estimators = 50)
start_time = time.time()
rf1.fit(X_train_scaled, train_labels)
training_time = time.time() - start_time
# Predictions
pstart_time = time.time()
y_pred1 = rf1.predict(X_test_scaled)
prediction_time = time.time() - pstart_time
# Accuracy
accuracy = accuracy_score(test_labels, y_pred1)
print(f"Accuracy: {accuracy}")
print("Accuracy for model : %.2f" % (accuracy_score(test_labels, y_pred1) * 100))
print(f"Training Time: {training_time:.4f} seconds")
print(f"Prediction Time: {prediction_time:.4f} seconds")

# Model size
import joblib
import os
joblib.dump(rf1, 'rf1_model.pkl')
file_size_bytes = os.path.getsize('rf1_model.pkl')
file_size_kb = file_size_bytes / 1024
file_size_mb = file_size_kb / 1024
print(f"Saved model file size: {file_size_kb:.2f} KB / {file_size_mb:.2f} MB")

# Evaluation of model (Confusion matrix, classification report, ROC curve)
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
con_matrix = confusion_matrix(test_labels, test_pred)
report = classification_report(test_labels, test_pred)
fpr, tpr, thresholds = roc_curve(test_labels, test_pred)
roc_auc = auc(fpr, tpr)
print("Confusion Matrix:\n", con_matrix)
print("\nClassification Report:\n", report)
print("\nFalse positive rate:\n", fpr)
print("\nROC AUC:", roc_auc)
tn, fp, fn, tp = con_matrix.ravel()
fpr = fp / (fp + tn)
print(f"False Positive Rate: {fpr:.4f}")



################################################################################################################################
################################################################################################################################
# CONCEPT DRIFT
# Implementation of RF with Encoding and Normalization (Time stamps not included in encoding)
# No Feature Selection
# Training of the model on Dataset1 ( 2008 - 2018)
# Testing of the model on Dataset2 ( 2019)
# Testing of the odel on Dataset3 (2020)
################################################################################################################################
################################################################################################################################

#############  TRAIN DATASET (2008 -2018)
train_data = pd.read_csv(r'D:\NFRP\combined_mal_ben_shuffled.csv')
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
# CONCEPT DRIFT 
# Implementation of RF with Encoding and Normalization (Time stamps not included in encoding)
# With Feature Selection
################################################################################################################################
################################################################################################################################
#############################################  TRAIN DATASET
train_data_2 = pd.read_csv(r'D:\NFRP\combined_mal_ben_shuffled.csv')
train_data_2.head()
train_data_2.drop(columns=['Package'], inplace=True)
train_data_2.drop(columns=['MalFamily'], inplace=True)
train_data_2.drop(columns=['Categories'], inplace=True)
train_data_2.drop(columns=['sha256'], inplace=True)
train_data_2.drop(columns=['Scanners'], inplace=True)
train_data_2.drop(columns=['Detection_Ratio'], inplace=True)
train_data_2.drop(columns=['TimesSubmitted'], inplace=True)
#data.drop(columns=['EarliestModDate'], inplace=True)   #already deleted
#data.drop(columns=['HighestModDate'], inplace=True)
train_data_2.drop(columns=['Highest-date'], inplace=True)
train_data_2.drop(columns=['Year'], inplace=True)
train_data_2.drop(columns=['year_month'], inplace=True)
# ENCODING
D1 =  pd.get_dummies(train_data_2)
D1.head()
X = D1.drop('Malware', axis=1)  # Replace 'target_column' with the name of your target variable
y = D1['Malware']
train_f_D1 = X #features
train_l_D1 = y #target label

# NORMALIZE
scaler = MinMaxScaler()
train_f_D1_scaled = scaler.fit_transform(train_f_D1)


# Select From Model
from sklearn.feature_selection import SelectFromModel
sel = SelectFromModel(RandomForestClassifier(n_estimators = 55), max_features = 78)
sel.fit(train_f_D1_scaled, train_l_D1)
train_features_selected = sel.transform(train_f_D1_scaled)
test_features_selected_19 = sel.transform(test_f_D2_scaled)
test_features_selected_20 = sel.transform(test_f_D3_scaled)
selected_features_mask = sel.get_support()
num_selected_features = sum(selected_features_mask)
print(f"Number of selected features: {num_selected_features}")

# Model training
rf_classifier_2 = RandomForestClassifier(n_estimators=55)
start_t_time = time.time()
rf_classifier_2.fit(train_features_selected, train_l_D1)
trainingTime = time.time() - start_t_time
# Make predictions on the test data 2019, 2020
pstart_time_19 = time.time()
predictions_19 = rf_classifier_2.predict(test_features_selected_19)
prediction_time_19 = time.time() - pstart_time_19
pstart_time_20 = time.time()
predictions_20 = rf_classifier_2.predict(test_features_selected_20)
prediction_time_20 = time.time() - pstart_time_20


accuracy_19 = accuracy_score(test_l_D2, predictions_19)
print(f"Accuracy: {accuracy_19}")
print("Accuracy for model : %.2f" % (accuracy_score(test_l_D2, predictions_19) * 100))
accuracy_20 = accuracy_score(test_l_D3, predictions_20)
print(f"Accuracy: {accuracy_20}")
print("Accuracy for model : %.2f" % (accuracy_score(test_l_D3, predictions_20) * 100))
print(f"Training Time: {trainingTime:.4f} seconds")
print(f"Prediction Time 2019: {prediction_time_19:.4f} seconds")
print(f"Prediction Time 2020: {prediction_time_20:.4f} seconds")

# Model size
joblib.dump(rf_classifier_2, 'rf_FS_subset_sel.pkl')
file_size_bytes = os.path.getsize('rf_FS_subset_sel.pkl')
file_size_kb = file_size_bytes / 1024
file_size_mb = file_size_kb / 1024
print(f"Saved model file size: {file_size_kb:.2f} KB / {file_size_mb:.2f} MB")

#Evaluation of model 2019
# confusion matrix, classification report, ROC curve
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
con_matrix = confusion_matrix(test_l_D2, predictions_19)
report = classification_report(test_l_D2, predictions_19)
fpr, tpr, thresholds = roc_curve(test_l_D2, predictions_19)
roc_auc = auc(fpr, tpr)
print("Confusion Matrix:\n", con_matrix)
print("\nClassification Report:\n", report)
print("\nFalse positive rate:\n", fpr)
print("\nROC AUC:", roc_auc)

#Evaluation of model 2020
# confusion matrix, classification report, ROC curve
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
con_matrix = confusion_matrix(test_l_D3, predictions_20)
report = classification_report(test_l_D3, predictions_20)
fpr, tpr, thresholds = roc_curve(test_l_D3, predictions_20)
roc_auc = auc(fpr, tpr)
print("Confusion Matrix:\n", con_matrix)
print("\nClassification Report:\n", report)
print("\nFalse positive rate:\n", fpr)
print("\nROC AUC:", roc_auc)

 # Variance Threshold
threshold = 0.010
vt_selector = VarianceThreshold(threshold)

# Fit and transform your training data with the VarianceThreshold selector
X_train_selected = vt_selector.fit_transform(train_f_D1_scaled)
# Apply the same selector to your test data
X_test_selected19 = vt_selector.transform(test_f_D2_scaled)
X_test_selected20 = vt_selector.transform(test_f_D3_scaled)

selected_feature_mask  = vt_selector.get_support()
num_selected_features = selected_feature_mask.sum()
print("Number of selected features:", num_selected_features)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_selected, train_l_D1)
# Predictions
y_pred_19 = rf_classifier.predict(X_test_selected19)
y_pred_20 = rf_classifier.predict(X_test_selected20)
# Accuracy 2019
accuracy4 = accuracy_score(test_l_D2, y_pred_19)
print(f"Accuracy: {accuracy4}")
print("Accuracy for model 2019 : %.2f" % (accuracy_score(test_l_D2, y_pred_19) * 100))
# Accuracy 2020
accuracy5 = accuracy_score(test_l_D3, y_pred_20)
print(f"Accuracy: {accuracy4}")
print("Accuracy for model 2020: %.2f" % (accuracy_score(test_l_D3, y_pred_20) * 100))



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
