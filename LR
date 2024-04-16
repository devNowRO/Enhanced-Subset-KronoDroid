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
# Implementation of LR with Encoding and Normalization (Time stamps not included in encoding)
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


# Encoding
data2 =  pd.get_dummies(data)
data2.head()

# Seperating target column and features
X = data2.drop('Malware', axis=1)  # Replace 'target_column' with the name of your target variable
y = data2['Malware']

# Training and testing data
train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size = 0.25, 
                                                                            random_state = 42)
# Scaling (Min-Max scaler)
scaler = MinMaxScaler()
# Fit the scaler on your training data and transform both training and testing data
X_train_scaled = scaler.fit_transform(train_features)
X_test_scaled = scaler.transform(test_features)


# Train the logistic regression model
logreg = LogisticRegression(solver = 'lbfgs', max_iter = 500)
logreg.fit(X_train_scaled, train_labels)

# Prediction and accuracy
test_pred = logreg.predict(X_test_scaled)
accuracy = accuracy_score(test_labels, test_pred)
print(f"Accuracy: {accuracy}")
print("Accuracy for model : %.2f" % (accuracy_score(test_labels, test_pred) * 100))



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


################################################################################################################################
################################################################################################################################
# Implementation of LR with Encoding and Normalization (Time stamps not included in encoding)
# No Feature Selection
# Training of the model on Dataset1 ( 2008 - 2020)
# Train (75 %), Test (25 %)
################################################################################################################################
################################################################################################################################


############ LR training and testing on data from 2008 - 2020   ####################################################
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
# Trin test split (75 %, 25 %)
train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size = 0.25, 
# Scaling                                                                            random_state = 42)
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
# Implementation of LR with Encoding and Normalization (Time stamps not included in encoding)
# With Feature Selection
# Training of the model on Dataset1 ( 2008 - 2020)
# Train (75 %), Test (25 %)
################################################################################################################################
################################################################################################################################

data_3 = pd.read_csv(r'D:\NFRP\kronodroid-combined-2008-to-2020.csv', low_memory=False)
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
# We have kept the column year_month
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
# Define a threshold (e.g., 0.01) for variance
threshold = 0.0162
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
# Filter Method ( SelectkBest)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import chi2
k = 40  # Slecet top 40 features
feature_selector = SelectKBest(score_func= mutual_info_classif, k= k)
X_selected2 = feature_selector.fit_transform(X_scaled, y)
# Train test split
X_train, X_test, y_train, y_test = train_test_split(X_selected2, y, test_size=0.2, random_state=42)

# LOGISTIC REGRESSION
logreg = LogisticRegression(solver = 'lbfgs', max_iter = 500)
logreg.fit(X_train, y_train)
# Prediction and accuracy
test_pred = logreg.predict(X_test)
accuracy = accuracy_score(y_test, test_pred)
print(f"Accuracy: {accuracy}")
print("Accuracy for model : %.2f" % (accuracy_score(y_test, test_pred) * 100))

accuracy = accuracy_score(y_test, test_pred)
confusion = confusion_matrix(y_test, test_pred)
classification_rep = classification_report(y_test, test_pred)
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion)
print("Classification Report:\n", classification_rep)

# FEATURE SELECTION
# Embedded Method ( Regularization)
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(penalty='l1', solver='liblinear')
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# SelectFromModel
attribute_selector = SelectFromModel(logreg, max_features= 80)
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

################################################################################################################

##################################################### LR GPU ###################################################
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
