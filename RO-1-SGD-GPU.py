# Code written as part of NFRP-22-41-55 (1st March 2023 - 30 April 2024)
# =======================================================================

# !nvidia-smi
# This get the RAPIDS-Colab install files and test check your GPU.  Run this and the next cell only.
# Please read the output of this cell.  If your Colab Instance is not RAPIDS compatible, it will warn #you and give you remediation steps.
!git clone https://github.com/rapidsai/rapidsai-csp-utils.git
!python rapidsai-csp-utils/colab/pip-install.py

import cudf
cudf.__version__

import cuml
cuml.__version__

import cugraph
cugraph.__version__

import cuspatial
cuspatial.__version__

import cuxfilter
cuxfilter.__version__

import cudf

from google.colab import drive
drive.mount('/content/drive')

# %cd /content/drive/My Drive/

import pandas as pd
from sklearn.metrics import accuracy_score

df = pd.read_csv(r"kronodroid-combined-2008-to-2020.csv")
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

df.iloc[:, 470:486]
df.shape

# ENCODING
df2 =  pd.get_dummies(df)
df2.head()

# SPLIT
X = df2.drop('Malware', axis=1)  # Replace 'target_column' with the name of your target variable
y = df2['Malware']

from sklearn.model_selection import train_test_split
train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size = 0.25,
                                                                            random_state = 42)
# NORMALIZATION
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(train_features)
X_test_scaled = scaler.transform(test_features)

from cuml.linear_model import MBSGDClassifier
cu_mbsgd_classifier = MBSGDClassifier(batch_size= 18, epochs = 100, eta0=0.004)

import time
start_time = time.time()
cu_mbsgd_classifier.fit(X_train_scaled, train_labels)
training_time = time.time() - start_time

import joblib
import os
# Saving trained model to a file
model_filename = 'sgd_model_gpu.pkl'
joblib.dump(cu_mbsgd_classifier, model_filename)

# Getting size of the saved model
model_file_size_bytes = os.path.getsize(model_filename)
print(f"Model file size (bytes): {model_file_size_bytes}")
model_file_size_megabytes = (model_file_size_bytes / (1024 * 1024))  # Convert bytes to megabytes
print(f"Model file size (MB): {model_file_size_megabytes:} MB")

# Prediction and accuracy
from sklearn.metrics import accuracy_score
pstart_time = time.time()
test_pred = cu_mbsgd_classifier.predict(X_test_scaled)
prediction_time = time.time() - pstart_time
accuracy = accuracy_score(test_labels, test_pred)
print(f"Accuracy: {accuracy}")
print("Accuracy for model : %.2f" % (accuracy_score(test_labels, test_pred) * 100))
# Continue with model evaluation or predictions

print(f"Training Time: {training_time:.4f} seconds")
print(f"Prediction Time: {prediction_time:.4f} seconds")

#Evaluation of model2019
# confusion matrix, classification report, ROC curve
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

# Calculate False Positive Rate (FPR)
fpr = fp / (fp + tn)

print(f"False Positive Rate: {fpr:.4f}")

###################################################################################################3
