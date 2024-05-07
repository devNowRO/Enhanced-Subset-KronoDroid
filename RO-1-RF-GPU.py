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

import cuspatiala
cuspatial.__version__

import cuxfilter
cuxfilter.__version__



import cudf

# Commented out IPython magic to ensure Python compatibility.
from google.colab import drive
drive.mount('/content/drive')

# %cd /content/drive/My Drive/

import pandas as pd
from sklearn.metrics import accuracy_score
from cuml.preprocessing import MinMaxScaler
from cuml.ensemble import RandomForestClassifier
from cuml.metrics import accuracy_score
import time

df = pd.read_csv(r"kronodroid-combined-2008-to-2020.csv")
# Assuming you have loaded your data into a DataFrame named df
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
# Create a Min-Max scaler
scaler = MinMaxScaler()

# Fit the scaler on your training data and transform both training and testing data
X_train_scaled = scaler.fit_transform(train_features)
X_test_scaled = scaler.transform(test_features)



#rf_model_gpu = RandomForestClassifier(n_bins = 64, bootstrap=False, n_estimators=150, max_depth=90, random_state=42,  n_streams=1)
rf_model_gpu = RandomForestClassifier(n_estimators=55)
start_time_rf = time.time()
rf_model_gpu.fit(X_train_scaled, train_labels)
#rf_model.fit(X_train_scal, y_train)

training_time_rf = time.time()- start_time_rf

import joblib
import os
# Save the trained model to a file
model_filename_rf = 'RF_model_gpu.pkl'
joblib.dump(rf_model_gpu, model_filename_rf)

# Get the size of the saved model
model_file_size_bytes_rf = os.path.getsize(model_filename_rf)
print(f"Model file size (bytes): {model_file_size_bytes_rf}")

model_file_size_megabytes_rf = (model_file_size_bytes_rf / (1024 * 1024))  # Convert bytes to megabytes

print(f"Model file size (MB): {model_file_size_megabytes_rf:} MB")

# Prediction and accuracy
from sklearn.metrics import accuracy_score
pstart_time_rf = time.time()
test_pred_RF = rf_model_gpu.predict(X_test_scaled)
prediction_time_rf = time.time() - pstart_time_rf
accuracy = accuracy_score(test_labels, test_pred_RF)
print(f"Accuracy: {accuracy}")
print("Accuracy for model : %.2f" % (accuracy_score(test_labels, test_pred_RF) * 100))
# Continue with model evaluation or predictions

print(f"Training Time: {training_time_rf:.4f} seconds")
print(f"Prediction Time: {prediction_time_rf:.4f} seconds")

#Evaluation of model2019
# confusion matrix, classification report, ROC curve
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
con_matrix_rf = confusion_matrix(test_labels, test_pred_RF)
report_rf = classification_report(test_labels, test_pred_RF)
fpr1, tpr1, thresholds1 = roc_curve(test_labels, test_pred_RF)
roc_auc1 = auc(fpr1, tpr1)
print("Confusion Matrix:\n", con_matrix_rf)
print("\nClassification Report:\n", report_rf)
print("\nFalse positive rate:\n", fpr1)
print("\nROC AUC:", roc_auc1)

tn1, fp1, fn1, tp1 = con_matrix.ravel()

# Calculate False Positive Rate (FPR)
fpr1 = fp1 / (fp1 + tn1)

print(f"False Positive Rate: {fpr1:.4f}")


