# -*- coding: utf-8 -*-
"""LR-RF-GPU.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Py2nuo_uPwv_B1tlMXw7FYeQDcQHIzU0
"""

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

# Commented out IPython magic to ensure Python compatibility.
from google.colab import drive
drive.mount('/content/drive')

# %cd /content/drive/My Drive/

import pandas as pd
from sklearn.metrics import accuracy_score

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
from cuml.preprocessing import MinMaxScaler

# Create a Min-Max scaler
scaler = MinMaxScaler()

# Fit the scaler on your training data and transform both training and testing data
X_train_scaled = scaler.fit_transform(train_features)
X_test_scaled = scaler.transform(test_features)

from cuml.linear_model import LogisticRegression
from cuml.metrics import accuracy_score
import time

import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# CNN model architecture
model = Sequential()
model.add(Dense(10, input_dim = X_train_scaled.shape[1], activation='relu', kernel_initializer='he_normal'))
model.add(Dense(5,activation='relu', kernel_initializer='he_normal'))
model.add(Dense(1, activation='relu'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 10
batch_size = 48
start_time_nn = time.time()
model.fit(X_train_scaled, train_labels, epochs=epochs, batch_size=batch_size, verbose = 2)
train_time_nn = time.time() - start_time_nn
# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test_scaled, test_labels)
print("Test accuracy:", test_accuracy)

pred_time_nn = time.time()

pred = model.predict(X_test_scaled)

prediction_time = time.time() - pred_time_nn
print(f"Training Time: {train_time_nn} seconds")
print(f"Prediction Time: {prediction_time} seconds")

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')

print("GPU:", tf.config.list_physical_devices('GPU'))
print("Num GPUs:", len(physical_devices))

from sklearn.metrics import accuracy_score
binary_pred = (pred > 0.5).astype(int)
accuracy = accuracy_score(test_labels,binary_pred)
print("Accuracy:", accuracy)
print("Accuracy for model NN : %.2f" % (accuracy_score(test_labels,binary_pred) * 100))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test_scaled, test_labels)
print("Test accuracy:", test_accuracy)

model.save("NN_model_gpu.keras")
from tensorflow.keras.models import load_model

loaded_model = load_model("NN_model_gpu.keras")

import os
file_size_bytes = os.path.getsize('NN_model_gpu.keras')
file_size_kb = file_size_bytes / 1024
file_size_mb = file_size_kb / 1024

print(f"Saved model file size: {file_size_kb:.2f} KB / {file_size_mb:.2f} MB")

#Evaluation of model2019
# confusion matrix, classification report, ROC curve
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
con_matrix_nn = confusion_matrix(test_labels, binary_pred)
report_nn = classification_report(test_labels, binary_pred)
fpr3, tpr3, thresholds3 = roc_curve(test_labels, binary_pred)
roc_auc3 = auc(fpr3, tpr3)
print("Confusion Matrix:\n", con_matrix_nn)
print("\nClassification Report:\n", report_nn)
print("\nFalse positive rate:\n", fpr3)
print("\nROC AUC:", roc_auc3)

tn3, fp3, fn3, tp3 = con_matrix_nn.ravel()

# Calculate False Positive Rate (FPR)
fpr3 = fp3 / (fp3 + tn3)

print(f"False Positive Rate: {fpr3:.4f}")

