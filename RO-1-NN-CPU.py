import sklearn
import os
import json
import pickle
import random
import time
import numpy as np
import pandas as pd
import sklearn.metrics
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import BatchNormalization

seed_value = 42
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
random.seed(seed_value)

data = pd.read_csv(r"D:\NFRP\kronodroid-combined-2008-to-2020.csv")
data.head()

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
data =  pd.get_dummies(data)
data.head()

X = data.drop('Malware', axis=1)  # Replace 'target_column' with the name of your target variable
y = data['Malware']

# Split
train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size = 0.25, 
                                                                            random_state = 42)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(train_features)
X_test_scaled = scaler.transform(test_features)

# NN model architecture
model = Sequential()
model.add(Dense(10, input_dim = X_train_scaled.shape[1], activation='relu', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Dense(7, activation='relu', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))

start_time = time.time()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
epochs = 10
batch_size = 32
model.fit(X_train_scaled, train_labels, epochs=epochs, batch_size=batch_size, verbose = 2)
training_time = time.time() - start_time
print(f"Training Time: {training_time} seconds")
model.save("krono_DL_model.keras")

# Predictions
p_start_time = time.time()
predictions = model.predict(X_test_scaled)
prediction_time = time.time() - p_start_time
print(f"Prediction Time: {prediction_time} seconds")
binary_predictions = (predictions > 0.5).astype(int)
accuracy = accuracy_score(test_labels,binary_predictions)
print("Accuracy:", accuracy)
print("Accuracy for model : %.2f" % (accuracy_score(test_labels,binary_predictions) * 100))
test_loss, test_accuracy = model.evaluate(X_test_scaled, test_labels)
print("Test accuracy:", test_accuracy)

# Model size
loaded_model = load_model("krono_DL_model.keras")
file_size_bytes = os.path.getsize('krono_DL_model.keras')
file_size_kb = file_size_bytes / 1024
file_size_mb = file_size_kb / 1024
print(f"Saved model file size: {file_size_kb:.2f} KB / {file_size_mb:.2f} MB")

#Evaluation of model2019
# confusion matrix, classification report, ROC curve
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
con_matrix = confusion_matrix(test_labels, binary_predictions)
report = classification_report(test_labels, binary_predictions)
fpr, tpr, thresholds = roc_curve(test_labels, binary_predictions)
roc_auc = auc(fpr, tpr)
print("Confusion Matrix:\n", con_matrix)
print("\nClassification Report:\n", report)
print("\nFalse positive rate:\n", fpr)
print("\nROC AUC:", roc_auc)
tn, fp, fn, tp = con_matrix.ravel()
fpr = fp / (fp + tn)
print(f"False Positive Rate: {fpr:.4f}")


### Feature Selection
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

k_best = SelectKBest(score_func= chi2, k=78)
X_train_selected = k_best.fit_transform(X_train_scaled, train_labels)
X_test_selected = k_best.transform(X_test_scaled)

# NN model architecture
model = Sequential()
model.add(Dense(10, input_dim = X_train_selected.shape[1], activation='relu', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Dense(7, activation='relu', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))

fs_start_time = time.time()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
epochs = 10
batch_size = 32
model.fit(X_train_selected, train_labels, epochs=epochs, batch_size=batch_size, verbose = 2)
training_time = time.time() - fs_start_time
print(f"Training Time FS: {training_time} seconds")

model.save("krono_DL_model2.keras")

fs_p_start_time = time.time()
pred = model.predict(X_test_selected)
prediction_time = time.time() - fs_p_start_time
print(f"Prediction Time: {prediction_time} seconds")

binary_pred = (pred > 0.5).astype(int)
accuracy = accuracy_score(test_labels,binary_pred)
print("Accuracy:", accuracy)
print("Accuracy for model with FS : %.2f" % (accuracy_score(test_labels,binary_pred) * 100))
test_loss, test_accuracy = model.evaluate(X_test_selected, test_labels)
print("Test accuracy:", test_accuracy)

# Model size
loaded_model = load_model("krono_DL_model2.keras")
file_size_bytes = os.path.getsize('krono_DL_model2.keras')
file_size_kb = file_size_bytes / 1024
file_size_mb = file_size_kb / 1024
print(f"Saved model file size: {file_size_kb:.2f} KB / {file_size_mb:.2f} MB")

# confusion matrix, classification report, ROC curve
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
con_matrix = confusion_matrix(test_labels, binary_pred)
report = classification_report(test_labels, binary_pred)
fpr, tpr, thresholds = roc_curve(test_labels, binary_pred)
roc_auc = auc(fpr, tpr)
print("Confusion Matrix:\n", con_matrix)
print("\nClassification Report:\n", report)
print("\nFalse positive rate:\n", fpr)
print("\nROC AUC:", roc_auc)
tn, fp, fn, tp = con_matrix.ravel()
# Calculate False Positive Rate (FPR)
fpr = fp / (fp + tn)
print(f"False Positive Rate: {fpr:.4f}")


## Feature Selection
# Principal Component Analysis
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Apply PCA for dimensionality reduction
pca = PCA(n_components=78)  # Set the desired number of components
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# NN model architecture
model = Sequential()
model.add(Dense(10, input_dim = X_train_pca.shape[1], activation='relu', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Dense(7, activation='relu', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))

# Compile the model
pca_start_time = time.time()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
epochs = 10
batch_size = 32
model.fit(X_train_pca, train_labels, epochs=epochs, batch_size=batch_size, verbose = 2)
training_time = time.time() - pca_start_time
print(f"Training Time FS: {training_time} seconds")

model.save("krono_DL_model_pca.keras")

# Prediction
pca_p_start_time = time.time()
pred = model.predict(X_test_pca)
prediction_time = time.time() - pca_p_start_time
print(f"Prediction Time: {prediction_time} seconds")

binary_pred_pca = (pred > 0.5).astype(int)
accuracy = accuracy_score(test_labels,binary_pred_pca)
print("Accuracy:", accuracy)
print("Accuracy for model with FS : %.2f" % (accuracy_score(test_labels,binary_pred_pca) * 100))
# Evaluate on the test set
accuracy = model.evaluate(X_test_pca, test_labels)[1]
print("Test Accuracy:", accuracy)

# Model size
loaded_model = load_model("krono_DL_model_pca.keras")
file_size_bytes = os.path.getsize('krono_DL_model_pca.keras')
file_size_kb = file_size_bytes / 1024
file_size_mb = file_size_kb / 1024
print(f"Saved model file size: {file_size_kb:.2f} KB / {file_size_mb:.2f} MB")

# confusion matrix, classification report, ROC curve
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
con_matrix = confusion_matrix(test_labels, binary_pred_pca)
report = classification_report(test_labels, binary_pred_pca)
fpr, tpr, thresholds = roc_curve(test_labels, binary_pred_pca)
roc_auc = auc(fpr, tpr)
print("Confusion Matrix:\n", con_matrix)
print("\nClassification Report:\n", report)
print("\nFalse positive rate:\n", fpr)
print("\nROC AUC:", roc_auc)
tn, fp, fn, tp = con_matrix.ravel()
# Calculate False Positive Rate (FPR)
fpr = fp / (fp + tn)
print(f"False Positive Rate: {fpr:.4f}")

######################################################################################################
