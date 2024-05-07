# Code written as part of NFRP-22-41-55 (1st March 2023 - 30 April 2024)
# =======================================================================

################################################################################################################################
################################################################################################################################
# # SGD CONCEPT DRIFT 
# Implementation of SGD with Encoding and Normalization (Time stamps not included in encoding)
# Training of the model on Dataset1 ( 2008 - 2018)
# Testing of the model on Dataset2 ( 2019)
# Testing of the odel on Dataset3 (2020)
################################################################################################################################
################################################################################################################################


#############################################  TRAIN DATASET
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


train_data.head()
train_data.shape

# ENCODING
D1 =  pd.get_dummies(train_data)
D1.head()
X = D1.drop('Malware', axis=1)  # Replace 'target_column' with the name of your target variable
y = D1['Malware']


# SPLIT
#train_f_D1, test_f_D1, train_l_D1, test_l_D1 = train_test_split(X, y, test_size = 0.25, random_state = 42)
train_f_D1 = X
train_l_D1 = y
# NORMALIZE
from sklearn.preprocessing import MinMaxScaler
# Create a Min-Max scaler
scaler = MinMaxScaler()
# Fit the scaler on your training data and transform both training and testing data
train_f_D1_scaled = scaler.fit_transform(train_f_D1)


#############################################  TEST DATASET 2019
test_data = pd.read_csv(r'D:\NFRP\Test-set-2019.csv')
test_data.head()
test_data.shape

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
# We have kept the column year_month

test_data.shape
#  ENCODING
D2 =  pd.get_dummies(test_data)
D2.head()
X2 = D2.drop('Malware', axis=1)  # Replace 'target_column' with the name of your target variable
y2 = D2['Malware']
# SPLIT
test_f_D2 = X2
test_l_D2 = y2
# NORMALIZE
from sklearn.preprocessing import MinMaxScaler
# Create a Min-Max scaler
scaler2 = MinMaxScaler()
# Fit the scaler on your training data and transform both training and testing data
test_f_D2_scaled = scaler.fit_transform(test_f_D2)

#############################################  TEST DATASET 2020
test_data2 = pd.read_csv(r'D:\NFRP\Test-set-2020.csv')
test_data2.head()

test_data2.shape
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
# We have kept the column year_month

test_data2.shape
#  ENCODING
D3 =  pd.get_dummies(test_data2)
D3.head()
X3 = D3.drop('Malware', axis=1)  # Replace 'target_column' with the name of your target variable
y3 = D3['Malware']
# SPLIT
test_f_D3 = X3
test_l_D3 = y3
# NORMALIZE
from sklearn.preprocessing import MinMaxScaler
# Create a Min-Max scaler
scaler3 = MinMaxScaler()
# Fit the scaler on your training data and transform both training and testing data
test_f_D3_scaled = scaler.fit_transform(test_f_D3)


############################################# Stochastic Gradient Descent Concept Drift
import time
import joblib
import os
from sklearn.linear_model import SGDClassifier
sgd_clf_cd = SGDClassifier(random_state=42)
start_time_cd = time.time()
sgd_clf_cd.fit(train_f_D1_scaled, train_l_D1)
training_time_cd = time.time() - start_time_cd

# Save the trained model to a file
model_filename = 'sgd_model_kronodroid_nofs_ro1.pkl'
joblib.dump(sgd_clf_cd, model_filename)
# Get the size of the saved model
model_size_bytes = os.path.getsize(model_filename)
print(f"Model file size (bytes): {model_size_bytes}")
model_size_mb = (model_file_size_bytes / (1024 * 1024))  # Convert bytes to megabytes
print(f"Model file size (MB): {model_size_mb:} MB")

###################################### Prediction 2019
# Prediction and accuracy
from sklearn.metrics import accuracy_score
pstart_time_cd = time.time()
test_pred_cd = sgd_clf_cd.predict(test_f_D2_scaled)
prediction_time_cd = time.time() - pstart_time_cd
accuracy = accuracy_score(test_l_D2, test_pred_cd)
print(f"Accuracy: {accuracy}")
print("Accuracy for model : %.2f" % (accuracy_score(test_l_D2, test_pred_cd) * 100))
# Continue with model evaluation or predictions
print(f"Training Time: {training_time_cd:.4f} seconds")
print(f"Prediction Time: {prediction_time_cd:.4f} seconds")
#Evaluation of model2019
# confusion matrix, classification report, ROC curve
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
con_matrix = confusion_matrix(test_l_D2, test_pred_cd)
report = classification_report(test_l_D2, test_pred_cd)
fpr, tpr, thresholds = roc_curve(test_l_D2, test_pred_cd)
roc_auc = auc(fpr, tpr)
print("Confusion Matrix:\n", con_matrix)
print("\nClassification Report:\n", report)
print("\nFalse positive rate:\n", fpr)
print("\nROC AUC:", roc_auc)
tn, fp, fn, tp = con_matrix.ravel()
# Calculate False Positive Rate (FPR)
fpr = fp / (fp + tn)
print(f"False Positive Rate: {fpr:.4f}")

#######################################   TEST SET 2020
# Prediction and accuracy
p20_start_time = time.time()
pred_2020 = sgd_clf_cd.predict(test_f_D3_scaled)
p20_prediction_time = time.time() - p20_start_time
accuracy_2020 = accuracy_score(test_l_D3, pred_2020)
print(f"Accuracy: {accuracy_2020}")
print("Accuracy for model : %.2f" % (accuracy_score(test_l_D3, pred_2020) * 100))
# Continue with model evaluation or predictions
print(f"Training Time: {training_time_cd:.4f} seconds")
print(f"Prediction Time: {p20_prediction_time:.4f} seconds")
#Evaluation of model2020
# confusion matrix, classification report, ROC curve
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
con_matrix = confusion_matrix(test_l_D3, pred_2020)
report = classification_report(test_l_D3, pred_2020)
fpr, tpr, thresholds = roc_curve(test_l_D3, pred_2020)
roc_auc = auc(fpr, tpr)
print("Confusion Matrix:\n", con_matrix)
print("\nClassification Report:\n", report)
print("\nFalse positive rate:\n", fpr)
print("\nROC AUC:", roc_auc)
tn, fp, fn, tp = con_matrix.ravel()
# Calculate False Positive Rate (FPR)
fpr = fp / (fp + tn)
print(f"False Positive Rate: {fpr:.4f}")

#Concept Drift without FS completed above
