import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Loading data
url = "ваше_посилання_на_датасет"
data = pd.read_csv(url)

# Suppose that the dataset has columns 'x', 'y', 'z', and 'activity'
X = data[['x', 'y', 'z']]
y = data['activity']

# Separate data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Calculation of timestamp features
def calculate_features(df):
    features = pd.DataFrame()
    features['mean_x'] = df['x'].rolling(window=50).mean()
    features['mean_y'] = df['y'].rolling(window=50).mean()
    features['mean_z'] = df['z'].rolling(window=50).mean()
    features['std_x'] = df['x'].rolling(window=50).std()
    features['std_y'] = df['y'].rolling(window=50).std()
    features['std_z'] = df['z'].rolling(window=50).std()
    features['max_x'] = df['x'].rolling(window=50).max()
    features['max_y'] = df['y'].rolling(window=50).max()
    features['max_z'] = df['z'].rolling(window=50).max()
    features['min_x'] = df['x'].rolling(window=50).min()
    features['min_y'] = df['y'].rolling(window=50).min()
    features['min_z'] = df['z'].rolling(window=50).min()
    features['median_x'] = df['x'].rolling(window=50).median()
    features['median_y'] = df['y'].rolling(window=50).median()
    features['median_z'] = df['z'].rolling(window=50).median()
    return features.dropna()

X_train_features = calculate_features(X_train)
X_test_features = calculate_features(X_test)

# Scaling data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_features)
X_test_scaled = scaler.transform(X_test_features)

# SVM model
svm_model = SVC()
svm_model.fit(X_train_scaled, y_train[:len(X_train_scaled)])
y_pred_svm = svm_model.predict(X_test_scaled)

# Random Forest pattern
rf_model = RandomForestClassifier()
rf_model.fit(X_train_scaled, y_train[:len(X_train_scaled)])
y_pred_rf = rf_model.predict(X_test_scaled)

# Evaluation of the models
print("SVM Classification Report:")
print(classification_report(y_test[:len(y_pred_svm)], y_pred_svm))

print("Random Forest Classification Report:")
print(classification_report(y_test[:len(y_pred_rf)], y_pred_rf))

"""
Explanation.
Downloading and preparing data:

Download the dataset from the accelerometer and separate it into features (X) and target column (y).
Divide the data into training and test sets.
Calculation of temporal features:

Determine the temporal features using a rolling window with a size of 50.
Calculate the mean, standard deviation, maximum, minimum, and median for each axis.
Building models:

Scale the data using StandardScaler.
Create and train SVM and Random Forest models.
Model evaluation:

We use classification_report to compare the results of both models across different metrics.
Conclusions.
Comparing models based on several metrics (precision, recall, F1-score) gives a more complete picture of model performance than just an accuracy score.
Based on the results, you can draw conclusions about which model is better at classifying different types of activities based on accelerometer data.
"""