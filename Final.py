# -*- coding: utf-8 -*-
"""
Created on Fri May 10 20:02:19 2024

@author: mathi
"""

#%% librairy import
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.decomposition import FastICA
from sklearn.manifold import TSNE
import umap
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.svm import SVC
from sklearn.base import clone
from sklearn.linear_model import Lasso
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from keras_tuner.tuners import RandomSearch, Hyperband
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow import keras

#%%

# Set a seed value
seed_value = 42

# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

# 4. Set `tensorflow` pseudo-random generator at a fixed value
tf.random.set_seed(seed_value)
#%% data import

x_train = pd.read_pickle('./DATA/A5_2024_xtrain.gz')
y_train = pd.read_pickle('./DATA/A5_2024_ytrain.gz')
x_test = pd.read_pickle('./DATA/A5_2024_xtest.gz')


# Create a mapping of labels to numeric values
label_mapping = {label: i for i, label in enumerate(y_train.unique())}
# Map labels to numeric values
y_train_mapped = y_train.map(label_mapping)

x_train[['node', 'event']] = x_train[['node', 'event']].apply(lambda x: x.astype(int))
x_test[['node', 'event']] = x_test[['node', 'event']].apply(lambda x: x.astype(int))

#%%

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

model = RandomForestClassifier()
model.fit(x_train, y_train_mapped)  # Assume x_train, y_train have been defined

selector = SelectFromModel(model, threshold='mean')
x_train_selected = selector.transform(x_train)

#%% Final Dataframe

data = pd.DataFrame(x_train_selected)

#%% Split and scale datasets 

# Split the training data for model training and validation
X_train, X_val, y_train, y_val = train_test_split(data, y_train_mapped, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)

#%%
from imblearn.over_sampling import SMOTE

# Assuming you have already loaded your dataset into x_train and y_train
smote = SMOTE(random_state=seed_value)
X_train, y_train = smote.fit_resample(X_train, y_train)

#%%
# Define the model-building function
def build_model(hp):
    model = Sequential()
   
    model.add(Dense(hp.Int('units_layer0', 16, 2048, step=16),
                    activation='relu', input_shape=(input_dim,)))
    model.add(Dropout(0.25))
    
    for j in range(hp.Int('num_dense_layers', 1, 2)):
        model.add(Dense(
            hp.Int(f'units_{j}', 16, 2048, step=16),
            activation='relu'
        ))
        model.add(Dropout(0.25))

    model.add(Dense(len(np.unique(y_train)), activation='softmax'))

    model.compile(
        optimizer=Adam(
            learning_rate=hp.Choice('learning_rate', [1e-3, 1e-4]),
            clipvalue=0.5
        ),
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    return model

#%%

input_dim = X_train.shape[1]
# Configure and run the tuner
tuner = Hyperband(
    build_model,
    objective='val_accuracy',  # changed to 'val_accuracy' for a clearer performance indicator
    max_epochs=15,
    factor=3,
    directory=os.path.join("./A5/models/", "Hyperparameter tuning", "MLP"),
    project_name='models_tuning'
)

tuner.search(
    X_train,
    y_train,
    epochs=50,
    validation_data=(X_val, y_val),
    callbacks=[
        keras.callbacks.TensorBoard(log_dir=os.path.join("./models/", "Hyperparameter tuning", "MLP", "log")),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    ]
)

# Get the best model and print summary
best_model = tuner.get_best_models(num_models=1)[0]
print("Best model summary:")
best_model.summary()

#%% MLP model Classification

# Assuming you know the input dimension (number of features) and number of classes
input_dim = X_train.shape[1]  # Number of features
num_classes = len(set(y_train))  # Assuming y_train is not one-hot encoded

model = Sequential([
    Dense(1500, activation='relu', input_shape=(input_dim,)),
    Dropout(0.25),
    Dense(900, activation='relu'),
    Dropout(0.25),
    Dense(num_classes, activation='softmax')  # Use softmax for multi-class classification
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001, clipvalue = 0.5), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=3,mode='min')
mc = tf.keras.callbacks.ModelCheckpoint('.best_model.h5', save_best_only=True)

history = model.fit(X_train, y_train, epochs=100,batch_size=15,
                          validation_data=(X_val, y_val), callbacks=[es,mc])


# Evaluate the model
evaluation = model.evaluate(X_val, y_val)
print(f"Validation Loss: {evaluation[0]}, Validation Accuracy: {evaluation[1]}")

# Get predictions and convert probabilities to class labels
y_pred_prob = model.predict(X_val)
y_pred = np.argmax(y_pred_prob, axis=1)
#%%

# Function to calculate BCR
def calculate_BCR(y_true, y_pred, n_classes):
    # Get the confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
    # True Positives are on the diagonal
    TP = np.diag(cm)
    # Total instances for each class
    ni = np.sum(cm, axis=1)
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        pi = np.true_divide(TP, ni)
        pi[~np.isfinite(pi)] = 0  # replace NaNs and inf with 0
    # Calculate BCR
    BCR = np.mean(pi)
    return BCR

# Determine the number of classes
n_classes = len(np.unique(y_train))

# Calculate BCR
bcr = calculate_BCR(y_val, y_pred, n_classes)
print(f"Balanced Classification Rate (BCR): {bcr:.3f}")


#%%

scaler = StandardScaler().fit(x_train_selected)
X_train = scaler.transform(x_train_selected)

#%%
from imblearn.over_sampling import SMOTE

# Assuming you have already loaded your dataset into x_train and y_train
smote = SMOTE(random_state=seed_value)
X_train, y_train = smote.fit_resample(X_train, y_train_mapped)

#%% MLP model Classification

# Assuming you know the input dimension (number of features) and number of classes
input_dim = X_train.shape[1]  # Number of features
num_classes = len(set(y_train))  # Assuming y_train is not one-hot encoded

model = Sequential([
    Dense(1500, activation='relu', input_shape=(input_dim,)),
    Dropout(0.25),
    Dense(900, activation='relu'),
    Dropout(0.25),
    Dense(num_classes, activation='softmax')  # Use softmax for multi-class classification
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001, clipvalue = 0.5), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=3,mode='min')
mc = tf.keras.callbacks.ModelCheckpoint('.best_model.h5', save_best_only=True)

history = model.fit(X_train, y_train, epochs=100,batch_size=15,
                          validation_split = 0.2, callbacks=[es,mc])


#%%

# Select the relevant features from x_test
x_test_selected = selector.transform(x_test)

# Scale x_test using the previously fitted scaler
X_test_scaled = scaler.transform(x_test_selected)

# Predict using the best model
y_test_pred_prob = model.predict(X_test_scaled)
y_test_pred = np.argmax(y_test_pred_prob, axis=1)

# Inverse the label mapping
inverse_label_mapping = {v: k for k, v in label_mapping.items()}
y_test_pred_labels = [inverse_label_mapping[label] for label in y_test_pred]


# Create a DataFrame with the predictions
# Include the example IDs and format the class labels with double quotes
predictions_df = pd.DataFrame({
    "": range(len(y_test_pred_labels)),  # Generate index from 0 to length of predictions
    "label": y_test_pred_labels
})

# Save to CSV file with double-quoted strings for the 'label' column
predictions_df.to_csv("./predictions.csv", index=False)
print("Predictions saved to 'predictions.csv'.")






