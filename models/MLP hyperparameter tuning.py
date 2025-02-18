# -*- coding: utf-8 -*-
"""
Created on Thu May  9 14:21:56 2024

@author: mathi
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May  9 14:00:08 2024

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
#%% data import
path = "C:/Users/mathi/Documents/Université/Semestre 10/Machine Learning/A5/DATA/"
x_train = pd.read_pickle(path + 'A5_2024_xtrain.gz')
y_train = pd.read_pickle(path + 'A5_2024_ytrain.gz')
x_test = pd.read_pickle(path + 'A5_2024_xtest.gz')

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
#data = np.array(pd.concat([data, x_train_non_gene], axis = 1))

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
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

#%%
# Define the model-building function
def build_model(hp):
    model = Sequential()
   
    model.add(Dense(hp.Int('units_layer0', 16, 2048, step=16),
                    activation='relu', input_shape=(input_dim,)))
    model.add(Dropout(0.25))
    
    for j in range(hp.Int('num_dense_layers', 1, 3)):
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
    directory=os.path.join("C:/Users/mathi/Documents/Université/Semestre 10/Machine Learning/A5/models/", "Hyperparameter tuning", "MLP"),
    project_name='models_tuning'
)

tuner.search(
    X_train,
    y_train,
    epochs=50,
    validation_data=(X_val, y_val),
    callbacks=[
        keras.callbacks.TensorBoard(log_dir=os.path.join("C:/Users/mathi/Documents/Université/Semestre 10/Machine Learning/A5/models/", "Hyperparameter tuning", "MLP", "log")),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    ]
)

# Get the best model and print summary
best_model = tuner.get_best_models(num_models=1)[0]
print("Best model summary:")
best_model.summary()
