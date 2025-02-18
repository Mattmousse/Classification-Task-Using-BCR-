# -*- coding: utf-8 -*-
"""
Created on Thu May  9 14:00:08 2024

@author: mathi
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras_tuner.tuners import RandomSearch, Hyperband
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import scipy.stats, os, json
from numpy import hstack
import matplotlib.dates as mdates
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

#%%
# Define the model-building function
def build_model(hp):
    model = Sequential()
    model.add(Conv1D(
        filters=hp.Int('filters_1', min_value=16, max_value=64, step=16),
        kernel_size=hp.Choice('kernel_size_1', values=[3, 5]),
        activation='relu',
        input_shape= (X_train.shape[1], 1)
    ))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(hp.Float('dropout_1', min_value=0.1, max_value=0.3, step=0.1)))

    # Multiple Conv1D layers based on the tuner's decision
    for i in range(hp.Int('num_conv_layers', 1, 3)):  # Number of Conv2D layers
        model.add(Conv1D(
            filters=hp.Int(f'filters_{i}', 64, 512, step=64),
            kernel_size=hp.Choice(f'kernel_size_{i}', values=[3, 5]),
            activation='relu',
            padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.2))

    # Flattening the 2D arrays for fully connected layers
    model.add(Flatten())

    # Adding Dense layers based on the tuner's decision
    for j in range(hp.Int('num_dense_layers', 1, 3)):
        model.add(Dense(
            hp.Int(f'units_{j}', 64, 1024, step=64),
            activation='relu'
        ))
        model.add(Dropout(0.2))

    model.add(Dense(len(np.unique(y_train_mapped)), activation='softmax'))

    model.compile(
        optimizer=Adam(hp.Choice('learning_rate', [1e-3, 1e-4])),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


#%%

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

gene_columns = [col for col in x_test.columns if 'G' in col]
x_train_gene = x_train[gene_columns]

non_gene_columns = [col for col in x_train.columns if 'G' not in col]
x_train_non_gene = x_train[non_gene_columns]

# Assuming x_train_gene is a pandas DataFrame
correlation_matrix = x_train_gene.corr()
# Create a mask to identify correlation above the threshold (excluding self-correlation)
high_corr_var = np.where(correlation_matrix > 0.95)
high_corr_var = [(correlation_matrix.index[x], correlation_matrix.columns[y]) for x, y in zip(*high_corr_var) if x != y and x < y]
# Create a set to store features to remove
to_remove = set()
for var1, var2 in high_corr_var:
    to_remove.add(var1)  # You can add logic to decide which one to remove

# Print out what will be removed
print("Columns to remove:", to_remove)

x_train_reduced = x_train_gene.drop(columns=list(to_remove))
print("Reduced DataFrame shape:", x_train_reduced.shape)


data = pd.DataFrame(x_train_reduced)
data = np.array(pd.concat([data, x_train_non_gene], axis = 1))

#%%

# Split the training data for model training and validation
X_train, X_val, y_train, y_val = train_test_split(data, y_train_mapped, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)

# Reshape data for Conv1D (batch, steps, channels)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
#%%

tuner = Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=15,
    factor=3,
    batchsize = 15,
    hyperband_iterations= 3,
    directory="C:/Users/mathi/Documents/Université/Semestre 10/Machine Learning/A5/models/" + "Hyperparameter tuning/CNN1d/",
    project_name='models_tuning'
)

# Setup TensorBoard callback
tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir= "C:/Users/mathi/Documents/Université/Semestre 10/Machine Learning/A5/models/" + "Hyperparameter tuning/CNN1d/log",
    update_freq='epoch'  # Logs metrics at the end of each epoch
)

# Setup EarlyStopping callback
early_stopping_callback = keras.callbacks.EarlyStopping(
    monitor='val_loss',    # Monitor the validation loss
    patience=3,            # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True  # Restores model weights from the epoch with the best value of the monitored quantity
)

tuner.search(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    # Use the TensorBoard callback.
    # The logs will be write to "/tmp/tb_logs".
    callbacks=[tensorboard_callback, early_stopping_callback]
)

#%%
# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]
print("Best model summary:" )
best_model.summary()


"""
Best val_accuracy So Far: 0.8452380895614624
Total elapsed time: 12h 24m 46s
Best model summary:
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv1d (Conv1D)             (None, 2075, 32)          192       
                                                                 
 batch_normalization (BatchN  (None, 2075, 32)         128       
 ormalization)                                                   
                                                                 
 max_pooling1d (MaxPooling1D  (None, 1037, 32)         0         
 )                                                               
                                                                 
 dropout (Dropout)           (None, 1037, 32)          0         
                                                                 
 conv1d_1 (Conv1D)           (None, 1037, 128)         12416     
                                                                 
 batch_normalization_1 (Batc  (None, 1037, 128)        512       
 hNormalization)                                                 
                                                                 
 max_pooling1d_1 (MaxPooling  (None, 518, 128)         0         
 1D)                                                             
                                                                 
 dropout_1 (Dropout)         (None, 518, 128)          0         
                                                                 
 flatten (Flatten)           (None, 66304)             0         
                                                                 
 dense (Dense)               (None, 512)               33948160  
                                                                 
 dropout_2 (Dropout)         (None, 512)               0         
                                                                 
 dense_1 (Dense)             (None, 896)               459648    
                                                                 
 dropout_3 (Dropout)         (None, 896)               0         
                                                                 
 dense_2 (Dense)             (None, 384)               344448    
                                                                 
 dropout_4 (Dropout)         (None, 384)               0         
                                                                 
 dense_3 (Dense)             (None, 3)                 1155      
                                                                 
=================================================================
Total params: 34,766,659
Trainable params: 34,766,339
Non-trainable params: 320
_________________________________________________________________

Value             |Best Value So Far |Hyperparameter
48                |32                |filters_1
5                 |5                 |kernel_size_1
0.2               |0.2               |dropout_1
1                 |1                 |num_conv_layers
512               |128               |filters_0
5                 |3                 |kernel_size_0
2                 |3                 |num_dense_layers
384               |512               |units_0
0.0001            |0.001             |learning_rate
896               |896               |units_1
448               |384               |units_2
512               |384               |filters_2
5                 |3                 |kernel_size_2
15                |15                |tuner/epochs
0                 |5                 |tuner/initial_epoch
0                 |2                 |tuner/bracket
0                 |2                 |tuner/round
"""

















