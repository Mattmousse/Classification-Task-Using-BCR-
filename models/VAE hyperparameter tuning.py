# -*- coding: utf-8 -*-
"""
Created on Thu May  9 18:27:52 2024

@author: mathi
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K, losses, optimizers
from keras_tuner import RandomSearch, Hyperband
from tensorflow.keras.losses import binary_crossentropy
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow import keras
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
# Define the sampling layer
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

# Function to create the VAE model
def build_vae(hp):
    # Encoder
    encoder_inputs = layers.Input(shape=(X_train.shape[1],))
    x = layers.Dense(
        units=hp.Int('encoder_dense_units', min_value=64, max_value=512, step=32),
        activation='relu'
    )(encoder_inputs)
    z_mean = layers.Dense(hp.Int('latent_dim', min_value=2, max_value=20, step=2))(x)
    z_log_var = layers.Dense(hp.Int('latent_dim', min_value=2, max_value=20, step=2))(x)
    z = layers.Lambda(sampling, output_shape=(hp.get('latent_dim'),))([z_mean, z_log_var])
    encoder = models.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')

    # Decoder
    decoder_inputs = layers.Input(shape=(hp.get('latent_dim'),))
    x = layers.Dense(
        units=hp.Int('decoder_dense_units', min_value=64, max_value=512, step=32),
        activation='relu'
    )(decoder_inputs)
    decoder_outputs = layers.Dense(X_train.shape[1], activation='sigmoid')(x)
    decoder = models.Model(decoder_inputs, decoder_outputs, name='decoder')

    # VAE Model
    outputs = decoder(encoder(encoder_inputs)[2])
    vae = models.Model(encoder_inputs, outputs, name='vae')

    # Add VAE loss
    reconstruction_loss = losses.binary_crossentropy(encoder_inputs, outputs) * X_train.shape[1]
    kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer=optimizers.Adam(hp.Choice('learning_rate', [1e-3, 1e-4])))
    
    return vae

# Setup the tuner
tuner = Hyperband(
    build_vae,
    objective='val_loss',
    max_trials=10,
    executions_per_trial=2,
    directory="C:/Users/mathi/Documents/Université/Semestre 10/Machine Learning/A5/models/" + "Hyperparameter tuning/VAE/",
    project_name='models_tuning'
)

# Print the search space summary
tuner.search_space_summary()

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
    x = X_train,
    validation_data=(X_val, None),
    epochs=50,
    # Use the TensorBoard callback.
    # The logs will be write to "/tmp/tb_logs".
    callbacks=[tensorboard_callback, early_stopping_callback]
)


# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]
best_model.summary()





























