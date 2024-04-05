# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 01:36:10 2023

@author: Xin Wang
"""
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, LSTM, Dropout
from keras.models import Model, Sequential
from keras import layers
import pickle  # You can also use other libraries like joblib

from keras import backend as K
from keras.layers import Layer

    
# Load the selected data from the pickle file
with open('selected_encoded_data.pkl', 'rb') as f:
    selected_data = pickle.load(f)
# Unpack the loaded data into variables
X_train_encoded, X_val_encoded, X_test_encoded, y_train, y_val, y_test = selected_data

from keras.layers import Conv1D, LSTM, BatchNormalization

# Create a more complex 1D CNN-LSTM Model with Attention for Prediction
cnn_lstm_model = Sequential()
# Add 1D Convolutional Layers
cnn_lstm_model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_encoded.shape[1], X_train_encoded.shape[2])))
# cnn_lstm_model.add(BatchNormalization())  # Ad
cnn_lstm_model.add(MaxPooling1D(pool_size=2))
# cnn_lstm_model.add(Dropout(0.2))
cnn_lstm_model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
# cnn_lstm_model.add(SelfAttention())  # Add Self-Attention Layer
# cnn_lstm_model.add(MaxPooling1D(pool_size=2))
# Add LSTM and Dropout Layers
# cnn_lstm_model.add(LSTM(128, return_sequences=True))
# cnn_lstm_model.add(Dropout(0.4))
cnn_lstm_model.add(LSTM(64, return_sequences=True))
# cnn_lstm_model.add(BatchNormalization())  # Ad
cnn_lstm_model.add(Dropout(0.1))
cnn_lstm_model.add(LSTM(32))
cnn_lstm_model.add(Dropout(0.1))
# cnn_lstm_model.add(Dense(32, activation='relu'))
# cnn_lstm_model.add(Dropout(0.1))
# cnn_lstm_model.add(Dense(16, activation='relu'))
cnn_lstm_model.add(Dense(1, activation='linear'))

from keras.optimizers import Adam
cnn_lstm_model.compile('adam', loss='mean_squared_error')
# optimizer = Adam(learning_rate=0.0005)
# cnn_lstm_model.compile(optimizer=optimizer, loss='mean_squared_error')

cnn_lstm_model.summary()

import tensorflow as tf
# Set up GPU growth (optional but recommended)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# best_val_loss = float('inf')  # Initialize with a high value
# best_weights = None
# epochs =70
# batch_size = 128
# for epoch in range(epochs):
#     history = cnn_lstm_model.fit(X_train_encoded, y_train, batch_size=batch_size, shuffle=True, verbose=1) 
#     # Evaluate on validation data
#     val_loss = cnn_lstm_model.evaluate(X_val_encoded, y_val, verbose=0)
#     print(f"Epoch {epoch+1}/{epochs} - Train Loss: {history.history['loss'][0]:.4f} - Validation Loss: {val_loss:.4f}")
#     # Update best weights if current validation loss is better
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         best_weights = cnn_lstm_model.get_weights()
# # Load the best weights
# cnn_lstm_model.set_weights(best_weights)

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
history = cnn_lstm_model.fit(X_train_encoded, y_train,
                    epochs=1000, batch_size=128, shuffle=True, validation_data=(X_val_encoded, y_val), callbacks=[early_stopping])  # 



import pickle
history_filename = 'paper_training_history_pro01.pkl'
with open(history_filename, 'wb') as file:
    pickle.dump(history.history, file)
print(f"Training history saved to {history_filename}")







history_filename = 'paper_training_history_pro01.pkl'
import pickle
import matplotlib.pyplot as plt
with open(history_filename, 'rb') as file:
    loaded_history = pickle.load(file)

FS=14
# Plot the loaded training history
plt.figure(figsize=(10, 6))
plt.plot(loaded_history['loss'])
plt.plot(loaded_history['val_loss'])
plt.title('Loaded Model loss')
plt.ylabel('Loss', fontsize=FS)
plt.xlabel('Epoch', fontsize=FS)
plt.legend(['Train', 'Validation'], loc='upper right')
plt.legend(fontsize=FS)
plt.tick_params(axis='both', labelsize=FS)
plt.show()
figname = 'paper_loss_proposed model.svg' 
fig=plt.gcf()
fig.savefig(figname, format='svg')







model_name = 'cnn_lstm_model_patience_drop013.h5'

cnn_lstm_model.save(model_name)
from keras.models import load_model
from keras.utils import custom_object_scope
# Define a dictionary with the custom objects (layers) you've used
custom_objects = {'SelfAttention': SelfAttention}
# Load the model using custom_object_scope
with custom_object_scope(custom_objects):
    cnn_lstm_model = load_model(model_name)

# Evaluate the model
from sklearn.metrics import mean_squared_error
predictions_train = cnn_lstm_model.predict(X_train_encoded)
mse = mean_squared_error(y_train, predictions_train)
print(f"Train Mean Squared Error: {mse}")

predictions_val = cnn_lstm_model.predict(X_val_encoded)
mse = mean_squared_error(y_val, predictions_val)
print(f"Validation Mean Squared Error: {mse}")

predictions = cnn_lstm_model.predict(X_test_encoded)
mse = mean_squared_error(y_test, predictions)
print(f"Test Mean Squared Error: {mse}")

import matplotlib.pyplot as plt
# Plot the predicted values and true values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions, color='b', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='r', linestyle='--')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True Values vs. Predictions')
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(y_test, label='True Values', color='b')
plt.plot(predictions, label='Predicted Values', color='r', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Values')
plt.title('True Value Curve vs. Predicted Value Curve')
plt.legend()
plt.show()





