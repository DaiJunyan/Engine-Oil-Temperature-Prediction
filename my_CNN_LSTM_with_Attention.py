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
from keras.optimizers import Adam

# with open('encoded_data.pkl', 'rb') as f:
#     X_train_encoded, X_val_encoded, X_test_encoded, y_train, y_val, y_test = pickle.load(f)
# # Select every 100th element from each variable
# X_train_encoded_selected = X_train_encoded[::10]
# X_val_encoded_selected = X_val_encoded[::10]
# X_test_encoded_selected = X_test_encoded[::10]
# y_train_selected = y_train[::10]
# y_val_selected = y_val[::10]
# y_test_selected = y_test[::10]

# # Save the selected variables to a new pickle file
# selected_data = (X_train_encoded_selected, X_val_encoded_selected, X_test_encoded_selected, y_train_selected, y_val_selected, y_test_selected)
# with open('selected_encoded_data.pkl', 'wb') as f:
#     pickle.dump(selected_data, f)    
    

# Load the selected data from the pickle file
with open('selected_encoded_data.pkl', 'rb') as f:
    selected_data = pickle.load(f)
# Unpack the loaded data into variables
X_train_encoded, X_val_encoded, X_test_encoded, y_train, y_val, y_test = selected_data



class SelfAttention(Layer):
    def __init__(self, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(3, input_shape[-1], input_shape[-1]),
                                      initializer='uniform',
                                      trainable=True)
        super(SelfAttention, self).build(input_shape)

    def call(self, x):
        WQ = K.dot(x, self.kernel[0])
        WK = K.dot(x, self.kernel[1])
        WV = K.dot(x, self.kernel[2])

        QK = K.batch_dot(WQ, K.permute_dimensions(WK, (0, 2, 1)))
        QK = QK / (64 ** 0.5)  # Scale the dot products

        QK = K.softmax(QK)
        V = K.batch_dot(QK, WV)

        return V

    def compute_output_shape(self, input_shape):
        return input_shape


# Create a more complex 1D CNN-LSTM Model with Attention for Prediction
cnn_lstm_model = Sequential()
# Add 1D Convolutional Layers
cnn_lstm_model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_encoded.shape[1], X_train_encoded.shape[2])))
# cnn_lstm_model.add(BatchNormalization())  # Ad
cnn_lstm_model.add(SelfAttention())  # Add Self-Attention Layer
cnn_lstm_model.add(MaxPooling1D(pool_size=2))

cnn_lstm_model.add(LSTM(64, return_sequences=True))
# cnn_lstm_model.add(BatchNormalization())  # Ad
cnn_lstm_model.add(SelfAttention())  # Add Self-Attention Layer
cnn_lstm_model.add(Dropout(0.1))
cnn_lstm_model.add(LSTM(32))
cnn_lstm_model.add(Dropout(0.1))
# cnn_lstm_model.add(Dense(32, activation='relu'))
# cnn_lstm_model.add(Dropout(0.1))
# cnn_lstm_model.add(Dense(16, activation='relu'))
cnn_lstm_model.add(Dense(1, activation='linear'))


optimizer = Adam(learning_rate=0.0001)
cnn_lstm_model.compile(optimizer=optimizer, loss='mean_squared_error')
cnn_lstm_model.summary()

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dropout, Dense
from keras.models import Sequential
import tensorflow as tf
# Set up GPU growth (optional but recommended)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
history = cnn_lstm_model.fit(X_train_encoded, y_train,
                    epochs=1000, batch_size=128, shuffle=True, validation_data=(X_val_encoded, y_val), callbacks=[early_stopping])  # 

# import pickle
# history_filename = 'cnn_lstm_with_attention.pkl'
# with open(history_filename, 'wb') as file:
#     pickle.dump(history.history, file)
# print(f"Training history saved to {history_filename}")
# model_name = 'cnn_lstm_with_attention.h5'
# cnn_lstm_model.save(model_name)
# from keras.models import load_model
# from keras.utils import custom_object_scope
# # Define a dictionary with the custom objects (layers) you've used
# custom_objects = {'SelfAttention': SelfAttention}
# # Load the model using custom_object_scope
# with custom_object_scope(custom_objects):
#     cnn_lstm_model = load_model(model_name)

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
