# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 01:36:10 2023

@author: Xin Wang
"""
import numpy as np
import pickle  # You can also use other libraries like joblib

from keras.models import Sequential
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam
import tensorflow as tf

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

# Create a more complex 1D CNN-LSTM Model with Attention for Prediction
cnn_model = Sequential()
# Add 1D Convolutional Layers
cnn_model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train_encoded.shape[1], X_train_encoded.shape[2])))
cnn_model.add(BatchNormalization())  # Ad
cnn_model.add(MaxPooling1D(pool_size=2))
cnn_model.add(Dropout(0.1))  # Add Dropout with a rate of 0.1
# Add 2nd 1D Convolutional Layer
cnn_model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
cnn_model.add(BatchNormalization())  # Ad
cnn_model.add(MaxPooling1D(pool_size=2))

cnn_model.add(Dropout(0.1))  # Add Dropout with a rate of 0.1
cnn_model.add(Flatten())
cnn_model.add(Dense(1))  # Output layer with 1 neuron for regression

cnn_model.compile(optimizer='adam', loss='mean_squared_error')
cnn_model.summary()


# Set up GPU growth (optional but recommended)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=20)
history = cnn_model.fit(X_train_encoded, y_train,
                    epochs=1000, batch_size=128, shuffle=True, validation_data=(X_val_encoded, y_val), callbacks=[early_stopping])  # 


# Evaluate the model
from sklearn.metrics import mean_squared_error
predictions_train = cnn_model.predict(X_train_encoded)
mse = mean_squared_error(y_train, predictions_train)
print(f"Train Mean Squared Error: {mse}")
predictions_val = cnn_model.predict(X_val_encoded)
mse = mean_squared_error(y_val, predictions_val)
print(f"Validation Mean Squared Error: {mse}")
predictions = cnn_model.predict(X_test_encoded)
mse = mean_squared_error(y_test, predictions)
print(f"Test Mean Squared Error: {mse}")


y_true = y_test
y_pred=predictions
import numpy as np
mse2 = np.mean((y_true - y_pred) ** 2)
print("MSE:", mse2)
mae = np.mean(np.abs(y_true - y_pred))
print("MAE:", mae)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
print("MAPE:", mape)



