# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 01:36:10 2023

@author: Xin Wang
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.layers import Input, Conv1D, MaxPooling1D, LSTM, UpSampling1D
from keras.models import Model
from keras import layers
import pickle  # You can also use other libraries like joblib

data = np.load('Xy_data.npz')
X = data['X']
y = data['y']

time_steps = X.shape[1]
num_features = X.shape[2]


nan_indices = np.isnan(X)
nan_count = np.sum(nan_indices)
if nan_count > 0:
    print(f"Found {nan_count} NaN values in the input data X.")
else:
    print("No NaN values found in the input data X.")



# Calculate the sizes for each split
total_samples = X.shape[0]
train_size = int(0.6 * total_samples)
valid_size = int(0.2 * total_samples)
# Split the data
X_train = X[:train_size, :, :]
X_val = X[train_size:train_size + valid_size, :, :]
X_test = X[train_size + valid_size:, :, :]
y_train = y[:train_size]
y_val = y[train_size:train_size + valid_size, :]
y_test = y[train_size + valid_size:, :]

del X, y
# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, num_features)).reshape(-1, time_steps, num_features)
X_val_scaled = scaler.transform(X_val.reshape(-1, num_features)).reshape(-1, time_steps, num_features)
X_test_scaled = scaler.transform(X_test.reshape(-1, num_features)).reshape(-1, time_steps, num_features)
del X_train, X_val, X_test
# # Autoencoder for Dimensionality Reduction

input_dim = (time_steps, num_features)
input_layer = Input(shape=input_dim)
encoded = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(input_layer)
encoded = MaxPooling1D(pool_size=2)(encoded)
encoded = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(encoded)
encoded = MaxPooling1D(pool_size=2)(encoded)
# encoded = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(encoded)
# encoded = MaxPooling1D(pool_size=2)(encoded)
# encoded = LSTM(64, activation='tanh', return_sequences=True)(encoded)
encoded = LSTM(16, activation='tanh', return_sequences=True)(encoded)
# decoded = LSTM(64, activation='tanh', return_sequences=True)(encoded)
# decoded = UpSampling1D(2)(decoded)
# decoded = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(decoded)

# Define the decoder architecture

decoded = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(encoded)
decoded = UpSampling1D(size=2)(decoded)
decoded = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(decoded)
decoded = UpSampling1D(size=2)(decoded)
decoded = Conv1D(filters=num_features, kernel_size=3, activation='linear', padding='same')(decoded)

autoencoder = Model(input_layer, decoded)
encoder = Model(input_layer, encoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
autoencoder.summary()


from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
# Train the autoencoder with early stopping
history = autoencoder.fit(X_train_scaled, X_train_scaled,
                epochs=500, batch_size=128, shuffle=True,
                validation_data=(X_val_scaled, X_val_scaled),
                callbacks=[early_stopping])
from keras.models import save_model
autoencoder.save('autoencoder_linear_64_32_16_All_paper.h5')






from keras.models import load_model
# Load the saved autoencoder model
loaded_autoencoder = load_model('autoencoder_linear_64_32_16_All_paper.h5')

encoder = Model(loaded_autoencoder.input, loaded_autoencoder.layers[5].output)
# # Get the reduced-dimensional representations
X_train_encoded = encoder.predict(X_train_scaled)
X_val_encoded = encoder.predict(X_val_scaled)
X_test_encoded = encoder.predict(X_test_scaled)
# Save the variables using pickle
with open('encoded_data.pkl', 'wb') as f:
    pickle.dump((X_train_encoded, X_val_encoded, X_test_encoded, y_train, y_val, y_test), f)




# import pickle
# history_filename = 'paper_autoencoder_paper.pkl'
# with open(history_filename, 'wb') as file:
#     pickle.dump(history.history, file)
# print(f"Training history saved to {history_filename}")


# import pickle
# import matplotlib.pyplot as plt
# with open(history_filename, 'rb') as file:
#     loaded_history = pickle.load(file)

# # Plot the loaded training history
# plt.figure(figsize=(10, 6))
# plt.plot(loaded_history['loss'])
# plt.plot(loaded_history['val_loss'])
# plt.title('Loaded Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper right')
# plt.show()












