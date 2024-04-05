
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import tensorflow as tf
import warnings
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
warnings.simplefilter(action = 'ignore', category = FutureWarning)

df = pd.read_csv('Dataset_FeatureBased.csv')

X = df.iloc[:, 0:55].values
y = df['Target'].values


# Split the data in time-series order
train_size = int(0.6 * len(df))
val_size = int(0.2 * len(df))
test_size = len(df) - train_size - val_size

train_data = df.iloc[:train_size]
val_data = df.iloc[train_size:train_size + val_size]
test_data = df.iloc[train_size + val_size:]

X_train = train_data.iloc[:, 0:55].values
y_train = train_data['Target'].values

X_valid = val_data.iloc[:, 0:55].values
y_valid = val_data['Target'].values

X_test = test_data.iloc[:, 0:55].values
y_test = test_data['Target'].values




# Define the model
def create_model():
    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=regularizers.l1_l2(l1=kernel_reg, l2=kernel_reg)))
    model.add(Dropout(dropout_prob))
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=kernel_reg, l2=kernel_reg)))
    model.add(Dropout(dropout_prob))
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=kernel_reg, l2=kernel_reg)))
    model.add(Dropout(dropout_prob))
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=kernel_reg, l2=kernel_reg)))
    model.add(Dropout(dropout_prob))
    model.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=kernel_reg, l2=kernel_reg)))
    model.add(Dropout(dropout_prob))
    # model.add(Dense(8, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=kernel_reg, l2=kernel_reg)))

    model.add(Dense(1, activation='linear'))  # Linear activation for regression
    
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mean_squared_error', metrics=['mean_squared_error'])

    return model

import tensorflow as tf
# Set up GPU growth (optional but recommended)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


# Hyperparameters
dropout_prob = 0.1
kernel_reg = 0.0001
# lr = 0.001

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)

# Create the model
model = create_model()
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=200, batch_size=128, validation_data=(X_valid, y_valid), callbacks=[early_stopping])

# Evaluate the model on the test set
mse, _ = model.evaluate(X_test, y_test)

print(f"Test MSE: {mse}")



clf=model

predicted_train = clf.predict(X_train)

predicted_validation = clf.predict(X_valid)
predicted_test = clf.predict(X_test)


print('MLP Train:',np.sqrt(mean_squared_error(y_train, predicted_train)))
print('MLP Validation:',np.sqrt(mean_squared_error(y_valid, predicted_validation)))
print('MLP Testing:',np.sqrt(mean_squared_error(y_test, predicted_test)))



# y_true = y_test
# y_pred = np.transpose(predicted_test)
# import numpy as np
# mse2 = np.mean((y_true - y_pred) ** 2)
# print("MSE:", mse2)
# mae = np.mean(np.abs(y_true - y_pred))
# print("MAE:", mae)
# mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
# print("MAPE:", mape)





