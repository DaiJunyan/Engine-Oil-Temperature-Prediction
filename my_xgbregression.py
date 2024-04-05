# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 16:15:09 2021

@author: Xin Wang
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from bayes_opt import BayesianOptimization

import warnings
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




# Modify the xgb_evaluate function to remove the eval_metric argument in XGBRegressor
def xgb_evaluate(eta, colsample_bytree, subsample, reg_lambda, max_depth, n_estimators):
                 
    clf = XGBRegressor(
        objective='reg:squarederror',
        tree_method='gpu_hist',
        predictor='gpu_predictor',
        eta = eta,
        colsample_bytree=colsample_bytree,
        subsample=subsample,
        reg_lambda=reg_lambda,
        max_depth = int(max_depth),
        n_estimators=int(n_estimators)
    )
    clf.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    value = clf.predict(X_valid)
    return -mean_squared_error(y_valid, value, squared=False)


def bayesOpt(X_train, y_train, X_valid, y_valid, init_points, n_iter):
    xgbBO = BayesianOptimization(xgb_evaluate, {
        'colsample_bytree': (0.2,1),
        'reg_lambda': (0,2),
        'n_estimators': (20, 2000),
        'max_depth': (2,30),
        'subsample': (0.3, 1.0),  # Change for big datasets
        'eta':(0.001,0.2)})

    xgbBO.maximize(init_points, n_iter)

    return xgbBO.max


opt_params = bayesOpt(X_train, y_train, X_valid, y_valid, init_points=20, n_iter=60)



# opt_params=    {'target': -0.32915804347007493,
#   'params': {'colsample_bytree': 0.8589323957922519,
#   'eta': 0.1652453570404098,
#   'max_depth': 2.695787684235122,
#   'n_estimators': 693.4798507176464,
#   'reg_lambda': 0.08626714893816168,
#   'subsample': 0.9276209318554387}}

# # Train the final model with the optimal parameters
final_model = XGBRegressor(
    objective='reg:squarederror',
    tree_method='gpu_hist',
    predictor='gpu_predictor',
    eta=opt_params['params']['eta'],
    colsample_bytree=opt_params['params']['colsample_bytree'],
    subsample=opt_params['params']['subsample'],
    reg_lambda=opt_params['params']['reg_lambda'],  
    max_depth=int(opt_params['params']['max_depth']),
    n_estimators=int(opt_params['params']['n_estimators'])
)

# Train the final model on the entire training set
final_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

# Predict on the test set
y_pred = final_model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE) on the test set: {mse:.2f}")
print(f"R-squared (R2) on the test set: {r2:.2f}")


# import joblib
# joblib.dump(final_model, 'mymodelXGB.pkl')

# y_pred = final_model.predict(X_train)
# mse = mean_squared_error(y_train, y_pred)
# r2 = r2_score(y_train, y_pred)
# rmse = np.sqrt(mean_squared_error(y_train, y_pred))
# print(f"Mean Squared Error (MSE) on the Train set: {mse:.2f}")
# # print(f"R-squared (R2) on the Train set: {r2:.2f}")
# print(f"RMSE on the train set: {rmse:.2f}")

# y_pred = final_model.predict(X_valid)
# mse = mean_squared_error(y_valid, y_pred)
# r2 = r2_score(y_valid, y_pred)
# rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
# print(f"Mean Squared Error (MSE) on the Validation set: {mse:.2f}")
# # print(f"R-squared (R2) on the Validatiom set: {r2:.2f}")
# print(f"RMSE on the Validatiom set: {rmse:.2f}")

# y_pred = final_model.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# print(f"Mean Squared Error (MSE) on the test set: {mse:.2f}")
# # print(f"R-squared (R2) on the test set: {r2:.2f}")
# print(f"RMSE on the test set: {rmse:.2f}")


