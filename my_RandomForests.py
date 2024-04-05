
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from bayes_opt import BayesianOptimization

from sklearn.ensemble import RandomForestRegressor

import warnings
from sklearn.model_selection import cross_val_score
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




def rf_evaluate(max_depth, n_estimators, min_samples_split):
    clf = RandomForestRegressor(
        max_depth=int(max_depth),
        n_estimators=int(n_estimators),
        min_samples_split=int(min_samples_split),
        n_jobs=-1
    )
    
    scores = -cross_val_score(clf, X_train, y_train, cv=5, scoring='neg_mean_squared_error')  # Use neg_mean_squared_error for MSE
    print(np.nanmean(scores))
    return np.nanmean(scores)

def bayesOpt(init_points, n_iter):
    rfBO = BayesianOptimization(rf_evaluate, {
        'max_depth': (1, 20),
        'n_estimators': (10, 1000),
        'min_samples_split': (2, 15)
    })

    rfBO.maximize(init_points=init_points, n_iter=n_iter)
    return rfBO.max


# Optimize the hyparameters using Bayesian optimization algorithm
opt_params = bayesOpt(init_points=5, n_iter=15)
print(opt_params)

params = opt_params['params']
params['max_depth'] = int(params['max_depth'])
params['n_estimators'] = int(params['n_estimators'])


# opt_params ={'target': -0.6555460997489311,
#  'params': {'max_depth': 10,
#   'n_estimators': 20}}

clf = RandomForestRegressor(
    max_depth=int(params['max_depth']),
    n_estimators=int(params['n_estimators']),
    n_jobs=-1
)

# from sklearn.externals import joblib  # Import joblib for model saving
# model_filename = 'random_forest_regressor_model.pkl'
# joblib.dump(clf, model_filename)


clf.fit( X_train, y_train)
predicted_train = clf.predict(X_train)

predicted_validation = clf.predict(X_valid)
predicted_test = clf.predict(X_test)


print('RF Train:',np.sqrt(mean_squared_error(y_train, predicted_train)))
print('RF Validation:',np.sqrt(mean_squared_error(y_valid, predicted_validation)))
print('RF Testing:',np.sqrt(mean_squared_error(y_test, predicted_test)))



y_true = y_test
y_pred = predicted_test
import numpy as np
mse2 = np.mean((y_true - y_pred) ** 2)
print("MSE:", mse2)
mae = np.mean(np.abs(y_true - y_pred))
print("MAE:", mae)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
print("MAPE:", mape)


