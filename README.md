# Title
Prediction of the Temperature of Diesel Engine Oil in Railroad Locomotives Using Compressed Information-Based Data Fusion Method with Attention-Enhanced CNN-LSTM

# Overview
This repository presents the implementation and findings of a study proposing an attention-enhanced CNN-LSTM model tailored for predicting engine oil temperature in railroad locomotives. Leveraging compressed information from a myriad of sensor data associated with operational, system-related, and environmental factors, our model aims to provide accurate forecasts crucial for locomotive maintenance and performance optimization.

The results of our experiments demonstrate the effectiveness of the proposed model in predicting engine oil temperature with remarkable accuracy. We achieved Mean Square Error (MSE), Mean Absolute Error (MAE), and Mean Absolute Percentage Error (MAPE) values of 0.335, 0.369, and 0.586%, respectively. 


We collaborated with a rolling stock company to monitor the engine oil temperature and its impact factors by employing multiple sensors in locomotives. The collected factors were associated with operational, system-related, and environmental information, having 55 parameters in total. In this repository, we included a sample dataset ```XY_data.npz, Dataset_FeatureBased.csv ,selected_encoded_data.pkl```.

## Table - Input parameters collected from locomotives sensors
| Num | Parameter                                             | Num | Parameter                                             |
|-----|----------------------------------------------------|-----|----------------------------------------------------|
| 1   | Gear Position                                      | 28  | Bearing Temperature 4 (°C)                         |
| 2   | Locomotive Speed                                   | 29  | Bearing Temperature 5 (°C)                         |
| 3   | Altitude (m)                                       | 30  | Bearing Temperature 6 (°C)                         |
| 4   | Actual Diesel Engine Power (kW)                    | 31  | Low Temperature Radiator Inlet Water Temperature (°C) |
| 5   | Main Generator Excitation Power (kW)               | 32  | Thermocouple Compensation Temperature (°C)         |
| 6   | Main Generator DC Power (kW)                       | 33  | Intercooler Inlet Temperature (°C)                 |
| 7   | Auxiliary DC Power (kW)                            | 34  | Intercooler Outlet Temperature (°C)                |
| 8   | Wheel Circumference Power (kW)                     | 35  | Heat Exchanger Outlet Temperature (°C)             |
| 9   | Atmospheric Pressure (kPa)                         | 36  | After Compressor Temperature (°C)                  |
| 10  | Diesel Engine Speed Measured by Electric Spray (rpm) | 37  | Cylinder Head Exhaust Temperature 1 (°C)          |
| 11  | Engine Oil Inlet Pressure 1 (kPa)                  | 38  | Cylinder Head Exhaust Temperature 2 (°C)          |
| 12  | Engine Oil Inlet Pressure 2 (kPa)                  | 39  | Cylinder Head Exhaust Temperature 3 (°C)          |
| 13  | Crankcase Pressure 1 (Pa)                          | 40  | Cylinder Head Exhaust Temperature 4 (°C)          |
| 14  | Crankcase Pressure 2 (Pa)                          | 41  | Cylinder Head Exhaust Temperature 5 (°C)          |
| 15  | Engine Oil Outlet Pressure (kPa)                   | 42  | Cylinder Head Exhaust Temperature 6 (°C)          |
| 16  | High Temperature Water Pump Outlet Pressure (kPa)  | 43  | Cylinder Head Exhaust Temperature 7 (°C)          |
| 17  | Low Temperature Water Pump Outlet Pressure (kPa)   | 44  | Cylinder Head Exhaust Temperature 8 (°C)          |
| 18  | High Temperature Water Outlet Pressure (kPa)       | 45  | Cylinder Head Exhaust Temperature 9 (°C)          |
| 19  | Stable Pressure Box Pressure (kPa)                 | 46  | Cylinder Head Exhaust Temperature 10 (°C)         |
| 20  | Engine Oil Inlet Temperature (°C)                 | 47  | Cylinder Head Exhaust Temperature 11 (°C)         |
| 21  | Engine Oil Outlet Temperature (°C)                | 48  | Cylinder Head Exhaust Temperature 12 (°C)         |
| 22  | High Temperature Water Inlet Temperature (°C)      | 49  | Left Supercharger Speed (rpm)                     |
| 23  | High Temperature Water Outlet Temperature (°C)     | 50  | Right Supercharger Speed (rpm)                    |
| 24  | Stable Pressure Box Temperature (°C)               | 51  | Left Turbine Inlet Temperature (°C)               |
| 25  | Bearing Temperature 1 (°C)                         | 52  | Right Turbine Inlet Temperature (°C)              |
| 26  | Bearing Temperature 2 (°C)                         | 53  | Ambient Temperature (°C)                          |
| 27  | Bearing Temperature 3 (°C)                         | 54  | RFC1 Output Line Current Effective Value (IaA)    |
|     |                                                    | 55  | RFC2 Output Line Current Effective Value (IaA)    |



## Table - Test Scenario for determining hyperparameters of the model

| Test Scenario | Number of CNN layer | Number of LSTM layer | MSE | Note |
|---------------|----------------------|----------------------|-----|------|
| 1             | 1                    | 2                    | 0.497 | No attention |
| 2             | 0                    | 2                    | 1.187 | No Attention, Plain LSTM |
| 3             | 2                    | 0                    | 0.766 | No Attention, Plain CNN |
| 4             | 1                    | 1                    | 0.926 | No attention, Underfitting |
| 5             | 1                    | 3                    | 0.631 | No attention, Overfitting |
| 6             | 1                    | 2                    | 0.335 | With attention |


# Code
Our model was implemented using the following tool versions:
- Keras version 2.11.0
- Python 3.9

## Installation 
1. Install Python3.9
2. Install keras and other dependences
    ```
    pip install keras==2.11.0
    pip install scikit-learn pandas numpy xgboost bayesian-optimization
    ```

## Training and testing your model
To train and test MLP, Random Forests, and XGBoost, you can directly run the corresponding python file:
```
python my_MLP.py
```

For other models, i.e., CNN, LSTM, CNN-LSTM, CNN-LSTM with attention,
you will first need to run my_autoencoder.py to generate processed data and than run the corresponding file to train and test the models.
```
python my_autoencoder.py
python my_CNN_LSTM.py
```

# Paper
Prediction of the Temperature of Diesel Engine Oil in Railroad Locomotives Using Compressed Information-Based Data Fusion Method with Attention-Enhanced CNN-LSTM (Under review)
## Authors
Xin Wang, Rutgers University

Xiang Liu, Rutgers University

Yun Bai, Hong Kong University of Science and Technology (Guangzhou)