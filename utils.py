import numpy as np

def combine_arima_svr(arima_pred, svr_pred):
    return arima_pred + svr_pred

def create_lag_features(data, lag=3):
    X, y = [], []
    for i in range(lag, len(data)):
        X.append(data[i-lag:i])
        y.append(data[i])
    return np.array(X), np.array(y)