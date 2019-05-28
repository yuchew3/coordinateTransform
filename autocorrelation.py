import pandas as pd
import numpy as np
import ca_data_utils

def load_data(step):
    v = ca_data_utils.load_v_matrix().T[9:39992:step]
    labels = ca_data_utils.load_labels()[9:39992:step]
    return v, labels
    


if __name__ == "__main__":
    X, labels = load_data(2)
    X = X[:,::4]
    print(X.shape)
    length = 10
    lags = 5
    # dataset = np.fromfunction(lambda i, j: pd.Series(X[i]).autocorr(lag=j+1), (len(X), lags))
    # for i in range(X.shape[0]-length):
    #     series = pd.Series(X[:,i])
    #     for j in range(lags):
    #         corr = series.autocorr(lag=j+1)
    #         dataset[i, j] = corr

    dataset = np.zeros((X.shape[0]-length+1, X.shape[1]*lags))
    for i in range(dataset.shape[0]):
        x = X[i:i+length]
        for p in range(X.shape[1]):
            series = pd.Series(x[:,p])
            for j in range(lags):
                corr = series.autocorr(lag=j+1)
                dataset[i, p*lags+j] = corr

    np.save('../data/autocorr_2', dataset)
    print(dataset.shape)