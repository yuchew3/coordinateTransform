import pandas as pd
import numpy as np
import ca_data_utils

def load_data(step):
    v = ca_data_utils.load_v_matrix().T[9:39992:step]
    labels = ca_data_utils.load_labels()[9:39992:step]
    return v, labels
    


if __name__ == "__main__":
    X, labels = load_data(2)
    # X = X[:,0]
    lags = 10
    dataset = np.fromfunction(lambda x, y: pd.Series(X[x]).autocorr(lag=y+1), (len(X), lags))
    # dataset = np.zeros((len(X), lags))
    # for i in range(len(X)):
    #     series = pd.Series(X[i])
    #     for j in range(lags):
    #         corr = series.autocorr(lag=j+1)
    #         dataset[i, j] = corr
    print(dataset.shape)
