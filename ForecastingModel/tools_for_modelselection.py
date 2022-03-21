from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score


def mean_sq_err(actual, prediction):
    mse = mean_squared_error(actual, prediction)
    return mse
