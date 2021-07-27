import numpy as np
from sklearn import metrics

# numpy metrics
def accuracy(y_true, y_pred):
    return np.mean(np.equal(np.argmax(y_pred, axis=-1), np.argmax(y_true, axis=-1)))

def l2_relative_error(y_true, y_pred):
    return np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true)

def nanl2_relative_error(y_true, y_pred):
    """Return the L2 relative error treating Not a Numbers (NaNs) as zero."""
    err = y_true - y_pred
    err = np.nan_to_num(err)
    y_true = np.nan_to_num(y_true)
    return np.linalg.norm(err) / np.linalg.norm(y_true)

def _absolute_percentage_error(y_true, y_pred):
    return 100 * np.abs(
        (y_true - y_pred) / np.clip(np.abs(y_true), np.finfo(np.float32).eps, None)
    )

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(_absolute_percentage_error(y_true, y_pred))

def max_absolute_percentage_error(y_true, y_pred):
    return np.amax(_absolute_percentage_error(y_true, y_pred))

def absolute_percentage_error_std(y_true, y_pred):
    return np.std(_absolute_percentage_error(y_true, y_pred))

def mean_squared_error_outlier(y_true, y_pred):
    error = np.ravel((y_true - y_pred) ** 2)
    error = np.sort(error)[: -len(error) // 1000]
    return np.mean(error)

mean_squared_error = metrics.mean_squared_error


def get(identifier):
    metric_identifier = {
        "accuracy": accuracy,
        "l2 relative error": l2_relative_error,
        "nanl2 relative error": nanl2_relative_error,
        "mean squared error": mean_squared_error,
        "MSE": mean_squared_error,
        "mse": mean_squared_error,
        "MAPE": mean_absolute_percentage_error,
        "max APE": max_absolute_percentage_error,
        "APE SD": absolute_percentage_error_std,
        "mse outlier": mean_squared_error_outlier,
    }
    if isinstance(identifier, str):
        return metric_identifier[identifier]
    elif callable(identifier):
        return identifier
    else:
        raise ValueError("Could not interpret metric function identifier:", identifier)