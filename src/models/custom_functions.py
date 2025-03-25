import numpy as np

def loss_custom(y_pred, y_real):

    #Ensuring np array:
    y_pred = np.array(y_pred)
    y_real = np.array(y_real)
    ponderation = np.where(y_real >= 0.5, 1.2, 1.0)
    sq_errs = ponderation * (y_pred-y_real)**2
    mean_samples = np.mean(sq_errs, axis=1)
    mean_all_sqrt = np.sqrt(np.mean(mean_samples))
    return mean_all_sqrt

