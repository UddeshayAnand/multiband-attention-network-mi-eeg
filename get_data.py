"""Read the labeled BNCI MATLAB release of BCI Competition IV-2a."""
import os
import numpy as np
import scipy.io as sio


def load_BCI2a_data(data_path, subject, training, all_trials=True):
    """Load BCI Competition IV Dataset 2a for a single subject."""
    if subject not in range(1, 10):
        raise ValueError("subject must be between 1 and 9")
    n_channels   = 22
    n_tests      = 6 * 48
    window_length = 7 * 250          # 7 seconds @ 250 Hz

    fs = 250
    t1 = int(2 * fs)                 # crop 2-6 s (motor imagery window)
    t2 = int(6 * fs)

    class_return = np.zeros(n_tests)
    data_return  = np.zeros((n_tests, n_channels, window_length))

    NO_valid_trial = 0
    suffix = "T" if training else "E"
    file_path = os.path.join(data_path, f"A0{subject}{suffix}.mat")

    if not os.path.isfile(file_path):
        folder = "Training" if training else "Evaluation"
        file_path = os.path.join(data_path, folder, f"A0{subject}{suffix}.mat")
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Missing A0{subject}{suffix}.mat under {data_path}")
    a      = sio.loadmat(file_path)
    a_data = a['data']

    for ii in range(a_data.size):
        a_data1    = a_data[0, ii]
        a_data3    = a_data1[0, 0]
        a_X        = a_data3[0]
        a_trial    = a_data3[1]
        a_y        = a_data3[2]
        a_artifacts = a_data3[5]

        for trial in range(a_trial.size):
            if a_artifacts[trial] != 0 and not all_trials:
                continue
            start = int(a_trial[trial].item())
            data_return[NO_valid_trial, :, :] = np.transpose(
                a_X[start : start + window_length, :22]
            )
            class_return[NO_valid_trial] = int(a_y[trial].item())
            NO_valid_trial += 1

    data_return  = data_return[:NO_valid_trial, :, t1:t2]   # (N, 22, 1000)
    class_return = (class_return[:NO_valid_trial] - 1).astype(int)
    return data_return, class_return


def get_data(path, subject):
    """Return train/test arrays for one subject, shape (N, 22, 1000)."""
    X_train, y_train = load_BCI2a_data(path, subject, training=True)
    X_test,  y_test  = load_BCI2a_data(path, subject, training=False)
    return X_train, y_train, X_test, y_test
