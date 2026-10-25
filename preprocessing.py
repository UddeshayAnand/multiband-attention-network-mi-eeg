"""Notebook CutCat augmentation, splitting, and Butterworth filtering."""
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split


def augment_with_cutcat(X_data, y_data, expansion_factor=25):
    """
    CutCat augmentation: splice 4 quarter-segments from two same-class trials.
    Pattern: [A_seg1, B_seg2, A_seg3, B_seg4]

    Args:
        X_data          : (N, channels, time)
        y_data          : (N,)
        expansion_factor: number of synthetic trials per original trial
    Returns:
        X_final, y_final: original + synthetic concatenated
    """
    print(f"CutCat augmentation (factor={expansion_factor})...")
    if expansion_factor < 0:
        raise ValueError("expansion_factor must be nonnegative")
    if expansion_factor == 0:
        return X_data.astype(np.float32), y_data.copy()
    n_trials, n_channels, n_time = X_data.shape
    quarter_len = n_time // 4

    # Group indices by class
    class_groups = {c: np.where(y_data == c)[0] for c in np.unique(y_data)}

    X_aug, y_aug = [], []
    for i in range(n_trials):
        label   = y_data[i]
        trial_a = X_data[i]
        for _ in range(expansion_factor):
            idx_b   = np.random.choice(class_groups[label])
            trial_b = X_data[idx_b]

            new_trial = np.concatenate([
                trial_a[:, 0             : quarter_len],
                trial_b[:, quarter_len   : 2*quarter_len],
                trial_a[:, 2*quarter_len : 3*quarter_len],
                trial_b[:, 3*quarter_len :              ],
            ], axis=1)

            X_aug.append(new_trial)
            y_aug.append(label)

    X_aug = np.array(X_aug, dtype=np.float32)
    y_aug = np.array(y_aug, dtype=int)

    X_final = np.concatenate([X_data.astype(np.float32), X_aug], axis=0)
    y_final = np.concatenate([y_data, y_aug], axis=0)
    print(f"  Before: {X_data.shape[0]} trials  →  After: {X_final.shape[0]} trials")
    return X_final, y_final


def apply_band_filters(X_data, fs=250, order=4):
    """
    Band-pass filter EEG data.
    Input shape : (N, time, channels)
    Returns     : tuple of filtered arrays (delta, theta, alpha, beta, gamma)
    """
    # Remove DC offset
    X_data = X_data - np.mean(X_data, axis=1, keepdims=True)

    bands = {
        'delta': (0.5,  4),
        'theta': (4,    8),
        'alpha': (8,   13),
        'beta' : (13,  30),
        'gamma': (30,  80),   # kept below 100 to avoid Nyquist issues at 250 Hz
    }

    n_samples, n_time, n_channels = X_data.shape
    # Reshape to (samples*channels, time) for filtering along time axis
    X_reshaped = X_data.transpose(0, 2, 1).reshape(-1, n_time)

    band_data = {}
    for name, (low, high) in bands.items():
        nyq = 0.5 * fs
        Wn  = [low / nyq, high / nyq]
        b, a = butter(order, Wn, btype='bandpass')
        padlen   = min(150, n_time // 2)
        filtered = filtfilt(b, a, X_reshaped, axis=1, padlen=padlen)
        # Reshape back: (N, time, channels)
        band_data[name] = filtered.reshape(n_samples, n_channels, n_time).transpose(0, 2, 1)

    return (band_data['delta'], band_data['theta'],
            band_data['alpha'], band_data['beta'], band_data['gamma'])


def make_inputs(X):
    """Four streams in the notebook order: raw, theta, alpha, beta."""
    _, theta, alpha, beta, _ = apply_band_filters(X)
    return [X, theta, alpha, beta]


def prepare_training_data(X, y, expansion_factor=25, val_fraction=0.1, seed=42):
    """Preserve the notebook's augmentation-before-validation-split protocol."""
    X_aug, y_aug = augment_with_cutcat(X, y, expansion_factor)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_aug.transpose(0, 2, 1), y_aug, test_size=val_fraction,
        stratify=y_aug, random_state=seed,
    )
    return make_inputs(X_tr), make_inputs(X_val), y_tr, y_val
