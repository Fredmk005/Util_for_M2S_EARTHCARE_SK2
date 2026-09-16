import numpy as np
from scipy.interpolate import interp1d

def create_covariance_smoothness(altitude_km,
                                 tropopause_km=12.0,
                                 sigma_log_trop=0.5,
                                 sigma_log_strat=0.3,
                                 corr_length_km=1.5):
    n = len(altitude_km)
    dz = np.diff(altitude_km)                 # (n-1,)

    # First-difference operator
    D = np.zeros((n - 1, n))
    for i in range(n - 1):
        D[i, i]     = -1.0 / dz[i]
        D[i, i + 1] =  1.0 / dz[i]

    # Prior variance profile (log space)
    z_trans = 1.0 / (1.0 + np.exp(-(altitude_km - tropopause_km) / 2.0))
    sigma_log = sigma_log_trop * (1 - z_trans) + sigma_log_strat * z_trans
    variance  = sigma_log**2

    # Sa^-1 = D^T diag(1/L_c^2) D + diag(1/variance)
    # where L_c is the correlation length 
    Lc = corr_length_km * np.ones(n - 1)
    Sa_inv = D.T @ np.diag(1.0 / Lc**2) @ D + np.diag(1.0 / variance)
    Sa = np.linalg.inv(Sa_inv)

    return 0.5 * (Sa + Sa.T)

def build_tikhonov_matrix(altitude_km, order=2, alt_pts=None, w_pts=None):
    """
    Build Tikhonov regularization matrix with altitude-dependent weights.
    """
    n_state = len(altitude_km)
    
    if order == 1:
        L = np.zeros((n_state - 1, n_state))
        for i in range(n_state - 1):
            L[i, i] = -1.0
            L[i, i+1] = 1.0
        reg_alt = (altitude_km[:-1] + altitude_km[1:]) / 2.0
    elif order == 2:
        L = np.zeros((n_state - 2, n_state))
        for i in range(n_state - 2):
            L[i, i] = 1.0
            L[i, i+1] = -2.0
            L[i, i+2] = 1.0
        reg_alt = altitude_km[1:-1]
    else:
        raise ValueError("order must be 1 or 2")
    
    # Altitude-dependent weights
    if alt_pts is None:
        alt_pts = [8.0, 12.0, 16.0, 20.0, 25.0]
        w_pts = [0.8, 1.0, 0.01, 0.001, 0.0001]
    
    w_interp = interp1d(alt_pts, w_pts, kind='linear', fill_value='extrapolate')
    reg_weights = np.clip(w_interp(reg_alt), 0.05, 5.0)
    W = np.diag(reg_weights)
    
    return L, W