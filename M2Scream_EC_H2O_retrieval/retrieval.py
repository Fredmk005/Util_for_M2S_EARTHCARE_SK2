import numpy as np
from skretrieval.retrieval import RetrievalTarget
from utils import gaussian_filter
from prior import build_tikhonov_matrix

class H2OTarget(RetrievalTarget):
    """H2O retrieval target with log-normal state vector."""
    
    def __init__(self, engine, atmosphere, altitude_mask, apriori_vmr,
                 prior_covariance, y_error_matrix, measurement_mask,
                 wavelengths_nm, filter_center_nm, filter_fwhm_nm,
                 tikhonov_strength=1e-4, order=2, altitude_grid_m=None):
        """
        Initialize H2O retrieval target.
        
        Parameters
        ----------
        engine : sk.Engine
            Sasktran2 engine
        atmosphere : sk.Atmosphere
            Sasktran2 atmosphere
        altitude_mask : ndarray of bool
            Mask for retrieved altitudes
        apriori_vmr : ndarray
            A priori VMR on full altitude grid
        prior_covariance : ndarray (n_state, n_state)
            Prior covariance matrix
        y_error_matrix : ndarray (n_meas, n_meas)
            Measurement error covariance matrix
        measurement_mask : ndarray of bool
            Mask for valid measurements
        wavelengths_nm : ndarray
            Wavelength grid
        filter_center_nm, filter_fwhm_nm : float
            Spectral filter parameters
        tikhonov_strength : float
            Tikhonov regularization strength
        order : int
            Tikhonov order (1 or 2)
        altitude_grid_m : ndarray
            Full altitude grid in meters
        """
        self.atmosphere = atmosphere
        self.mask = altitude_mask
        self.engine = engine
        self._xa_full = np.asarray(apriori_vmr)
        self._xa = self._xa_full[self.mask]
        self._n_state = len(self._xa)
        self._wavelengths = np.asarray(wavelengths_nm)
        self._filter_transmission = gaussian_filter(
            self._wavelengths, filter_center_nm, filter_fwhm_nm
        )
        self._measurement_mask = measurement_mask
        self._y_error_matrix = y_error_matrix
        self._fixed_mask = measurement_mask
        self._n_wav = len(wavelengths_nm)
        self._n_view = None
        
        # Build inverse prior covariance with Tikhonov
        alt_km = altitude_grid_m[self.mask] / 1000.0
        L, W = build_tikhonov_matrix(alt_km, order=order)

        Sa_log = prior_covariance.copy()

        # Numerical symmetrization
        Sa_log = 0.5 * (Sa_log + Sa_log.T)

        # Small numerical floor
        Sa_log += 1e-8 *np.mean(np.diag(Sa_log)) * np.eye(self._n_state)

        Sa_inv = np.linalg.inv(Sa_log)

        # Tikhonov is also applied in log(VMR) space
        self._inv_Sa = Sa_inv + tikhonov_strength * L.T @ W @ L

        self._inv_Sa = 0.5 * (self._inv_Sa + self._inv_Sa.T)
    
    def state_vector(self):
        """Get current state vector (log VMR)."""
        vmr = self.atmosphere["H2O"].vmr[self.mask]
        return np.log(np.clip(vmr, 1e-20, 1.0))
    
    def update_state(self, x):
        """Update atmosphere with new state vector."""
        x = np.clip(x, -50.0, np.log(0.1))
        new_vmr = self.atmosphere["H2O"].vmr.copy()
        new_vmr[self.mask] = np.exp(x)
        self.atmosphere["H2O"].vmr = new_vmr
    
    def apriori_state(self):
        """Get a priori state vector."""
        return np.log(np.clip(self._xa, 1e-20, 1.0))
    
    def inverse_apriori_covariance(self):
        """Get inverse a priori covariance matrix."""
        return self._inv_Sa
    
    def lower_bound(self):
        """Get lower bounds for state vector."""
        return np.full(self._n_state, -50.0)
    
    def upper_bound(self):
        """Get upper bounds for state vector."""
        return np.full(self._n_state, np.log(0.1))
    
    def measurement_vector(self, l1_data):
        """Calculate measurement vector and Jacobian."""
        ds_l1 = l1_data.to_raw()
        rad = np.squeeze(ds_l1["radiance"].values)
        
        if rad.ndim != 2:
            raise ValueError(f"Radiance shape {rad.shape}, expected (wavelength, viewing)")
        
        n_wav, n_view = rad.shape
        self._n_view = n_view
        
        # Apply filter
        filtered = rad * self._filter_transmission[:, None]
        
        # Apply mask
        mask = self._fixed_mask
        if mask is None:
            raise ValueError("Measurement mask not set")
        
        mask_2d = mask.reshape(n_wav, n_view)
        y_full = filtered.reshape(-1)
        y = y_full[mask]
        
        # Calculate Jacobian
        output = self.engine.calculate_radiance(self.atmosphere)
        wf = np.squeeze(output["wf_H2O_vmr"].values)
        
        wf_ret = wf[self.mask, :, :]  # (n_state, n_wav, n_view)
        current_vmr = self.atmosphere["H2O"].vmr[self.mask]
        wf_log = wf_ret * current_vmr[:, None, None]  # log Jacobian
        wf_filtered = wf_log * self._filter_transmission[None, :, None]
        wf_flat = wf_filtered.reshape(self._n_state, n_wav * n_view).T
        K = wf_flat[mask, :]
        
        print(f"Jacobian shape: {K.shape}")
        print(f"Number of measurements used: {np.sum(mask)}")
        
        return {"y": y, "jacobian": K, "y_error": self._y_error_matrix}

class SasktranRadiance:
    """Wrapper for sasktran2 radiance output."""
    
    def __init__(self, ds):
        self.ds = ds
    
    def to_raw(self):
        return self.ds

class H2OSASKTRAN2ForwardModel:
    """Forward model for H2O retrieval."""
    
    def __init__(self, engine, atmosphere):
        self.engine = engine
        self.atmosphere = atmosphere
    
    def calculate_radiance(self):
        out = self.engine.calculate_radiance(self.atmosphere)
        return SasktranRadiance(out)