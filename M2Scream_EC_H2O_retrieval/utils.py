import numpy as np
from scipy.interpolate import interp1d

def gaussian_filter(wavelength_nm, center_nm, fwhm_nm):
    """Create Gaussian spectral filter."""
    sigma = fwhm_nm / (2 * np.sqrt(2 * np.log(2)))
    return np.exp(-(wavelength_nm - center_nm)**2 / (2 * sigma**2))

def interp_log_profile(z_new, z_old, profile, floor=1e-20):
    """Interpolate profile in log space."""
    profile = np.asarray(profile)
    profile = np.maximum(profile, floor)
    return np.exp(np.interp(z_new, z_old, np.log(profile)))

def calculate_spectral_bin_width(instrument):
    """Calculate spectral bin width from instrument parameters."""
    if "resolving_power" in instrument:
        fwhm = instrument["littrow_wavelength_nm"] / instrument["resolving_power"]
        return fwhm / 2
    if "spectral_range_nm" in instrument and "image_size_horizontal_pixels" in instrument:
        return instrument["spectral_range_nm"] / instrument["image_size_horizontal_pixels"]
    raise ValueError("Cannot determine spectral bin width")

def smooth_transition(distance, transition_width):
    """Smooth sigmoid transition function."""
    return 1 / (1 + np.exp(-distance / (transition_width / 4)))