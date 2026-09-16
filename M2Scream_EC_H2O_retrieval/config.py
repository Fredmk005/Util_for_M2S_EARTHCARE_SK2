import numpy as np

H_PLANCK = 6.62607015e-34
C_LIGHT = 299792458.0
### from: 
#Spatial Heterodyne 
#Observations of Water
#Measurement Theoretical Basis 
#Document by langille, degenstein,bourassa,rieger
INSTRUMENT = {
    "satellite_altitude_m": 456000,
    "optical_axis_tangent_height_m": 22500,
    "distance_to_tangent_m": 2368.394e3,
    "littrow_wavelength_nm": 1362.0,
    "grating_ruling_lines_per_mm": 700,
    "littrow_angle_deg": np.rad2deg(0.497),
    "resolving_power": 55447.939,
    "spectral_range_nm": 12.576,
    "vertical_fov_deg": 0.96,
    "horizontal_fov_deg": 1.52,
    "image_size_vertical_pixels": 1024,
    "image_size_horizontal_pixels": 1280,
    "illuminated_pixels": 326,
    "grating_per_pixel_throughput": 7.08e-11 * 1e-4,
    "aft_optics_magnification": 3.4,
    "filter_center_nm": 1365.158,
    "filter_fwhm_nm": 2.0,
    "aliasing_left_nm": 0.979,
    "aliasing_right_nm": 0.965,
    "pixel_size_um": 20,
    "well_depth_e": 4000e3,
    "adc_bits": 13,
    "read_noise_e": 700.0,
    "dark_signal_e_per_s": 6.24e3, ### this is actually the +7C dark current, instrument is cooled to -14C so divide by2^2.1(approximate) the urd show document I was goin off of is an early draft so some details seem to be off
    "quantum_efficiency": 0.8,
    "integration_time_s": 20*60.0,
    "binning": 2,
    "modulation_efficiency": 0.9,
    "radiometric_relative_uncertainties": (0.01, 0.012),
    "min_snr": 5.0,
}

RETRIEVAL_CONFIG = {
    "altitude_step_m": 500.0,
    "altitude_min_m": 7000.0,
    "altitude_max_m": 27000.0,
    "viewing_alt_min_m": 2400.0,
    "viewing_alt_max_m": 42100.0,
    "wavelength_min_nm": 1355.741,
    "wavelength_max_nm": 1368.317,
    "wavelength_step_nm": 0.01,
    "tikhonov_strength": 10,
    "tikhonov_order": 2,
    "sigma_log_strat":1.0,
    "sigma_log_trop":0.4,
    "altitude_step_m_viewing": 100.0, ###ish, if each vertical line on sensor is a tangent altitude
    "earthcare_file": r"D:/level2b/ACM_CAP_2B/2024/12/01/BA/ECA_EXBA_ACM_CAP_2B_20241201T115332Z_20250911T092732Z_02902F/ECA_EXBA_ACM_CAP_2B_20241201T115332Z_20250911T092732Z_02902F.h5",
    "merra_folder": r"D:EarthAccessData"
}

MERRA_CONFIG = {
    "date": "2024-12-01 12:00:00",
    "timezone": "UTC",
    "earth_radius_m": 6371000 
}

MIE_CONFIG = {
    "wavelengths": {
        "short": np.arange(350, 360.5, 0.5),
        "long": np.arange(1353, 1371, 0.5)
    },
    "sulfate_radius": 300,
    "sulfate_mode_width": 1.42,
    "dust_radius": 80,
    "dust_mode_width": 1.8,
}