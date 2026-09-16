import numpy as np
from config import INSTRUMENT, H_PLANCK, C_LIGHT
from utils import calculate_spectral_bin_width


def calculate_measurement_noise(radiance_spectrum, wavelengths_nm, transmission):
    """
    Compute filtered radiance, error variance, SNR and mask.

    Assumptions
    -----------
    - radiance_spectrum : (n_wav, n_view) spectral radiance, mean/expected
      value from a radiative transfer model.
    - Each element of the wavelength axis is treated as ONE independent
      spectral bin with its own detector pixel.
    - INSTRUMENT["read_noise_e"]      : read noise per BINNED pixel,
      cumulative after binning (e.g. 700 e- for low gain 2x2, 70 e- for
      high gain 2x2). No further binning factor is applied.
    - INSTRUMENT["dark_signal_e_per_s"] : dark current per BINNED pixel
      at the operating temperature. Set via temperature scaling if only
      the datasheet value at 280 K is known.
    - INSTRUMENT["well_depth_e"]      : full-well capacity per BINNED pixel.
    - INSTRUMENT["adc_bits"]          : ADC bit depth.
    """
    radiance_spectrum = np.asarray(radiance_spectrum).squeeze()
    wavelengths_nm    = np.asarray(wavelengths_nm)
    transmission      = np.asarray(transmission)

    if radiance_spectrum.ndim != 2:
        raise ValueError(f"Expected (wavelength, viewing), got {radiance_spectrum.shape}")

    n_wav, n_view = radiance_spectrum.shape
    rt_bin = np.median(np.diff(wavelengths_nm))

    binning  = INSTRUMENT["binning"]
    raw_px   = INSTRUMENT["image_size_horizontal_pixels"]
    raw_py   = INSTRUMENT["image_size_vertical_pixels"]

    if raw_px % binning or raw_py % binning:
        raise ValueError("Detector dimensions must be divisible by binning")
    n_binned = (raw_px // binning) * (raw_py // binning)

    total_throughput = (INSTRUMENT["grating_per_pixel_throughput"]* INSTRUMENT["illuminated_pixels"])
    wavelength_m     = INSTRUMENT["filter_center_nm"] * 1e-9
    photon_energy    = H_PLANCK * C_LIGHT / wavelength_m
    system_eff       = (INSTRUMENT["modulation_efficiency"]* INSTRUMENT["quantum_efficiency"])
    t_int  = INSTRUMENT["integration_time_s"]

    filtered = radiance_spectrum * transmission[:, None]    # (n_wav, n_view)


    integrated    = np.sum(filtered, axis=0) * rt_bin
    integrated    = np.maximum(integrated, 1e-10)
    photon_rate   = integrated / photon_energy * total_throughput
    signal_e      = photon_rate * t_int * system_eff        # (n_view,)

    signal_per_bin = (filtered * rt_bin / photon_energy * total_throughput * t_int * system_eff)
    signal_per_bin = np.maximum(signal_per_bin, 1e-10)     

    dark_rate     = INSTRUMENT["dark_signal_e_per_s"]/(2**2.1)
    dark_e_pix    = dark_rate * t_int                     

    read_var_pix  = INSTRUMENT["read_noise_e"] ** 2      

    # Quantisation
    if "well_depth_e" in INSTRUMENT:
        adc_step      = INSTRUMENT["well_depth_e"] / (2 ** INSTRUMENT["adc_bits"])
        quant_var_pix = (adc_step ** 2) / 12.0
    else:
        quant_var_pix = 0.0


    shot_var_per_bin  = signal_per_bin                   
    dark_var_per_bin  = np.full_like(signal_per_bin, dark_e_pix)
    read_var_per_bin  = np.full_like(signal_per_bin, read_var_pix)
    quant_var_per_bin = np.full_like(signal_per_bin, quant_var_pix)

    total_var_per_bin = (shot_var_per_bin
                         + dark_var_per_bin
                         + read_var_per_bin
                         + quant_var_per_bin)

    snr_per_bin = np.abs(signal_per_bin) / np.sqrt(np.maximum(total_var_per_bin, 1e-30))

    shot_var  = np.maximum(signal_e, 0.0)                
    dark_var  = n_wav * dark_e_pix
    read_var  = n_wav * read_var_pix
    quant_var = n_wav * quant_var_pix

    total_var_e = shot_var + dark_var + read_var + quant_var

    snr_int = np.abs(signal_e) / np.sqrt(np.maximum(total_var_e, 1e-30))

    rad_per_e = photon_energy / (total_throughput * rt_bin * t_int * system_eff)

    err_var = total_var_per_bin * rad_per_e ** 2          

    radiometric_sigma = float(np.sqrt(np.sum(
        np.square(INSTRUMENT["radiometric_relative_uncertainties"])
    )))
    err_var = err_var + (filtered * radiometric_sigma) ** 2

    err_var = np.maximum(
        np.nan_to_num(err_var, nan=np.inf, posinf=np.inf),
        np.finfo(float).tiny
    )

    mask_per_bin = snr_per_bin > INSTRUMENT["min_snr"]
    mask         = mask_per_bin.reshape(-1)

    good_altitudes = np.sum(mask_per_bin, axis=0) > 0

    print("\n" + "=" * 60)
    print("SHS SIGNAL CALCULATION")
    print("=" * 60)
    print(f"Spectral bin width: {calculate_spectral_bin_width(INSTRUMENT):.5f} nm")
    print(f"Number SNR range: {np.nanmin(snr_per_bin):.3e} to {np.nanmax(snr_per_bin):.3e}")
    print(f"Integrated SNR range: {np.nanmin(snr_int):.3e} to {np.nanmax(snr_int):.3e}")
    print(f"Number wavelengths with SNR > {INSTRUMENT['min_snr']}: "
          f"{np.count_nonzero(mask_per_bin)}")
    print(f"Number viewing altitudes with good SNR: "
          f"{np.sum(good_altitudes)}/{n_view}")
    print(f"Total measurements with SNR > {INSTRUMENT['min_snr']}: "
          f"{np.count_nonzero(mask)}")
    print("=" * 60 + "\n")
    test_scale = 1/20 ### TESTING SCALE VALUE
    err_var = err_var*test_scale
    return filtered, err_var, snr_per_bin, mask