import sys
import os
import numpy as np
import pandas as pd
import sasktran2 as sk
from skretrieval.retrieval.rodgers import Rodgers
import pickle
import traceback

# Add the parent directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Local imports
from config import INSTRUMENT, RETRIEVAL_CONFIG, MERRA_CONFIG
from utils import interp_log_profile, gaussian_filter
from noise import calculate_measurement_noise
from prior import create_covariance_smoothness
from atmosphere import setup_hitran_databases, setup_sasktran_config, create_atmosphere
from retrieval import H2OTarget, H2OSASKTRAN2ForwardModel, SasktranRadiance

# Import external packages 
from External import MERRA2SCREAM as merra
from External import EARTHCARECLOUDS as ecc


def prepare_merra_data(along_track, dbecc, ds_merra):
    """Prepare MERRA-2 data for the given track."""
    lat, lon = ecc.lat_lon(along_track, dbecc)
    lonlat = merra._lon_lat(lon, lat, ds_merra)
    PTA = merra._PTA_grid(lonlat["lon_idx"], lonlat["lat_idx"], ds_merra)
    VMR = merra._species_vmr(lonlat["lon_idx"], lonlat["lat_idx"], ds_merra)
    
    return lat, lon, PTA, VMR


def interpolate_merra_to_grid(PTA, VMR, alt_step):
    """Interpolate MERRA-2 data to regular altitude grid."""
    alt_orig = PTA["altitude"]
    temp_orig = PTA["temp"]
    press_orig = PTA["pressure"]
    
    alt_min = RETRIEVAL_CONFIG["altitude_min_m"]
    alt_max = RETRIEVAL_CONFIG["altitude_max_m"]
    alt = np.arange(alt_min, alt_max + alt_step, alt_step)
    
    temp = np.interp(alt, alt_orig, temp_orig)
    press = np.exp(np.interp(alt, alt_orig, np.log(np.maximum(press_orig, 1e-20))))
    
    VMR_uniform = {}
    for sp in ["H2O", "O3", "HCL"]:
        VMR_uniform[sp] = interp_log_profile(alt, alt_orig, VMR[sp])
    
    return alt, temp, press, VMR_uniform


def get_cloud_data(along_track, dbecc):
    """Extract cloud data from EarthCARE."""
    alt_aero = ecc.altitude_grid(along_track, dbecc)
    ext_liq = ecc.liquid_extinction(along_track, dbecc)
    ext_ice = ecc.ice_extinction(along_track, dbecc)
    ext_by_type = ecc.aerosol_extinction_by_type(along_track, dbecc)
    
    dist_liquid = ecc.particle_dist(along_track, dbecc, "liquid")
    dist_ice = ecc.particle_dist(along_track, dbecc, "ice")
    
    cloud_data = {
        "altitude": alt_aero,
        "liquid_extinction": ext_liq,
        "ice_extinction": ext_ice,
        "liquid_median_radius": dist_liquid.get("median_radius", np.nan),
        "ice_median_radius": dist_ice.get("median_radius", np.nan),
        "liquid_mode_width": dist_liquid.get("mode_width", np.nan),
        "ice_mode_width": dist_ice.get("mode_width", np.nan),
    }
    
    aerosol_data = {
        "altitude": alt_aero,
        "sulfate_extinction": ext_by_type.get(7, 0),
        "dust_extinction": ext_by_type.get(1, 0),
    }
    
    return cloud_data, aerosol_data


def process_track(along_track, dbecc, ds_merra, hitran_dbs):
    """Process a single along-track."""
    print(f"\n{'='*70}\nProcessing along-track: {along_track}\n{'='*70}")
    
    try:
        # 1. Prepare data
        lat, lon, PTA, VMR = prepare_merra_data(along_track, dbecc, ds_merra)
        alt_step = RETRIEVAL_CONFIG["altitude_step_m"]
        alt, temp, press, VMR_uniform = interpolate_merra_to_grid(PTA, VMR, alt_step)
        
        # 2. Setup geometry
        cos_sza = merra.solar_cos_zenith(
            PTA.get("lat", lat), PTA.get("lon", lon),
            pd.Timestamp(MERRA_CONFIG["date"], tz=MERRA_CONFIG["timezone"])
        )
        print(f"Cos SZA: {cos_sza}")
        
        # 3. Create atmosphere
        wavelengths_nm = np.arange(
            RETRIEVAL_CONFIG["wavelength_min_nm"],
            RETRIEVAL_CONFIG["wavelength_max_nm"],
            RETRIEVAL_CONFIG["wavelength_step_nm"]
        )
        
        cloud_data, aerosol_data = get_cloud_data(along_track, dbecc)
        
        config = setup_sasktran_config(num_threads=12, num_streams=12)
        atm, geom = create_atmosphere(
            alt, temp, press, VMR_uniform, hitran_dbs,
            wavelengths_nm, cloud_data=cloud_data, aerosol_data=aerosol_data,
            cos_sza=cos_sza, config=config
        )
        
        # 4. Setup viewing geometry
        viewing_alts = np.arange(
            RETRIEVAL_CONFIG["viewing_alt_min_m"],
            RETRIEVAL_CONFIG["viewing_alt_max_m"],
            RETRIEVAL_CONFIG["altitude_step_m_viewing"]
        )
        viewing_geo = sk.ViewingGeometry()
        for ht in viewing_alts:
            ray = sk.TangentAltitudeSolar(
                tangent_altitude_m=ht, relative_azimuth=0,
                observer_altitude_m=INSTRUMENT["satellite_altitude_m"],
                cos_sza=cos_sza
            )
            viewing_geo.add_ray(ray)
        
        # 5. Calculate radiance
        engine = sk.Engine(config, geom, viewing_geo)
        output = engine.calculate_radiance(atm)
        print("Radiance Calculated")
        
        # 6. Process radiance and noise
        rad_full = np.squeeze(output.radiance.values)
        if rad_full.ndim != 2:
            raise ValueError(f"Radiance shape {rad_full.shape}, expected 2D")
        
        filter_trans = gaussian_filter(
            wavelengths_nm, INSTRUMENT["filter_center_nm"],
            INSTRUMENT["filter_fwhm_nm"]
        )
        
        filtered_rad, err_var_full, snr, mask = calculate_measurement_noise(
            rad_full, wavelengths_nm, filter_trans
        )
        
        if not np.any(mask):
            print(f"WARNING: No measurements with SNR > {INSTRUMENT['min_snr']}. Skipping.")
            return None
        
        y = filtered_rad.reshape(-1)[mask]
        y_err = np.maximum(err_var_full.reshape(-1)[mask], np.finfo(float).tiny)
        y_error_matrix = np.diag(y_err)
        
        print(f"Measurements after SNR mask: {len(y)}")
        
        # 7. Setup retrieval
        retrieval_mask = (alt >= RETRIEVAL_CONFIG["altitude_min_m"]) & (alt <= RETRIEVAL_CONFIG["altitude_max_m"])
        n_state = np.sum(retrieval_mask)
        alt_ret_km = alt[retrieval_mask] / 1000.0
        
        # Create prior
        np.random.seed(42)
        h2o_true = VMR_uniform["H2O"][retrieval_mask]
        prior_vmr = h2o_true * 0.80 * (1.0 + 0.05 * np.random.randn(n_state))
        
        # Get tropopause height
        tropo_data = ecc.tropopause_dependent_error(along_track, dbecc, alt)
        tropopause_km = tropo_data["tropopause_height_m"] / 1000.0
        
        Sa = create_covariance_smoothness(
            alt_ret_km,
            tropopause_km=tropopause_km,
            sigma_log_trop=RETRIEVAL_CONFIG['sigma_log_trop'],
            sigma_log_strat=RETRIEVAL_CONFIG['sigma_log_strat']
        )
        
        prior_full = np.zeros_like(alt, dtype=float)
        prior_full[retrieval_mask] = prior_vmr
        
        # 8. Run retrieval
        target = H2OTarget(
            engine, atm, retrieval_mask, prior_full, Sa,
            y_error_matrix, mask,
            wavelengths_nm=wavelengths_nm,
            filter_center_nm=INSTRUMENT["filter_center_nm"],
            filter_fwhm_nm=INSTRUMENT["filter_fwhm_nm"],
            tikhonov_strength=RETRIEVAL_CONFIG["tikhonov_strength"],
            order=RETRIEVAL_CONFIG["tikhonov_order"],
            altitude_grid_m=alt
        )
        
        fwd = H2OSASKTRAN2ForwardModel(engine, atm)
        l1_data = SasktranRadiance(output)
        
        minimizer = Rodgers(
            max_iter=10,
            lm_damping=0.1,
            convergence_factor=1.01,
            iterative_update_lm=True,
            apply_cholesky_scaling=True
        )
        
        print(f"\nStarting retrieval for track {along_track}")
        result = minimizer.retrieve(l1_data, fwd, target)
        
        if result is None or "xs" not in result or len(result["xs"]) == 0:
            print(f"Retrieval failed for track {along_track}")
            return None
        
        final_x = result["xs"][-1]
        if np.isscalar(final_x):
            print("Retrieval returned scalar, skipping.")
            return None
        
        print(f"Retrieval succeeded for track {along_track}")
        print(f"  Iterations: {len(result['xs'])}")
        
        # 9. Process results
        h2o_retrieved = prior_full.copy()
        h2o_retrieved[retrieval_mask] = np.exp(np.clip(final_x, -50, 50))
        
        A = result["averaging_kernel"]
        DOFS = np.trace(A)
        row_sums = np.sum(A, axis=1)
        
        result_data = {
            "h2o_retrieved": h2o_retrieved,
            "h2o_truth": VMR_uniform["H2O"],
            "h2o_apriori": prior_full,
            "altitude_grid": alt / 1000,
            "averaging_kernel": A,
            "DOFS": DOFS,
            "lat": lat,
            "lon": lon,
            "row_sums": row_sums,
            "tropopause_height_km": tropopause_km,
            "mask": mask,                              
            "snr": snr,                              
            "filtered_rad": filtered_rad,            
            "err_var": err_var_full,                 
            "wavelengths_nm": wavelengths_nm,          
            "viewing_altitude_grid": viewing_alts / 1000.0, 
            "min_snr": INSTRUMENT["min_snr"],
            "transmission": filter_trans, 
        }
        
        print(f"DOFS: {DOFS:.2f}/{A.shape[0]}")
        print(f"Successfully processed track {along_track}")
        
        return result_data
        
    except Exception as e:
        print(f"Error processing track {along_track}: {e}")
        traceback.print_exc()
        return None


def main():
    """Main execution function."""
    # Load data
    earthcare_file = RETRIEVAL_CONFIG["earthcare_file"]
    merra_folder = RETRIEVAL_CONFIG["merra_folder"]
    
    dbecc = ecc.access_earthcare_file(earthcare_file)
    ds_merra = merra.Access_MERRA2SCREAMDatasets(merra_folder)[-1]
    
    # Setup HITRAN databases
    hitran_dbs = setup_hitran_databases(
        RETRIEVAL_CONFIG["wavelength_min_nm"],
        RETRIEVAL_CONFIG["wavelength_max_nm"]
    )
    
    # Process tracks
    storage = {}
    atmo_track = [0] #np.concatenate([np.arange(0,1801,50),np.arange(2000,2301,50)])  # List of tracks to process
    
    for along_track in atmo_track:
        result = process_track(along_track, dbecc, ds_merra, hitran_dbs)
        if result is not None:
            storage[along_track] = result
    
    # Save results
    with open("storage.pkl", "wb") as f:
        pickle.dump(storage, f)
    print(f"\nSaved {len(storage)} tracks to storage.pkl")


if __name__ == "__main__":
    main()