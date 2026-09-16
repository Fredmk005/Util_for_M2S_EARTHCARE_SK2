import xarray
import numpy as np

def access_earthcare_file(Earthcare_file):
    """
    Takes a FILE with EarthCare data and transforms it into a list of xarray's.
    """
    file_path = Earthcare_file
    dataset = xarray.open_dataset(file_path, engine="h5netcdf", group="ScienceData")
    return dataset

def altitude_grid(along_track, ds):
    """
    Get altitude grid from EarthCARE data.
    """
    altitude_m_arr = ds["height"].isel(along_track=along_track).values[::-1]
    alt1 = np.ascontiguousarray(np.squeeze(np.maximum(altitude_m_arr, 0)), dtype=np.float64)
    return alt1

def aerosol_extinction(along_track, ds):
    """
    Get aerosol extinction from EarthCARE data.
    """
    aerosol_extinction_arr = ds["aerosol_extinction"].isel(along_track=along_track).fillna(0).values[::-1]
    aerosol_extinction = np.ascontiguousarray(np.squeeze(aerosol_extinction_arr), dtype=np.float64)
    return aerosol_extinction

def aerosol_type(along_track, ds):
    """
    Get aerosol type classification from EarthCARE data.
    """
    aerosol_type_arr = ds["aerosol_classification"].isel(along_track=along_track).values[::-1]
    aerosol_type = np.ascontiguousarray(np.squeeze(aerosol_type_arr), dtype=np.float64)
    return aerosol_type

def aerosol_extinction_by_type(along_track, ds):
    """
    Get aerosol extinction by type.
    0 = none
    1 = dust
    2 = sea salt
    3 = continental pollution
    4 = smoke
    5 = dusty smoke
    6 = dusty mix
    7 = stratospheric sulfate
    """
    masks = {}
    extinction_by_type = {}
    extinction = aerosol_extinction(along_track, ds)
    aero_type = aerosol_type(along_track, ds)
    for aerosol_class in range(8):
        masks[aerosol_class] = (aero_type == aerosol_class)
        extinction_by_type[aerosol_class] = np.where(
            masks[aerosol_class],
            extinction,
            0
        )
    return extinction_by_type

def liquid_extinction(along_track, ds):
    """
    Get liquid cloud extinction from EarthCARE data.
    """
    liquid_ext_arr = ds["liquid_extinction"].isel(along_track=along_track).values[::-1]
    liquid_ext = np.ascontiguousarray(np.squeeze(liquid_ext_arr), dtype=np.float64)
    return liquid_ext

def ice_extinction(along_track, ds):
    """
    Get ice cloud extinction from EarthCARE data.
    """
    ice_ext_arr = ds["ice_extinction"].isel(along_track=along_track).values[::-1]
    ice_ext = np.ascontiguousarray(np.squeeze(ice_ext_arr), dtype=np.float64)
    return ice_ext

def lat_lon(along_track, ds):
    """
    Get latitude and longitude from EarthCARE data.
    """
    lat = ds["latitude"].isel(along_track=along_track).item()
    lon = ds["longitude"].isel(along_track=along_track).item()
    return lat, lon

def particle_dist(along_track, ds, aerosol_class):
    """
    Get particle distribution parameters from EarthCARE data.
    Shape factor is 0.38 for lognormal liquid clouds.
    """
    shape_factors = {
        'liquid': 0.38,  # Water clouds
        'ice': 0.38,     # Ice clouds
    }
    
    if aerosol_class == "liquid":
        effective_radius = ds["liquid_effective_radius"].isel(along_track=along_track).values[::-1]
        effective_radius = np.nan_to_num(effective_radius, nan=0)
        mode_width = np.exp(shape_factors["liquid"])
        extinction = liquid_extinction(along_track, ds)
        median_radius = (effective_radius / (np.exp(2.5 * shape_factors["liquid"]**2))) * 1e9
        tot_ext = np.sum(extinction)
        if tot_ext > 0:
            median_radius_scalar = float(np.average(median_radius, weights=extinction))
        else:
            median_radius_scalar = np.nan
            print("no liquid cloud extinction")
            
    elif aerosol_class == "ice":
        effective_radius = ds["ice_effective_radius"].isel(along_track=along_track).values[::-1]
        effective_radius = np.nan_to_num(effective_radius, nan=0)
        mode_width = np.exp(shape_factors["ice"])
        extinction = ice_extinction(along_track, ds)
        median_radius = (effective_radius / (np.exp(2.5 * shape_factors['ice']**2))) * 1e9
        tot_ext = np.sum(extinction)
        if tot_ext > 0:
            median_radius_scalar = float(np.average(median_radius, weights=extinction))
        else:
            median_radius_scalar = np.nan
            print("no ice extinction")
    else:
        raise ValueError(f"Unknown aerosol class: {aerosol_class}")
    
    return {"mode_width": mode_width, "median_radius": median_radius_scalar}

def tropopause_dependent_error(along_track, ds, altitude_m):
    """
    Returns altitude and tropopause height dependent frac_uncert.
    Larger uncertainty under tropopause (500%), much smaller above (20%).
    
    Parameters:
    -----------
    along_track : int
        Along-track index
    ds : xarray.Dataset
        Earthcare dataset containing tropopause height
    altitude_m : np.ndarray
        Altitude grid in meters
    
    Returns:
    --------
    frac_uncert : np.ndarray
        Fractional uncertainty for each altitude
    tropopause_height_m : float
        Tropopause height in meters
    """
    # Get tropopause height in meters
    tropopause_height_m = ds["tropopause_height"].isel(along_track=along_track).values
    
    # Initialize fractional uncertainty array
    frac_uncert = np.ones(len(altitude_m))
    
    # Define uncertainties
    uncert_below_tropopause = 5.0
    uncert_above_tropopause = 0.1
    
    transition_halfwidth_m = 1000
    
    distance_from_tropo = altitude_m - tropopause_height_m
    
    def smooth_transition(distance, transition_width):
        """Smooth transition from 0 to 1 using sigmoid"""
        return 1 / (1 + np.exp(-distance / (transition_width / 4)))
    
    transition_factor = smooth_transition(distance_from_tropo, transition_halfwidth_m)
    
    frac_uncert = uncert_below_tropopause + (uncert_above_tropopause - uncert_below_tropopause) * transition_factor
    
    frac_uncert = np.clip(frac_uncert, 0.1, 10.0)
    
    return {"frac_uncert": frac_uncert, "tropopause_height_m": tropopause_height_m}

# Example usage (commented out)
# earthcare_file = "F:/level2b/ACM_CAP_2B/2024/12/01/BA/ECA_EXBA_ACM_CAP_2B_20241201T115332Z_20250911T092732Z_02902F/ECA_EXBA_ACM_CAP_2B_20241201T115332Z_20250911T092732Z_02902F.h5"
# ds = access_earthcare_file(earthcare_file)
# aero_type = aerosol_extinction_by_type(339, ds)