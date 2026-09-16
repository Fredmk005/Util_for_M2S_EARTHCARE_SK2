import xarray
import numpy as np
from pathlib import Path
from astropy.coordinates import EarthLocation, get_sun, AltAz
from astropy.time import Time
import astropy.units as u

def Access_MERRA2SCREAMDatasets(Earthaccess_folder):
    """
    Takes a folder with MERRA2SCREAM data and transforms it into a list of xarray's.
    """
    folder_path = Path(Earthaccess_folder)
    file_paths = [str(f.resolve()) for f in folder_path.iterdir() if f.is_file()]
    datasets = [xarray.open_mfdataset(file, engine="netcdf4") for file in file_paths]
    return datasets

def Access_MERRA2SCREAMUncerts(Earthaccess_folder):
    """
    Takes a folder with MERRA2SCREAM data and transforms it into a list of xarray's with uncertainty data.
    """
    folder_path = Path(Earthaccess_folder)
    file_paths = [str(f.resolve()) for f in folder_path.iterdir() if f.is_file()]
    datasets = [xarray.open_mfdataset(file, engine="netcdf4", decode_times=False) for file in file_paths]
    return datasets

def _lon_lat(lon, lat, dataset):
    """
    Find nearest grid indices for given longitude and latitude.
    """
    lon_grid = dataset.coords["lon"].values
    lat_grid = dataset.coords["lat"].values

    lon = ((lon + 180) % 360) - 180

    lon_idx = np.argmin(np.abs(lon_grid - lon))
    lat_idx = np.argmin(np.abs(lat_grid - lat))

    return {
        "lon_idx": int(lon_idx),
        "lat_idx": int(lat_idx),
        "lon": lon_grid[lon_idx],
        "lat": lat_grid[lat_idx]
    }

def _PTA_grid(longitude_idx, latitude_idx, dataset):
    """
    Creates Pressure, temperature, altitude grid.
    Returns dict with pressure, temp and altitude.
    """
    altitude_m_arr = dataset["H"].isel(lat=latitude_idx, lon=longitude_idx, lev=slice(None, None, -1)).values
    pressure_Pa_arr = dataset["PL"].isel(lat=latitude_idx, lon=longitude_idx, lev=slice(None, None, -1)).values
    temperature_K_arr = dataset["T"].isel(lat=latitude_idx, lon=longitude_idx, lev=slice(None, None, -1)).values
    
    pressure_Pa = np.ascontiguousarray(np.squeeze(pressure_Pa_arr), dtype=np.float64)
    temperature_K = np.ascontiguousarray(np.squeeze(temperature_K_arr), dtype=np.float64)
    altitude_m = np.ascontiguousarray(np.squeeze(altitude_m_arr), dtype=np.float64)
    
    return dict(pressure=pressure_Pa, temp=temperature_K, altitude=altitude_m)

def _species_vmr(longitude_idx, latitude_idx, dataset, uncert_dataset=None):
    """
    Extract species VMR profiles from MERRA2-SCREAM dataset.
    
    Inputs
    ------
    longitude_idx : int
        Longitude index in the MERRA2-SCREAM dataset.
    latitude_idx : int
        Latitude index in the MERRA2-SCREAM dataset.
    dataset : xarray.Dataset
        Dataset containing QV, O3, HNO3, HCL, and N2O profiles.
    uncert_dataset : xarray.Dataset, optional
        Dataset containing sigma_qv, the specific-humidity uncertainty.

    Outputs
    -------
    dict
        Dictionary containing O3, HNO3, N2O, HCL, H2O VMR profiles
        and H2O_Uncert, the corresponding H2O VMR 1-sigma uncertainty.
    """
    M_v = 18.01528
    M_q = 28.9644
    epsilon = M_v / M_q

    qv = np.ascontiguousarray(
        np.squeeze(
            dataset["QV"]
            .isel(
                lat=latitude_idx,
                lon=longitude_idx,
                lev=slice(None, None, -1),
            )
            .values
        ),
        dtype=np.float64,
    )

    qv = np.maximum(qv, 0.0)

    denominator = (
        epsilon
        + qv * (1.0 - epsilon)
    )

    h2o_vmr = (
        qv / denominator
    )
    
    if uncert_dataset is not None:
        sigma_qv = np.ascontiguousarray(
            np.squeeze(
                uncert_dataset["sigma_qv"]
                .isel(
                    lat=latitude_idx,
                    lon=longitude_idx,
                    lev=slice(None, None, -1),
                )
                .values
            ),
            dtype=np.float64,
        )

        sigma_qv = np.maximum(
            sigma_qv,
            0.0,
        )

        dh2o_dqv = (
            epsilon
            / denominator**2
        )

        h2o_vmr_uncert = (
            np.abs(dh2o_dqv)
            * sigma_qv
        )
    else:
        h2o_vmr_uncert = None

    o3_vmr = np.ascontiguousarray(
        np.squeeze(
            dataset["O3"]
            .isel(
                lat=latitude_idx,
                lon=longitude_idx,
                lev=slice(None, None, -1),
            )
            .values
        ),
        dtype=np.float64,
    )

    hno3_vmr = np.ascontiguousarray(
        np.squeeze(
            dataset["HNO3"]
            .isel(
                lat=latitude_idx,
                lon=longitude_idx,
                lev=slice(None, None, -1),
            )
            .values
        ),
        dtype=np.float64,
    )

    hcl_vmr = np.ascontiguousarray(
        np.squeeze(
            dataset["HCL"]
            .isel(
                lat=latitude_idx,
                lon=longitude_idx,
                lev=slice(None, None, -1),
            )
            .values
        ),
        dtype=np.float64,
    )

    n2o_vmr = np.ascontiguousarray(
        np.squeeze(
            dataset["N2O"]
            .isel(
                lat=latitude_idx,
                lon=longitude_idx,
                lev=slice(None, None, -1),
            )
            .values
        ),
        dtype=np.float64,
    )

    return {
        "O3": o3_vmr * 1e-6,
        "HNO3": hno3_vmr,
        "N2O": n2o_vmr,
        "HCL": hcl_vmr,
        "H2O": h2o_vmr,
        "H2O_Uncert": h2o_vmr_uncert,
    }

def solar_cos_zenith(lat, lon, timestamp):
    """
    Returns cosine of solar zenith angle as float from latitude, longitude and timestamp input.
    """
    t = Time(timestamp, scale="utc")

    loc = EarthLocation(lat=lat * u.deg, lon=lon * u.deg)

    altaz = get_sun(t).transform_to(
        AltAz(obstime=t, location=loc)
    )

    cos_sza = np.sin(altaz.alt)

    return cos_sza.value