import numpy as np
import sasktran2 as sk
from sasktran2_ext.continuum import MTCKDContinuum
from config import MIE_CONFIG


def setup_hitran_databases(wavelength_min_nm, wavelength_max_nm):
    """Setup HITRAN databases for all required molecules."""
    start_wavenumber = 1e7 / wavelength_max_nm
    end_wavenumber = 1e7 / wavelength_min_nm

    return {
        "CH4": sk.database.HITRANDatabase(
            molecule="CH4",
            start_wavenumber=start_wavenumber,
            end_wavenumber=end_wavenumber,
            wavenumber_resolution=0.01,
            reduction_factor=1,
            backend="sasktran2",
            profile="voigt",
        ),
        "CO2": sk.database.HITRANDatabase(
            molecule="CO2",
            start_wavenumber=start_wavenumber,
            end_wavenumber=end_wavenumber,
            wavenumber_resolution=0.01,
            reduction_factor=1,
            backend="sasktran2",
            profile="voigt",
        ),
        "H2O": sk.database.HITRANDatabase(
            molecule="H2O",
            start_wavenumber=start_wavenumber,
            end_wavenumber=end_wavenumber,
            wavenumber_resolution=0.01,
            reduction_factor=1,
            backend="sasktran2",
            profile="voigt",
        ),
        "O3": sk.database.HITRANDatabase(
            molecule="O3",
            start_wavenumber=start_wavenumber,
            end_wavenumber=end_wavenumber,
            wavenumber_resolution=0.01,
            reduction_factor=1,
            backend="sasktran2",
            profile="voigt",
        ),
        "HCL": sk.database.HITRANDatabase(
            molecule="HCl",
            start_wavenumber=start_wavenumber,
            end_wavenumber=end_wavenumber,
            wavenumber_resolution=0.01,
            reduction_factor=1,
            backend="sasktran2",
            profile="voigt",
        ),
    }


def setup_sasktran_config(num_threads=12, num_streams=12):
    config = sk.Config()

    config.num_threads = num_threads
    config.num_streams = num_streams
    config.num_stokes = 1

    config.multiple_scatter_source = (
        sk.MultipleScatterSource.DiscreteOrdinates
    )
    config.single_scatter_source = (
        sk.SingleScatterSource.DiscreteOrdinates
    )
    config.emission_source = (
        sk.EmissionSource.DiscreteOrdinates
    )

    return config


def create_atmosphere(
    altitude_m,
    temperature_k,
    pressure_pa,
    vmr_profiles,
    hitran_dbs,
    wavelengths_nm,
    cloud_data=None,
    aerosol_data=None,
    cos_sza=1.0,
    config=None,
):

    if config is None:
        config = setup_sasktran_config()

    geom = sk.Geometry1D(
        cos_sza=cos_sza,
        solar_azimuth=0,
        earth_radius_m=6371000,
        altitude_grid_m=altitude_m,
        interpolation_method=sk.InterpolationMethod.LinearInterpolation,
        geometry_type=sk.GeometryType.Spherical,
    )

    atm = sk.Atmosphere(
        geom,
        config,
        wavelengths_nm=wavelengths_nm,
    )

    atm.temperature_k = temperature_k
    atm.pressure_pa = pressure_pa

    atm["rayleigh"] = sk.constituent.Rayleigh()
    atm["solar_irradiance"] = sk.constituent.SolarIrradiance()
    atm["emission"] = sk.constituent.ThermalEmission()

    atm["earth_surface"] = (
        sk.constituent.SurfaceThermalEmission(
            temperature_k=300,
            emissivity=0.9,
        )
    )

    atm["continuum"] = MTCKDContinuum()

    atm["albedo"] = sk.constituent.LambertianSurface(0.09)
    atm["surface"] = sk.constituent.LambertianSurface(0.07)

    for species in ["H2O", "O3", "HCL"]:
        atm[species] = (
            sk.constituent.vmraltitudeabsorber.VMRAltitudeAbsorber(
                hitran_dbs[species],
                altitude_m,
                vmr_profiles[species],
            )
        )

    for species in ["CH4", "CO2"]:
        atm[species] = sk.climatology.mipas.constituent(
            species,
            hitran_dbs[species],
        )


    mie_wav = np.unique(
        np.asarray(
            np.concatenate(
                [
                    MIE_CONFIG["wavelengths"]["short"],
                    MIE_CONFIG["wavelengths"]["long"],
                ]
            ),
            dtype=float,
        )
    )

    mie_wav.sort()

    def safe_add_scatterer(
        name,
        mie_db,
        alt_grid,
        ext_array,
        ref_wavelength,
    ):
        try:
            atm[name] = sk.constituent.ExtinctionScatterer(
                mie_db,
                alt_grid,
                ext_array,
                ref_wavelength,
            )

        except Exception:
            import traceback

            print(f"Warning: Could not add scatterer '{name}'")
            traceback.print_exc()

    if cloud_data is not None:

        liq_rad = cloud_data.get(
            "liquid_median_radius",
            np.nan,
        )

        liq_ext = cloud_data.get(
            "liquid_extinction",
            None,
        )

        liq_width = cloud_data.get(
            "liquid_mode_width",
            0.38,
        )

        if (
            liq_ext is not None
            and len(liq_ext) > 0
            and np.any(liq_ext > 0)
            and np.isfinite(liq_rad)
            and liq_rad > 0
        ):

            try:
                print(
                    f"Creating liquid Mie database: "
                    f"radius={liq_rad:.6e} m, "
                    f"mode_width={liq_width}"
                )

                liquid_distribution = sk.mie.distribution.LogNormalDistribution()

                mie_liq = sk.database.MieDatabase(
                    liquid_distribution.freeze(median_radius=liq_rad,mode_width=liq_width),
                    sk.mie.refractive.Water(),
                    wavelengths_nm=mie_wav,
                    num_threads=16,
                )

                safe_add_scatterer(
                    "clouds_water",
                    mie_liq,
                    cloud_data["altitude"],
                    liq_ext,
                    355,
                )

            except Exception:
                import traceback

                print(
                    "Warning: Could not create "
                    "liquid Mie database"
                )
                traceback.print_exc()

        ice_rad = cloud_data.get(
            "ice_median_radius",
            np.nan,
        )

        ice_ext = cloud_data.get(
            "ice_extinction",
            None,
        )

        ice_width = cloud_data.get(
            "ice_mode_width",
            0.38,
        )

        if (
            ice_ext is not None
            and len(ice_ext) > 0
            and np.any(ice_ext > 0)
            and np.isfinite(ice_rad)
            and ice_rad > 0
        ):

            try:
                print(
                    f"Creating ice Mie database: "
                    f"radius={ice_rad:.6e} m, "
                    f"mode_width={ice_width}"
                )

                ice_distribution = sk.mie.distribution.LogNormalDistribution()

                mie_ice = sk.database.MieDatabase(
                    ice_distribution.freeze(mode_width=ice_width,median_radius=ice_rad),
                    sk.mie.refractive.Ice(),
                    wavelengths_nm=mie_wav,
                    num_threads=16,
                )

                safe_add_scatterer(
                    "ice_cloud",
                    mie_ice,
                    cloud_data["altitude"],
                    ice_ext,
                    355,
                )

            except Exception:
                import traceback

                print(
                    "Warning: Could not create "
                    "ice Mie database"
                )
                traceback.print_exc()

    if aerosol_data is not None:

        sulf_ext = aerosol_data.get(
            "sulfate_extinction",
            None,
        )

        if (
            sulf_ext is not None
            and len(sulf_ext) > 0
            and np.any(sulf_ext > 0)
        ):

            try:
                sulf_rad = MIE_CONFIG["sulfate_radius"]
                sulf_width = MIE_CONFIG["sulfate_mode_width"]

                print(
                    f"Creating sulfate Mie database: "
                    f"radius={sulf_rad:.6e} m, "
                    f"mode_width={sulf_width}"
                )

                sulfate_distribution = sk.mie.distribution.LogNormalDistribution()

                mie_sulf = sk.database.MieDatabase(
                    sulfate_distribution.freeze(median_radius=sulf_rad,mode_width=sulf_width),
                    sk.mie.refractive.H2SO4(),
                    wavelengths_nm=mie_wav,
                    num_threads=16,
                )

                safe_add_scatterer(
                    "stratospheric_sulfate",
                    mie_sulf,
                    aerosol_data["altitude"],
                    sulf_ext,
                    355,
                )

            except Exception:
                import traceback

                print(
                    "Warning: Could not create "
                    "sulfate Mie database"
                )
                traceback.print_exc()

        dust_ext = aerosol_data.get(
            "dust_extinction",
            None,
        )

        if (
            dust_ext is not None
            and len(dust_ext) > 0
            and np.any(dust_ext > 0)
        ):

            try:
                dust_rad = MIE_CONFIG["dust_radius"]
                dust_width = MIE_CONFIG["dust_mode_width"]

                print(
                    f"Creating dust Mie database: "
                    f"radius={dust_rad:.6e} m, "
                    f"mode_width={dust_width}"
                )

                dust_distribution = sk.mie.distribution.LogNormalDistribution()

                mie_dust = sk.database.MieDatabase(
                    dust_distribution.freeze(mode_width=dust_width,median_radius=dust_rad),
                    sk.mie.refractive.Dust(),
                    wavelengths_nm=mie_wav,
                    num_threads=16,
                )

                safe_add_scatterer(
                    "dust_cloud",
                    mie_dust,
                    aerosol_data["altitude"],
                    dust_ext,
                    355,
                )

            except Exception:
                import traceback

                print(
                    "Warning: Could not create "
                    "dust Mie database"
                )
                traceback.print_exc()

    return atm, geom
