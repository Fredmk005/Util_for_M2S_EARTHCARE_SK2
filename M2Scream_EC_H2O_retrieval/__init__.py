"""
H2O Retrieval Package

A modular package for retrieving H2O profiles from limb-sounding measurements
using sasktran2 and skretrieval.
"""

from .config import INSTRUMENT, RETRIEVAL_CONFIG, MERRa_CONFIG, MIE_CONFIG
from .utils import gaussian_filter, interp_log_profile
from .noise import calculate_measurement_noise
from .prior import create_altitude_dependent_covariance, create_combined_prior
from .atmosphere import setup_hitran_databases, setup_sasktran_config, create_atmosphere
from .retrieval import H2OTarget, H2OSASKTRAN2ForwardModel, SasktranRadiance

__version__ = "1.0.0"
__all__ = [
    'INSTRUMENT', 'RETRIEVAL_CONFIG', 'MERRA_CONFIG', 'MIE_CONFIG',
    'gaussian_filter', 'interp_log_profile',
    'calculate_measurement_noise',
    'create_altitude_dependent_covariance', 'create_combined_prior',
    'setup_hitran_databases', 'setup_sasktran_config', 'create_atmosphere',
    'H2OTarget', 'H2OSASKTRAN2ForwardModel', 'SasktranRadiance'
]