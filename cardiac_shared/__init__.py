"""
Cardiac Shared - Common utilities for cardiac imaging projects

This package provides shared IO, preprocessing, and utility functions
for cardiac imaging analysis across multiple projects.
"""

__version__ = "0.1.0"

from cardiac_shared.io.dicom import read_dicom_series, get_dicom_metadata
from cardiac_shared.io.nifti import load_nifti, save_nifti
from cardiac_shared.io.zip_handler import extract_zip, find_dicom_root

__all__ = [
    "read_dicom_series",
    "get_dicom_metadata",
    "load_nifti",
    "save_nifti",
    "extract_zip",
    "find_dicom_root",
]
