"""
Cardiac Shared - Common utilities for cardiac imaging projects

This package provides shared IO, hardware detection, environment detection,
and utility functions for cardiac imaging analysis across multiple projects.

Modules:
- io: DICOM, NIfTI, and ZIP file handling
- hardware: Hardware detection and CPU optimization
- environment: Runtime environment detection
- preprocessing: Image preprocessing utilities (planned)
- utils: General utility functions (planned)
"""

__version__ = "0.2.0"

# IO modules
from cardiac_shared.io.dicom import read_dicom_series, get_dicom_metadata
from cardiac_shared.io.nifti import load_nifti, save_nifti
from cardiac_shared.io.zip_handler import extract_zip, find_dicom_root

# Hardware detection
from cardiac_shared.hardware import (
    detect_hardware,
    HardwareInfo,
    GPUInfo,
    CPUInfo,
    RAMInfo,
    print_hardware_summary,
    get_optimal_config,
    CPUOptimizer,
    get_cpu_optimizer,
    apply_cpu_optimizations,
)

# Environment detection
from cardiac_shared.environment import (
    detect_runtime,
    RuntimeEnvironment,
    detect_colab,
    detect_wsl,
    print_environment_summary,
)

__all__ = [
    # Version
    '__version__',

    # IO
    'read_dicom_series',
    'get_dicom_metadata',
    'load_nifti',
    'save_nifti',
    'extract_zip',
    'find_dicom_root',

    # Hardware
    'detect_hardware',
    'HardwareInfo',
    'GPUInfo',
    'CPUInfo',
    'RAMInfo',
    'print_hardware_summary',
    'get_optimal_config',
    'CPUOptimizer',
    'get_cpu_optimizer',
    'apply_cpu_optimizations',

    # Environment
    'detect_runtime',
    'RuntimeEnvironment',
    'detect_colab',
    'detect_wsl',
    'print_environment_summary',
]
