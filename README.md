# Cardiac Shared

Shared utilities for cardiac imaging analysis projects.

## Installation

```bash
# Install with all features
pip install -e ".[all]"

# Install with DICOM support only
pip install -e ".[dicom]"

# Install with NIfTI support only
pip install -e ".[nifti]"
```

## Usage

```python
from cardiac_shared.io import read_dicom_series, load_nifti, extract_zip, find_dicom_root

# Read DICOM series
volume, metadata = read_dicom_series("/path/to/dicom/")

# Read NIfTI file
volume, metadata = load_nifti("/path/to/file.nii.gz")

# Extract ZIP and read DICOM
with extract_zip("/path/to/data.zip") as extracted_dir:
    dicom_root = find_dicom_root(extracted_dir)
    volume, metadata = read_dicom_series(dicom_root)
```

## Modules

- `cardiac_shared.io.dicom` - DICOM reading and metadata extraction
- `cardiac_shared.io.nifti` - NIfTI loading and saving
- `cardiac_shared.io.zip_handler` - ZIP extraction with DICOM discovery

## Projects Using This Package

- cardiac-ml-research
- ai-cac-research
- pcfa (Pericardial Fat Analysis)
- vbca (Vertebra Body Composition Analysis)
