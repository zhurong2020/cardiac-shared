# Cardiac Shared

Shared utilities for cardiac imaging analysis projects.

**Version**: 0.2.0

## Installation

```bash
# Install with all features
pip install -e ".[all]"

# Install with DICOM support only
pip install -e ".[dicom]"

# Install with NIfTI support only
pip install -e ".[nifti]"
```

## Modules

### IO Module

| Function | Description |
|----------|-------------|
| `read_dicom_series(path)` | Read DICOM series from directory |
| `get_dicom_metadata(ds)` | Extract metadata from DICOM dataset |
| `load_nifti(path)` | Load NIfTI file with metadata |
| `save_nifti(volume, path)` | Save numpy array as NIfTI |
| `extract_zip(path)` | Context manager for ZIP extraction |
| `find_dicom_root(path)` | Find DICOM directory in extracted ZIP |

### Hardware Module (v0.2.0)

| Function | Description |
|----------|-------------|
| `detect_hardware()` | Detect complete hardware info (GPU/CPU/RAM) |
| `HardwareInfo` | Dataclass with GPU, CPU, RAM, environment info |
| `print_hardware_summary(hw)` | Print formatted hardware summary |
| `get_optimal_config(hw)` | Get optimal inference configuration |
| `CPUOptimizer` | CPU optimization for hospital deployments |
| `apply_cpu_optimizations(config)` | Apply PyTorch CPU optimizations |

### Environment Module (v0.2.0)

| Function | Description |
|----------|-------------|
| `detect_runtime()` | Detect runtime environment |
| `RuntimeEnvironment` | Dataclass with environment info |
| `detect_colab()` | Check if running in Google Colab |
| `detect_wsl()` | Check if running in WSL |
| `print_environment_summary(env)` | Print environment summary |

## Usage

### IO Operations

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

### Hardware Detection

```python
from cardiac_shared import detect_hardware, print_hardware_summary

hw = detect_hardware()
print_hardware_summary(hw)

print(f"Performance Tier: {hw.performance_tier}")
print(f"Recommended Device: {hw.recommended_device}")
print(f"GPU Available: {hw.gpu.available}")
print(f"CPU Cores: {hw.cpu.physical_cores}")
```

### Environment Detection

```python
from cardiac_shared import detect_runtime, print_environment_summary

env = detect_runtime()
print_environment_summary(env)

print(f"Runtime Type: {env.runtime_type}")
print(f"Is WSL: {env.is_wsl}")
print(f"Is Hospital Environment: {env.is_hospital_environment}")
```

### CPU Optimization (Hospital Deployment)

```python
from cardiac_shared import detect_hardware, detect_runtime, CPUOptimizer

hw = detect_hardware()
env = detect_runtime()

if env.is_hospital_environment and not hw.gpu.available:
    optimizer = CPUOptimizer()
    config = optimizer.get_optimal_config()

    print(f"CPU Tier: {config.tier.value}")
    print(f"Recommended Workers: {config.num_workers}")
    print(f"Batch Size: {config.batch_size}")

    # Apply PyTorch optimizations
    optimizer.apply_torch_optimizations(config)
```

## Projects Using This Package

- cardiac-ml-research (main project)
- ai-cac-research (CAC scoring research)
- pcfa (Pericardial Fat Analysis)
- vbca (Vertebra Body Composition Analysis)

## Changelog

### v0.2.0 (2026-01-01)
- Added `hardware` module (detector, cpu_optimizer)
- Added `environment` module (runtime_detector)
- Migrated from cardiac-ml-research/shared/

### v0.1.0 (2026-01-01)
- Initial release
- IO modules: dicom, nifti, zip_handler
