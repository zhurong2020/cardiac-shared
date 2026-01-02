# Cardiac Shared

[![PyPI version](https://badge.fury.io/py/cardiac-shared.svg)](https://pypi.org/project/cardiac-shared/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Shared utilities for cardiac imaging analysis projects.

**Version**: 0.4.0 | **PyPI**: https://pypi.org/project/cardiac-shared/

## Installation

```bash
# Install from PyPI (recommended)
pip install cardiac-shared

# Install with optional dependencies
pip install cardiac-shared[all]      # All optional deps
pip install cardiac-shared[dicom]    # DICOM support
pip install cardiac-shared[nifti]    # NIfTI support
pip install cardiac-shared[gpu]      # GPU/PyTorch support
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

### Hardware Module

| Function | Description |
|----------|-------------|
| `detect_hardware()` | Detect complete hardware info (GPU/CPU/RAM) |
| `HardwareInfo` | Dataclass with GPU, CPU, RAM, environment info |
| `print_hardware_summary(hw)` | Print formatted hardware summary |
| `get_optimal_config(hw)` | Get optimal inference configuration |
| `CPUOptimizer` | CPU optimization for hospital deployments |

### Environment Module

| Function | Description |
|----------|-------------|
| `detect_runtime()` | Detect runtime environment |
| `RuntimeEnvironment` | Dataclass with environment info |
| `detect_colab()` | Check if running in Google Colab |
| `detect_wsl()` | Check if running in WSL |

### Parallel Module (v0.3.0)

| Class/Function | Description |
|----------------|-------------|
| `ParallelProcessor` | Unified parallel processing framework |
| `parallel_map()` | Quick parallel map without checkpoint |
| `parallel_map_with_checkpoint()` | Parallel map with resume support |
| `ProcessingResult` | Result dataclass for each processed item |
| `Checkpoint` | Checkpoint data for resume capability |

### Progress Module (v0.3.0)

| Class/Function | Description |
|----------------|-------------|
| `ProgressTracker` | Multi-level progress visualization |
| `create_tracker()` | Create and start a progress tracker |
| `ProgressLevel` | Progress tracking for a single level |

### Cache Module (v0.3.0)

| Class | Description |
|-------|-------------|
| `CacheManager` | Multi-level caching with resume capability |

### Batch Module (v0.3.0)

| Class | Description |
|-------|-------------|
| `BatchProcessor` | Generic batch processing framework |
| `BatchConfig` | Batch processing configuration |

### Config Module (v0.3.0)

| Class/Function | Description |
|----------------|-------------|
| `ConfigManager` | YAML/JSON configuration management |
| `load_config()` | Load configuration with defaults |

### Data Module (v0.4.0)

| Class/Function | Description |
|----------------|-------------|
| `IntermediateResultsRegistry` | Cross-project data discovery and sharing |
| `RegistryEntry` | Dataclass for registry entries |
| `get_registry()` | Get singleton registry instance |

## Usage Examples

### Hardware Detection

```python
from cardiac_shared import detect_hardware, detect_runtime

hw = detect_hardware()
print(f"GPU: {hw.gpu.device_name if hw.gpu.available else 'None'}")
print(f"CPU Cores: {hw.cpu.physical_cores}")
print(f"RAM: {hw.ram.total_gb:.1f} GB")

env = detect_runtime()
print(f"Runtime: {env.runtime_type}")  # wsl, linux, windows, colab
```

### Parallel Processing with Checkpoint

```python
from cardiac_shared.parallel import ParallelProcessor

def process_patient(patient_id):
    # Your processing logic
    return {"id": patient_id, "status": "done"}

processor = ParallelProcessor(
    max_workers=4,
    checkpoint_file="results/checkpoint.json"
)

results = processor.map_with_checkpoint(
    process_patient,
    patient_list,
    desc="Processing patients"
)

processor.print_summary(results)
```

### Progress Tracking

```python
from cardiac_shared.progress import ProgressTracker

tracker = ProgressTracker()
tracker.start_overall("Processing Pipeline", total=100)

for i, item in enumerate(items):
    tracker.start_step(f"Step {i+1}", total=3)

    tracker.update_substep("Loading data")
    # ... load data
    tracker.update_step_progress()

    tracker.update_substep("Processing")
    # ... process
    tracker.update_step_progress()

    tracker.complete_step()
    tracker.update_overall(i + 1)

tracker.finish()
```

### Cache Management

```python
from cardiac_shared.cache import CacheManager

cache = CacheManager("results/cache.json")

for patient_id in patient_list:
    if cache.is_completed(patient_id):
        continue  # Skip already processed

    result = process_patient(patient_id)
    cache.mark_completed(patient_id, result)
```

### Configuration Management

```python
from cardiac_shared.config import ConfigManager

config = ConfigManager("config/settings.yaml")
db_host = config.get("database.host", default="localhost")
config.set("processing.batch_size", 32)
config.save()
```

### Intermediate Results Registry (v0.4.0)

```python
from cardiac_shared.data import get_registry

# Get singleton registry instance
registry = get_registry()

# Check if TotalSegmentator results are available
if registry.exists('segmentation.totalsegmentator_organs.chd_v2'):
    organs_path = registry.get_path('segmentation.totalsegmentator_organs.chd_v2')
    heart_mask = organs_path / patient_id / 'heart.nii.gz'

# Get metadata
meta = registry.get_metadata('body_composition.vbca_stage1_labels.zal_v3.2')
print(f"Patient count: {meta.get('patient_count')}")

# List available results
available = registry.list_available('segmentation')

# Get usage suggestion for a project
suggestion = registry.suggest_input('pcfa', 'heart_masks', 'chd')
```

## Projects Using This Package

- [vbca](https://github.com/zhurong2020/vbca) - Vertebral Body Composition Analysis
- cardiac-ml-research - Main research project
- pcfa - Pericardial Fat Analysis
- ai-cac-research - CAC scoring research

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for full version history.

### v0.4.0 (2026-01-02)
- Added `data` module (IntermediateResultsRegistry)
- Cross-project intermediate results discovery and sharing
- Automatic Windows/WSL path conversion
- Usage pattern suggestions per project

### v0.3.0 (2026-01-02)
- Added `parallel` module (ParallelProcessor, checkpoint/resume)
- Added `progress` module (ProgressTracker, multi-level)
- Added `cache` module (CacheManager)
- Added `batch` module (BatchProcessor)
- Added `config` module (ConfigManager)
- Published to PyPI

### v0.2.0 (2026-01-02)
- Added `hardware` module (detector, cpu_optimizer)
- Added `environment` module (runtime_detector)

### v0.1.0 (2025-12-01)
- Initial release with IO modules

## License

MIT License - see [LICENSE](LICENSE) for details.
