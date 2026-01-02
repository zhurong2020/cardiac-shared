# cardiac-shared PyPI Release Plan

**Created**: 2026-01-02
**Version**: 1.0
**Status**: Planning

---

## 1. Overview

This document outlines the plan to publish `cardiac-shared` to PyPI, including module migration from `cardiac-ml-research/shared/`.

### Goals
- Simplify dependency management across all cardiac projects
- Enable `pip install cardiac-shared` for easy deployment
- Separate generic tools (public) from core algorithms (private)

---

## 2. Current Status (v0.2.0)

### Already Migrated

| Module | Functions | Status |
|--------|-----------|--------|
| `hardware.detector` | `detect_hardware()`, `HardwareInfo` | ✅ Done |
| `hardware.cpu_optimizer` | `CPUOptimizer`, `apply_cpu_optimizations()` | ✅ Done |
| `hardware.profiles` | `GPU_PROFILES`, `GPUProfile` | ✅ Done |
| `hardware.optimizer` | `PerformanceOptimizer`, `get_optimal_config()` | ✅ Done |
| `environment.runtime_detector` | `detect_runtime()`, `RuntimeEnvironment` | ✅ Done |
| `io.dicom` | `read_dicom_series()`, `get_dicom_metadata()` | ✅ Done |
| `io.nifti` | `load_nifti()`, `save_nifti()` | ✅ Done |
| `io.zip_handler` | `extract_zip()`, `find_dicom_root()` | ✅ Done |

### Known Issues (Pre-PyPI)
- [ ] Version mismatch: pyproject.toml (0.1.0) vs __init__.py (0.2.0)
- [ ] URL incorrect: should be zhurong2020/cardiac-shared

---

## 3. Migration Plan

### Phase 1: Fix and Publish v0.2.0 (30 min)
- Fix version number in pyproject.toml
- Fix GitHub URL
- Test PyPI upload with `twine`
- Verify installation works

### Phase 2: Migrate Core Processing (2-3 hours) - Priority ⭐⭐⭐

| Source | Target | Description |
|--------|--------|-------------|
| `parallel_processor.py` | `cardiac_shared.parallel` | Parallel processing framework |
| `progress_tracker.py` | `cardiac_shared.progress` | Multi-level progress tracking |
| `cache_manager.py` | `cardiac_shared.cache` | Multi-level caching with resume |

### Phase 3: Migrate Batch & Config (1-2 hours) - Priority ⭐⭐

| Source | Target | Description |
|--------|--------|-------------|
| `processing/batch_processor.py` | `cardiac_shared.batch` | Generic batch processor |
| `config_manager.py` | `cardiac_shared.config` | YAML config management |
| `directory_manager.py` | `cardiac_shared.paths` | Directory structure management |

### Phase 4: Migrate Utilities (1 hour) - Priority ⭐

| Source | Target | Description |
|--------|--------|-------------|
| `prerequisite_checker.py` | `cardiac_shared.checks` | Environment checks |
| `src/.../utils/common_utils.py` | `cardiac_shared.utils` | Common utilities |

### Phase 5: Test & Release v1.0.0 (2 hours)
- Comprehensive testing
- Documentation update
- PyPI release

---

## 4. Modules to Keep Private

These modules contain business logic or IP and should NOT be included in PyPI:

| Module | Reason | Owner Project |
|--------|--------|---------------|
| `licensing/` | Commercial license system | Private projects |
| `authorization_manager.py` | Auth verification | Private projects |
| `revocation_checker.py` | License revocation | Private projects |
| `usage_tracker.py` | Usage tracking | Private projects |
| `models/ai_cac.py` | AI model (IP) | ai-cac-research |
| `setup_wizard.py` | Setup flow (commercial) | Private projects |
| `presets/hospital_cpu.py` | Hospital presets | cardiac-ml-research |

---

## 5. Target Structure (v1.0.0)

```
cardiac_shared/
├── __init__.py           # Version: 1.0.0
├── hardware/             # ✅ Existing
│   ├── detector.py
│   ├── cpu_optimizer.py
│   ├── profiles.py
│   └── optimizer.py
├── environment/          # ✅ Existing
│   └── runtime_detector.py
├── io/                   # ✅ Existing
│   ├── dicom.py
│   ├── nifti.py
│   └── zip_handler.py
├── parallel/             # 📦 New
│   ├── __init__.py
│   ├── processor.py
│   └── checkpoint.py
├── progress/             # 📦 New
│   ├── __init__.py
│   └── tracker.py
├── cache/                # 📦 New
│   ├── __init__.py
│   └── manager.py
├── batch/                # 📦 New
│   ├── __init__.py
│   └── processor.py
├── config/               # 📦 New
│   ├── __init__.py
│   └── manager.py
├── paths/                # 📦 New
│   ├── __init__.py
│   └── directory.py
├── checks/               # 📦 New
│   ├── __init__.py
│   └── prerequisites.py
└── utils/                # 📦 New
    ├── __init__.py
    └── common.py
```

---

## 6. Usage After Migration

```python
# Installation
pip install cardiac-shared

# Import examples
from cardiac_shared import detect_hardware, detect_runtime
from cardiac_shared.parallel import ParallelProcessor
from cardiac_shared.progress import ProgressTracker
from cardiac_shared.cache import CacheManager
from cardiac_shared.batch import BatchProcessor

# Hardware detection
hw = detect_hardware()
print(f"GPU: {hw.gpu.device_name}")

# Parallel processing with progress
processor = ParallelProcessor(max_workers=4)
with ProgressTracker(total=100) as tracker:
    results = processor.run(items, process_func, callback=tracker.update)
```

---

## 7. Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2025-12 | Initial release (hardware, environment, io) |
| 0.2.0 | 2026-01 | Added optimizer, profiles |
| 1.0.0 | TBD | Full migration (parallel, progress, cache, batch, config) |

---

## 8. References

- GitHub: https://github.com/zhurong2020/cardiac-shared
- PyPI: https://pypi.org/project/cardiac-shared/ (pending)
- Related: vbca, pcfa, ai-cac-research
