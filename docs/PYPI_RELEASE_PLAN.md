# cardiac-shared PyPI Release Plan

**Created**: 2026-01-02
**Updated**: 2026-01-02
**Version**: 0.3.0
**Status**: ✅ PUBLISHED

---

## 1. Overview

This document outlines the plan to publish `cardiac-shared` to PyPI, including module migration from `cardiac-ml-research/shared/`.

### Goals
- ✅ Simplify dependency management across all cardiac projects
- ✅ Enable `pip install cardiac-shared` for easy deployment
- ✅ Separate generic tools (public) from core algorithms (private)

**PyPI URL**: https://pypi.org/project/cardiac-shared/

---

## 2. Current Status (v0.3.0) - RELEASED

### All Modules

| Module | Functions | Status |
|--------|-----------|--------|
| `hardware.detector` | `detect_hardware()`, `HardwareInfo` | ✅ Released |
| `hardware.cpu_optimizer` | `CPUOptimizer`, `apply_cpu_optimizations()` | ✅ Released |
| `hardware.profiles` | `GPU_PROFILES`, `GPUProfile` | ✅ Released |
| `hardware.optimizer` | `PerformanceOptimizer`, `get_optimal_config()` | ✅ Released |
| `environment.runtime_detector` | `detect_runtime()`, `RuntimeEnvironment` | ✅ Released |
| `io.dicom` | `read_dicom_series()`, `get_dicom_metadata()` | ✅ Released |
| `io.nifti` | `load_nifti()`, `save_nifti()` | ✅ Released |
| `io.zip_handler` | `extract_zip()`, `find_dicom_root()` | ✅ Released |
| `parallel.processor` | `ParallelProcessor`, `parallel_map()` | ✅ Released |
| `progress.tracker` | `ProgressTracker`, `create_tracker()` | ✅ Released |
| `cache` | `CacheManager` | ✅ Released |
| `batch` | `BatchProcessor`, `BatchConfig` | ✅ Released |
| `config` | `ConfigManager`, `load_config()` | ✅ Released |

---

## 3. Completed Migration

### Phase 1: Initial Release ✅
- [x] Fix version number in pyproject.toml
- [x] Fix GitHub URL
- [x] Create LICENSE file (MIT)
- [x] Create CHANGELOG.md

### Phase 2: Core Processing ✅
- [x] `parallel_processor.py` → `cardiac_shared.parallel`
- [x] `progress_tracker.py` → `cardiac_shared.progress`
- [x] `cache_manager.py` → `cardiac_shared.cache`

### Phase 3: Batch & Config ✅
- [x] `batch_processor.py` → `cardiac_shared.batch`
- [x] `config_manager.py` → `cardiac_shared.config`

### Phase 4: PyPI Release ✅
- [x] Build package with `python -m build`
- [x] Test local installation
- [x] Upload to PyPI with `twine`
- [x] Verify `pip install cardiac-shared` works

---

## 4. Modules Kept Private

These modules contain business logic or IP and are NOT included in PyPI:

| Module | Reason | Owner Project |
|--------|--------|---------------|
| `licensing/` | Commercial license system | Private projects |
| `authorization_manager.py` | Auth verification | Private projects |
| `revocation_checker.py` | License revocation | Private projects |
| `usage_tracker.py` | Usage tracking | Private projects |
| `models/ai_cac.py` | AI model (IP) | ai-cac-research |
| `setup_wizard.py` | Setup flow (commercial) | Private projects |

---

## 5. Current Structure (v0.3.0)

```
cardiac_shared/
├── __init__.py           # Version: 0.3.0
├── hardware/             # ✅ Released
│   ├── detector.py
│   ├── cpu_optimizer.py
│   ├── profiles.py
│   └── optimizer.py
├── environment/          # ✅ Released
│   └── runtime_detector.py
├── io/                   # ✅ Released
│   ├── dicom.py
│   ├── nifti.py
│   └── zip_handler.py
├── parallel/             # ✅ Released
│   ├── __init__.py
│   └── processor.py
├── progress/             # ✅ Released
│   ├── __init__.py
│   └── tracker.py
├── cache/                # ✅ Released
│   └── __init__.py
├── batch/                # ✅ Released
│   └── __init__.py
├── config/               # ✅ Released
│   └── __init__.py
├── preprocessing/        # Placeholder
└── utils/                # Placeholder
```

---

## 6. Installation

```bash
# Install from PyPI
pip install cardiac-shared

# Install with optional dependencies
pip install cardiac-shared[all]
pip install cardiac-shared[dicom]
pip install cardiac-shared[nifti]
pip install cardiac-shared[gpu]
```

---

## 7. Version History

| Version | Date | Changes | PyPI |
|---------|------|---------|------|
| 0.1.0 | 2025-12 | Initial (hardware, environment, io) | - |
| 0.2.0 | 2026-01-02 | Added optimizer, profiles | - |
| 0.3.0 | 2026-01-02 | Added parallel, progress, cache, batch, config | ✅ Published |

---

## 8. Future Plans

### Potential v1.0.0 additions
- [ ] `paths/` - Directory structure management
- [ ] `checks/` - Environment prerequisite checking
- [ ] `utils/common.py` - Additional utility functions

### Maintenance
- Update PyPI on each significant release
- Keep CHANGELOG.md updated
- Respond to GitHub issues

---

## 9. References

- **GitHub**: https://github.com/zhurong2020/cardiac-shared
- **PyPI**: https://pypi.org/project/cardiac-shared/
- **Token**: Stored in `~/.pypirc` (NOT in repo)

### Related Projects
- [vbca](https://github.com/zhurong2020/vbca) - Uses cardiac-shared>=0.3.0
- cardiac-ml-research - Main research project
- pcfa - Pericardial Fat Analysis
- ai-cac-research - CAC scoring research
