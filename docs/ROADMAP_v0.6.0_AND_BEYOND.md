# cardiac-shared Development Roadmap
# cardiac-shared 开发路线图

**Created**: 2026-01-03
**Current Version**: 0.5.1
**Next Version**: 0.6.0 (feature)
**Status**: v0.5.1 Released - Ready for v0.6.0

---

## Semantic Versioning Strategy

```
MAJOR.MINOR.PATCH
  │     │     └── Bug fixes, small optimizations (0.5.0 → 0.5.1)
  │     └──────── New features, modules (0.5.x → 0.6.0)
  └────────────── Breaking changes, API freeze (0.x → 1.0.0)
```

---

## 1. Current State Summary (v0.5.1)

### 1.1 Released Modules

| Module | Description | Version |
|--------|-------------|---------|
| `hardware` | GPU/CPU/RAM detection, CPUOptimizer, PerformanceOptimizer, GPU_PROFILES | v0.2.0 |
| `hardware.gpu_utils` | GPU stabilization time, performance tier | v0.5.1 |
| `environment` | Runtime detection (Colab/WSL/Windows/Linux) | v0.2.0 |
| `io` | DICOM, NIfTI, ZIP handling | v0.2.0 |
| `io.preloader` | AsyncNiftiPreloader, background preloading | v0.5.1 |
| `parallel` | ParallelProcessor with checkpoint/resume | v0.3.0 |
| `progress` | Multi-level ProgressTracker with ETA | v0.3.0 |
| `cache` | CacheManager with atomic writes | v0.3.0 |
| `batch` | BatchProcessor framework | v0.3.0 |
| `config` | ConfigManager for YAML/JSON | v0.3.0 |
| `data` | IntermediateResultsRegistry | v0.4.0 |
| `data_sources` | DataSourceManager (ZAL/CHD/Normal/Custom) | v0.5.0 |
| `vertebra` | VertebraDetector, ROI calculation | v0.5.0 |
| `tissue` | TissueClassifier (Alberta Protocol 2024) | v0.5.0 |

### 1.2 Placeholder Modules (Empty)

- `preprocessing/` - Reserved for future
- `utils/` - Reserved for future

---

## 2. Version 0.5.1 (RELEASED - 2026-01-03)

**Source**: RTX 4060 optimization research (2026-01-03)
**Objective**: Small optimizations for TotalSegmentator pipeline (~5-10% speedup)
**Scope**: No new modules, only enhancements to existing modules
**PyPI**: https://pypi.org/project/cardiac-shared/0.5.1/

### 2.1 Enhancement: `hardware/gpu_utils.py`

**Function**: `get_recommended_gpu_stabilization_time(gpu_info: dict = None) -> float`

**Purpose**: Replace hardcoded `time.sleep(5)` with dynamic GPU-aware wait time

**Logic**:
```python
GPU Model          | Wait Time | Reason
-------------------|-----------|---------------------------
RTX 4000+ series   | 1.0-2.0s  | Better memory management
RTX 3000 series    | 2.0-3.0s  | Standard performance
RTX 2000 series    | 3.0-4.0s  | Older architecture
Unknown/Other      | 5.0s      | Conservative default
```

**Usage**:
```python
from cardiac_shared.hardware import get_recommended_gpu_stabilization_time

wait_time = get_recommended_gpu_stabilization_time(gpu_info)
time.sleep(wait_time)  # Dynamic instead of hardcoded 5s
```

### 2.2 New Module: `io/preloader.py`

**Class**: `AsyncNiftiPreloader`

**Purpose**: Async preload next NIfTI file while processing current one

**Features**:
- Background thread preloading
- LRU memory cache (configurable size, default 2 files)
- Thread-safe operations
- Automatic resource cleanup

**Usage**:
```python
from cardiac_shared.io import AsyncNiftiPreloader

preloader = AsyncNiftiPreloader(max_cache_size=2)
preloader.start(nifti_file_list)

for nifti_file in nifti_file_list:
    ct_data = preloader.get(nifti_file)  # From cache or wait
    process(ct_data)  # Next file loading in background

preloader.stop()
```

### 2.3 Expected Benefits

| Optimization | Time Saved | Notes |
|--------------|------------|-------|
| GPU stabilization (5s→2s) | ~3-5% | ~6s per case |
| NIfTI async preload | ~2-3% | Eliminates I/O wait |
| **Total** | **~5-10%** | Precision unchanged |

**For 464 CHD cases**:
- Current: ~20 hours
- After v0.6.0: ~18-19 hours
- Savings: ~1-2 hours

### 2.4 Implementation Tasks (v0.5.1) - COMPLETED

```
[x] 1. Add to cardiac_shared/hardware/gpu_utils.py
    [x] 1.1 Implement get_recommended_gpu_stabilization_time()
    [x] 1.2 Add GPU model recognition (NVIDIA naming patterns)
    [x] 1.3 Unit tests (31 tests)

[x] 2. Add to cardiac_shared/io/preloader.py
    [x] 2.1 Implement AsyncNiftiPreloader class
    [x] 2.2 Thread-safe queue and cache
    [x] 2.3 Unit tests (7 tests)

[x] 3. Update exports and version
    [x] 3.1 Update __init__.py exports
    [x] 3.2 Bump version to 0.5.1
    [x] 3.3 Update CHANGELOG.md
    [x] 3.4 Release to PyPI (2026-01-03)
```

---

## 3. Version 0.6.0 Plan (MINOR - New Features)

**Source**: VBCA integration needs, FEEDBACK.md enhancement requests

### 3.1 DICOM Preprocessing Module (High Priority)

**Problem**: CHD dataset had 515→464 patients due to duplicates requiring manual cleanup

**New module**: `preprocessing/dicom.py`

```python
from cardiac_shared.preprocessing.dicom import (
    find_duplicate_patients,
    deduplicate_dicom_dirs,
    validate_dicom_series
)

# Find duplicates
duplicates = find_duplicate_patients(
    source_dir="D:/zhurong/chd",
    pattern=r"dicom_(\d+)",
    ignore_patterns=[".zip_"]
)

# Auto-deduplicate
deduplicate_dicom_dirs(source_dir, strategy='keep_original')
```

### 3.2 NIfTI Validation Module

**Problem**: No automatic validation after DICOM→NIfTI conversion

**Enhancement**: `io/nifti.py`

```python
from cardiac_shared.io.nifti import validate_nifti, NiftiValidationResult

result = validate_nifti(nifti_path)
print(f"Valid: {result.is_valid}")
print(f"Shape: {result.shape}")
print(f"Issues: {result.issues}")
```

### 3.3 External Datasets Registry

**Enhancement**: `data/external_registry.py`

- Stanford COCA (657 cases)
- TotalSegmentator v2.0 (1,672+ cases)
- NLST (26,254 cases)
- AMOS22 (600 cases)

---

## 4. Version 0.7.0 Plan (Future)

### 4.1 Path Management Module

**New module**: `paths/`

```python
from cardiac_shared.paths import PathManager

pm = PathManager(project='vbca')
pm.get_input_dir('zal')      # Auto Windows/WSL conversion
pm.get_output_dir('results')
pm.ensure_exists('output/labels')
```

### 4.2 Environment Checks Module

**New module**: `checks/`

- Prerequisite validation (GPU, RAM, disk space)
- Dependency checking

---

## 5. Version 1.0.0 Plan (Stable Release)

**Source**: ARCHITECTURE_DESIGN.md, PYPI_RELEASE_PLAN.md

### 5.1 Stability & API Freeze

- Finalize all public APIs
- Comprehensive documentation
- 100% test coverage for core modules

### 5.2 Additional Modules

| Module | Description | Priority |
|--------|-------------|----------|
| `utils/common.py` | Common utility functions | P3 |
| `logging/` | Unified logging configuration | P3 |

### 5.3 Refactor Private Projects

After v1.0.0 release:
- [ ] Update vbca to cardiac-shared v1.0.0
- [ ] Update pcfa to cardiac-shared v1.0.0
- [ ] Update ai-cac-research to cardiac-shared v1.0.0
- [ ] Standardize licensing module template

---

## 6. VBCA Integration Status

### 6.1 Current Integration (V3.4.1)

| Module | Status | Usage |
|--------|--------|-------|
| `hardware.detect_hardware()` | ✅ Used | GPU/CPU/RAM detection |
| `hardware.get_optimal_config()` | ✅ Used | Performance config |
| `environment.detect_runtime()` | ✅ Used | Runtime detection |
| `cache.CacheManager` | ✅ Used | Performance tracking |
| `progress.ProgressTracker` | ✅ Used | ETA display |
| `data_sources` | ❌ Not used | v0.5.0 new module |
| `vertebra` | ❌ Not used | v0.5.0 new module |
| `tissue` | ❌ Not used | v0.5.0 new module |

### 6.2 After v0.5.1 (VBCA V3.5)

| Task | From | To |
|------|------|-----|
| GPU wait time | `time.sleep(5)` | `get_recommended_gpu_stabilization_time()` |
| NIfTI loading | Sync `nib.load()` | `AsyncNiftiPreloader` |

### 6.3 CHD/Normal Dataset Processing

| Dataset | Cases | Current Est. | After v0.5.1 |
|---------|-------|--------------|--------------|
| CHD | 464 | ~20 hours | ~18-19 hours |
| Normal | 278 | ~12 hours | ~11 hours |

---

## 7. Development Environment

| Machine | GPU | Role |
|---------|-----|------|
| RTX 2060 | 6GB VRAM | Development, unit testing |
| RTX 4060 | 16GB VRAM | Production, integration testing |

**Workflow**:
1. Develop on RTX 2060
2. Unit test locally
3. Release to PyPI
4. Integration test on RTX 4060
5. Batch processing validation

---

## 8. Release Checklist

### For Each Release

```
[ ] Code complete and tested
[ ] Update CHANGELOG.md
[ ] Update version in pyproject.toml and __init__.py
[ ] Update README.md if needed
[ ] Run full test suite
[ ] Build: python -m build
[ ] Test local install: pip install dist/*.whl
[ ] Upload to PyPI: twine upload dist/*
[ ] Verify: pip install cardiac-shared==X.Y.Z
[ ] Tag git: git tag vX.Y.Z && git push --tags
[ ] Update dependent projects (vbca, etc.)
```

---

## 9. Summary: Version Roadmap

| Version | Type | Scope | Priority |
|---------|------|-------|----------|
| **v0.5.1** | PATCH | GPU优化 + 异步预加载 | HIGH |
| **v0.6.0** | MINOR | DICOM去重 + NIfTI验证 + 外部数据集 | MEDIUM |
| **v0.7.0** | MINOR | 路径管理 + 环境检查 | LOW |
| **v1.0.0** | MAJOR | API冻结 + 完整文档 | LOW |

---

## 10. Quick Reference: Command to Start

### For v0.5.1 (Small Optimizations)

```
I need to implement cardiac-shared v0.5.1 with small optimizations:

1. hardware/gpu_utils.py
   - get_recommended_gpu_stabilization_time() function
   - Returns 1-5 seconds based on GPU model

2. io/preloader.py (optional, if time permits)
   - AsyncNiftiPreloader class for background preloading

This is a PATCH release (v0.5.0 → v0.5.1), not a minor version bump.

Reference: cardiac-shared/docs/ROADMAP_v0.6.0_AND_BEYOND.md
```

### For v0.6.0 (New Features)

```
I need to implement cardiac-shared v0.6.0 with new features:

1. preprocessing/dicom.py - DICOM deduplication
2. io/nifti.py - Add validate_nifti() function
3. data/external_registry.py - External datasets registry

This is a MINOR release with new modules.

Reference: cardiac-shared/docs/ROADMAP_v0.6.0_AND_BEYOND.md
```

---

**Document Version**: 1.1
**Last Updated**: 2026-01-03
**Author**: Claude Code

*Generated with Claude Code*
