# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.1] - 2026-01-03

### Added
- GPU utilities module (`cardiac_shared.hardware.gpu_utils`)
  - `get_recommended_gpu_stabilization_time()` - Dynamic GPU wait time (1-5s based on GPU model)
  - `get_gpu_performance_tier()` - Quick GPU classification (high/medium/low)
  - `GPU_STABILIZATION_TIMES` - Lookup table for known GPU models
- Async NIfTI preloader (`cardiac_shared.io.preloader`)
  - `AsyncNiftiPreloader` - Background preloading with LRU cache
  - `preload_nifti_batch()` - Convenience function for batch processing
  - Thread-safe operations with statistics tracking

### Performance
- GPU stabilization time reduced from 5s to 1-2s for RTX 40 series
- Expected 5-10% speedup for TotalSegmentator pipelines
- For 464 CHD cases: ~20h → ~18-19h (~1-2h savings)

---

## [0.5.0] - 2026-01-03

### Added
- Data sources management module (`cardiac_shared.data_sources`)
  - `DataSourceManager` - Multi-source data management
  - Support for ZAL, CHD, Normal, Custom datasets
  - Automatic environment detection (WSL/Windows/Linux)
  - YAML configuration support
- Vertebra detection module (`cardiac_shared.vertebra`)
  - `VertebraDetector` - Parse TotalSegmentator output
  - ROI calculation for body composition analysis
  - Center slice and volume calculation
- Tissue classification module (`cardiac_shared.tissue`)
  - `TissueClassifier` with Alberta Protocol 2024 HU ranges
  - SAT/VAT/IMAT/Muscle filtering
  - Muscle quality assessment (Goodpaster criteria)

### Changed
- Version bumped to 0.5.0
- Added 44 new unit tests

---

## [0.4.0] - 2026-01-02

### Added
- Data registry module (`cardiac_shared.data`)
  - `IntermediateResultsRegistry` - Cross-project data discovery and sharing
  - `RegistryEntry` - Data class for registry entries
  - `get_registry()` - Singleton access function
  - Dot-notation key access (e.g., 'segmentation.totalsegmentator_organs.chd_v2')
  - Automatic Windows/WSL path conversion
  - Usage pattern suggestions for each project
  - Path validation and existence checking
  - Consumer tracking for impact analysis

### Changed
- Version bumped to 0.4.0
- Added pyyaml as core dependency
- Updated module documentation

---

## [0.3.0] - 2026-01-02

### Added
- Parallel processing module (`cardiac_shared.parallel`)
  - `ParallelProcessor` - Unified parallel processing framework
  - `parallel_map()`, `parallel_map_with_checkpoint()` convenience functions
  - Auto worker detection based on CPU/memory
  - Checkpoint/resume support for long-running tasks
- Progress tracking module (`cardiac_shared.progress`)
  - `ProgressTracker` - Multi-level progress visualization
  - `create_tracker()` convenience function
  - ETA estimation and time formatting
- Cache management module (`cardiac_shared.cache`)
  - `CacheManager` - Multi-level caching with resume capability
  - Atomic JSON writes for crash resistance
- Batch processing module (`cardiac_shared.batch`)
  - `BatchProcessor` - Generic batch processing framework
  - Flexible callback system
- Configuration module (`cardiac_shared.config`)
  - `ConfigManager` - YAML/JSON configuration management
  - Environment variable substitution
  - Dot-notation key access

### Changed
- Version bumped to 0.3.0
- Added tqdm as optional dependency for progress bars

---

## [0.2.0] - 2026-01-02

### Added
- Hardware detection module (`cardiac_shared.hardware`)
  - `detect_hardware()` - Detect GPU, CPU, and RAM
  - `HardwareInfo`, `GPUInfo`, `CPUInfo`, `RAMInfo` data classes
  - `CPUOptimizer` for CPU-only deployments
  - `PerformanceOptimizer` for optimal configuration
  - `GPU_PROFILES` for common GPU configurations
- Environment detection module (`cardiac_shared.environment`)
  - `detect_runtime()` - Detect Colab, WSL, Windows, Linux
  - `RuntimeEnvironment` data class
- IO module (`cardiac_shared.io`)
  - `read_dicom_series()` - Read DICOM series
  - `get_dicom_metadata()` - Extract DICOM metadata
  - `load_nifti()`, `save_nifti()` - NIfTI file operations
  - `extract_zip()`, `find_dicom_root()` - ZIP handling

### Changed
- Updated Python version support to 3.8+
- Added psutil as core dependency

## [0.1.0] - 2025-12-01

### Added
- Initial release
- Basic project structure
- Hardware detection foundation
- IO utilities foundation
