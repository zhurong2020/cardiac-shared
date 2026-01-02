# Cardiac Projects Architecture Design

**Created**: 2026-01-02
**Version**: 1.0
**Status**: Approved

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PyPI (Public)                                │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                    cardiac-shared                              │  │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ │  │
│  │  │hardware │ │  env    │ │   io    │ │parallel │ │ progress│ │  │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘ │  │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐             │  │
│  │  │  cache  │ │  batch  │ │ config  │ │  utils  │             │  │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘             │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                         pip install cardiac-shared
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      GitHub Private                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │
│  │    vbca     │  │    pcfa     │  │     ai-cac-research         │  │
│  │             │  │             │  │                             │  │
│  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ ┌─────────────┐│  │
│  │ │Algorithm│ │  │ │Algorithm│ │  │ │Algorithm│ │   Models    ││  │
│  │ │(Core IP)│ │  │ │(Core IP)│ │  │ │(Core IP)│ │  (Weights)  ││  │
│  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ └─────────────┘│  │
│  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐               │  │
│  │ │Licensing│ │  │ │Licensing│ │  │ │Licensing│               │  │
│  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘               │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Module Composition

### 2.1 Standard Project Structure (Post-Refactor)

Each private project should follow this structure:

```
{project}/
├── pyproject.toml          # Dependencies include cardiac-shared
├── requirements.txt        # cardiac-shared @ git+... or cardiac-shared>=1.0.0
│
├── src/                    # Core algorithms (IP)
│   ├── __init__.py
│   ├── algorithm.py        # Main algorithm implementation
│   ├── models/             # Model weights (if applicable)
│   └── ...
│
├── licensing/              # License validation (copied from template)
│   ├── __init__.py
│   ├── validator.py
│   └── keys/               # Public key for validation
│
├── cli/                    # Command-line interface
│   ├── __init__.py
│   └── main.py             # Uses cardiac_shared for hardware/progress
│
├── config/                 # Project-specific config
├── tests/                  # Unit tests
└── docs/                   # Documentation
```

### 2.2 Dependency Flow

```
cardiac-shared (PyPI)
    │
    ├── vbca
    │   ├── import cardiac_shared.hardware  → GPU detection
    │   ├── import cardiac_shared.parallel  → Batch processing
    │   ├── import cardiac_shared.progress  → Progress tracking
    │   └── import src.algorithm            → Core IP (local)
    │
    ├── pcfa
    │   ├── import cardiac_shared.hardware  → GPU detection
    │   ├── import cardiac_shared.io        → DICOM/NIfTI I/O
    │   ├── import cardiac_shared.cache     → Result caching
    │   └── import src.algorithm            → Core IP (local)
    │
    └── ai-cac-research
        ├── import cardiac_shared.hardware  → GPU detection
        ├── import cardiac_shared.batch     → Batch processor
        ├── import cardiac_shared.progress  → Progress tracking
        └── import src.models               → Core IP (local)
```

---

## 3. Refactoring Requirements

### 3.1 Changes Required for Private Projects

| Project | Current State | Required Changes | Effort |
|---------|---------------|------------------|--------|
| **vbca** | Uses cardiac_shared | Update import paths after v1.0.0 | Low |
| **pcfa** | Uses cardiac-ml-research/shared | Replace with cardiac_shared imports | Medium |
| **ai-cac-research** | Mixed | Standardize to cardiac_shared | Medium |

### 3.2 Import Migration Example

**Before (using local shared/):**
```python
from shared.hardware_manager import detect_hardware
from shared.parallel_processor import ParallelProcessor
from shared.progress_tracker import ProgressTracker
```

**After (using cardiac-shared PyPI):**
```python
from cardiac_shared import detect_hardware
from cardiac_shared.parallel import ParallelProcessor
from cardiac_shared.progress import ProgressTracker
```

### 3.3 Licensing Module

Each private project should have its own `licensing/` module (NOT from cardiac-shared):

```python
# In private project's licensing/validator.py
class LicenseValidator:
    """Project-specific license validation"""

    def __init__(self, project_name: str):
        self.project_name = project_name
        self.public_key = self._load_public_key()

    def validate(self, license_file: Path) -> bool:
        # RSA2048 signature verification
        # Machine ID binding
        # Expiration checking
        pass
```

---

## 4. Benefits of This Architecture

### 4.1 Single Module Release

| Benefit | Description |
|---------|-------------|
| **Independent versioning** | Each project can release independently |
| **Clear boundaries** | Core IP stays private, tools are shared |
| **Easy deployment** | `pip install project-name` (for licensed users) |
| **Reduced duplication** | Common code in one place |

### 4.2 Multi-Module Combination

```python
# Example: Combined pipeline using multiple modules
from cardiac_shared import detect_hardware
from cardiac_shared.parallel import ParallelProcessor
from cardiac_shared.progress import ProgressTracker

# Import specific algorithms
from vbca.algorithm import BodyCompositionAnalyzer
from pcfa.algorithm import PericardialFatAnalyzer
from ai_cac.algorithm import CalciumScorer

class CombinedCardiacPipeline:
    """
    Combined pipeline using all three modules.
    Only available to users licensed for all three.
    """

    def __init__(self):
        self.hw = detect_hardware()
        self.processor = ParallelProcessor()

        # Initialize each analyzer
        self.body_comp = BodyCompositionAnalyzer()
        self.pericardial = PericardialFatAnalyzer()
        self.calcium = CalciumScorer()

    def analyze(self, ct_scan: Path) -> dict:
        """Run all analyses on a CT scan"""
        results = {}

        with ProgressTracker(total=3, name="Combined Analysis") as tracker:
            results['body_composition'] = self.body_comp.analyze(ct_scan)
            tracker.update(1)

            results['pericardial_fat'] = self.pericardial.analyze(ct_scan)
            tracker.update(1)

            results['calcium_score'] = self.calcium.score(ct_scan)
            tracker.update(1)

        return results
```

### 4.3 Flexible Licensing

```
┌─────────────────────────────────────────────────────────────┐
│                    Licensing Options                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Option A: Single Module License                             │
│  ├── cardiac-shared (free, PyPI)                            │
│  └── vbca (licensed) ────────────────► Body Composition     │
│                                                              │
│  Option B: Multi-Module Bundle                               │
│  ├── cardiac-shared (free, PyPI)                            │
│  ├── vbca (licensed) ────────────────► Body Composition     │
│  └── pcfa (licensed) ────────────────► Pericardial Fat      │
│                                                              │
│  Option C: Full Suite License                                │
│  ├── cardiac-shared (free, PyPI)                            │
│  ├── vbca (licensed) ────────────────► Body Composition     │
│  ├── pcfa (licensed) ────────────────► Pericardial Fat      │
│  └── ai-cac (licensed) ──────────────► Calcium Scoring      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Migration Roadmap

### Phase 1: cardiac-shared PyPI (Current)
- [x] Create cardiac-shared repository
- [x] Migrate hardware/environment/io modules
- [ ] Fix version and URL issues
- [ ] Publish to PyPI

### Phase 2: Complete cardiac-shared (Next)
- [ ] Migrate parallel/progress/cache modules
- [ ] Migrate batch/config modules
- [ ] Release v1.0.0 to PyPI

### Phase 3: Refactor Private Projects
- [ ] Update vbca to use cardiac-shared v1.0.0
- [ ] Update pcfa to use cardiac-shared v1.0.0
- [ ] Update ai-cac-research to use cardiac-shared v1.0.0

### Phase 4: Standardize Licensing
- [ ] Create licensing template module
- [ ] Apply to all private projects
- [ ] Test license validation

---

## 6. Conclusion

This architecture provides:

1. **Clean separation** - Public tools vs Private algorithms
2. **Easy deployment** - Single `pip install` for tools
3. **IP protection** - Core algorithms stay private
4. **Flexibility** - Single or combined module licensing
5. **Maintainability** - Common code in one place
6. **Scalability** - Easy to add new modules

The refactoring effort is moderate but provides significant long-term benefits.
