# cardiac-shared - Claude Code Project Conventions

**Last Updated**: 2026-01-05 | **Version**: 1.0

---

## Project Overview

**cardiac-shared** - Shared utilities for cardiac imaging analysis projects.

- **PyPI**: https://pypi.org/project/cardiac-shared/
- **Current Version**: 0.8.1
- **GitHub**: https://github.com/zhurong2020/cardiac-shared

---

## Quick Reference

| Category | Location |
|----------|----------|
| Source | `cardiac_shared/` |
| Tests | `tests/` |
| Config | `pyproject.toml` |
| Changelog | `CHANGELOG.md` |
| Roadmap | `docs/ROADMAP.md` |

---

## Core Principles

### 1. This is a PyPI Package
- All changes must be published to PyPI
- Version bump required for each release
- Maintain backward compatibility

### 2. Module Structure
```
cardiac_shared/
  io/           # DICOM, NIfTI, ZIP handling
  hardware/     # GPU/CPU detection, optimization
  environment/  # Runtime detection (WSL/Windows/Linux)
  data/         # Registry, batch management
  preprocessing/  # Unified pipelines
  progress/     # Progress tracking
  cache/        # Checkpoint management
  vertebra/     # Spine detection
  tissue/       # Tissue classification
```

### 3. Testing Requirements
```bash
# Run all tests before release
pytest tests/ -v

# Check coverage
pytest tests/ --cov=cardiac_shared
```

---

## Release Process

```bash
# 1. Update version in pyproject.toml
# 2. Update CHANGELOG.md
# 3. Build
python -m build

# 4. Upload to PyPI
twine upload dist/*

# 5. Tag and push
git tag v0.x.x
git push origin v0.x.x
```

---

## Consumers

This package is used by:
- `vbca` - Vertebral Body Composition Analysis
- `pcfa` - Pericardial & Cardiac Fat Analysis
- `ai-cac-research` - CAC Score Research
- `cardiac-ml-research` - Parent project

**Breaking changes require coordination with all consumers.**

---

## Development

```bash
# Install in development mode
pip install -e .

# Run tests
pytest tests/ -v
```
