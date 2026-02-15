# Feedback & Bug Reports

**Purpose**: Record bugs, issues, and enhancement requests discovered during testing
**Usage**: Add entries to this file, commit and push, or create GitHub Issues

---

## How to Report

### Quick Format
```
## [BUG/FEATURE/DOC] Short Title
- **Date**: YYYY-MM-DD
- **Reporter**: Your name/machine
- **Version**: cardiac-shared version
- **Module**: affected module (e.g., data.registry)
- **Priority**: P0 (critical) / P1 (high) / P2 (medium) / P3 (low)
- **Status**: NEW / IN_PROGRESS / RESOLVED / WONTFIX
- **Description**: What happened / what's needed
- **Steps to Reproduce**: (for bugs)
- **Expected vs Actual**: (for bugs)
- **Suggested Fix**: (optional)
```

### Alternative: GitHub Issues
- https://github.com/zhurong2020/cardiac-shared/issues

---

## Pending Issues

<!-- Add new issues below this line -->

### [BUG] Example - Registry path not found on Windows
- **Date**: 2026-01-02
- **Reporter**: Example
- **Version**: 0.4.0
- **Module**: data.registry
- **Priority**: P1
- **Status**: NEW
- **Description**: Registry cannot find config file when running from different directory
- **Steps to Reproduce**:
  1. Install cardiac-shared
  2. Run `from cardiac_shared.data import get_registry`
  3. Registry shows warning about missing config
- **Expected**: Should find config via project root detection
- **Actual**: Warning about missing config
- **Suggested Fix**: Add fallback to user home directory

---

## Testing Checklist

### RTX 4060 Deployment Test (2026-01-02)
- [ ] `pip install cardiac-shared>=0.4.0` works
- [ ] `from cardiac_shared import detect_hardware` works
- [ ] `detect_hardware()` correctly identifies RTX 4060
- [ ] `from cardiac_shared.data import get_registry` works
- [ ] Registry can load config from vbca project
- [ ] Path conversion (Windows/WSL) works correctly
- [ ] vbca can process sample case end-to-end

### Module Tests
- [ ] IO module (DICOM, NIfTI)
- [ ] Hardware module (GPU detection)
- [ ] Environment module (WSL detection)
- [ ] Parallel module (multiprocessing)
- [ ] Progress module (tracker)
- [ ] Cache module
- [ ] Batch module
- [ ] Config module
- [ ] Data registry module

---

## Resolved Issues

<!-- Move resolved issues here -->

---

## Enhancement Requests

<!-- Feature requests go here -->

### [FEATURE] Add external datasets registry
- **Date**: 2026-01-02
- **Priority**: P1
- **Description**: Extend registry to support external datasets (Stanford COCA, TotalSegmentator, etc.)
- **Rationale**: ai-cac-research has validated many datasets with specific naming conventions

---

## Notes

- When testing on new machine, focus on:
  1. Installation process
  2. Hardware detection accuracy
  3. Path handling (especially Windows vs WSL)
  4. Config file discovery

- For urgent bugs, also create GitHub Issue for visibility

---

**Last Updated**: 2026-01-02
