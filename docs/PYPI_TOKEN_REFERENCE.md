# PyPI Token Reference

**Purpose**: Document PyPI token configuration (WITHOUT actual token values)
**Security**: Actual tokens stored in `~/.pypirc` only

---

## Current Token Configuration

### Token: cardiac-shared-project

| Property | Value |
|----------|-------|
| **Name** | cardiac-shared-project |
| **Scope** | Project: cardiac-shared |
| **Created** | 2026-01-02 |
| **Created By** | Rong Zhu |
| **Status** | Active |
| **Storage** | `~/.pypirc` (WSL) |

---

## Token Management

### Location
```
~/.pypirc                    # Linux/WSL/macOS
C:\Users\{user}\.pypirc      # Windows (if needed)
```

### File Format
```ini
[distutils]
index-servers =
    pypi

[pypi]
username = __token__
password = pypi-***PROJECT_TOKEN***
```

### Permissions
```bash
chmod 600 ~/.pypirc   # Linux/WSL/macOS only
```

---

## Usage

### Upload to PyPI
```bash
# From project directory
cd /home/wuxia/projects/cardiac-shared
python -m build
twine upload dist/*
```

### Verify Token Works
```bash
twine check dist/*
```

---

## Security Best Practices

1. **Never commit tokens** - `.pypirc` is NOT in git
2. **Use project-scoped tokens** - Limits damage if leaked
3. **Rotate tokens** - Delete and recreate if compromised
4. **One token per project** - Easier to manage and revoke

---

## Token Rotation Procedure

If token is compromised:

1. **Immediately** go to https://pypi.org/manage/account/
2. Click "API tokens"
3. Click "Remove" on compromised token
4. Click "Add API token"
5. Name: `cardiac-shared-project-YYYYMMDD`
6. Scope: `Project: cardiac-shared`
7. Copy new token
8. Update `~/.pypirc`
9. Verify: `twine upload dist/*`

---

## History

| Date | Action | Note |
|------|--------|------|
| 2026-01-02 | Created | Project-scoped token for cardiac-shared |

---

**IMPORTANT**: This document does NOT contain actual token values.
Tokens are stored securely in `~/.pypirc` only.
