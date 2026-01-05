# Versioning Policy / 版本管理策略

This project follows [Semantic Versioning 2.0.0](https://semver.org/).

## Version Format

```
MAJOR.MINOR.PATCH
```

## Version Increment Rules

| Change Type | Version Change | Examples |
|-------------|----------------|----------|
| **PATCH** (0.x.Y) | Bug fixes, typo corrections, data corrections | Fix calculation error, correct patient count |
| **MINOR** (0.X.0) | New features, new modules (backward compatible) | Add new utility module, add new function |
| **MAJOR** (X.0.0) | Breaking changes, API redesign | Remove/rename public API, change behavior |

## Pre-1.0 Development

During pre-1.0 development (0.x.y):
- MINOR version changes may include minor breaking changes
- PATCH versions are strictly backward compatible
- API is not considered stable until 1.0.0

## Release Checklist

Before releasing a new version:
1. [ ] Run all tests (`pytest`)
2. [ ] Update CHANGELOG.md with changes
3. [ ] Update version in pyproject.toml
4. [ ] Update version badge in README.md
5. [ ] Commit with message: `chore(release): vX.Y.Z`
6. [ ] Create git tag: `git tag vX.Y.Z`
7. [ ] Build and upload to PyPI

## Yanked Versions

The following versions have been yanked (not recommended for use):

| Version | Reason | Replacement |
|---------|--------|-------------|
| 0.7.0 | Hardcoded internal data in PyPI (privacy concern) | 0.8.1+ |
| 0.7.1 | Hardcoded internal data in PyPI (privacy concern) | 0.8.1+ |
| 0.8.0 | Rushed release, version number inconsistency | 0.8.1+ |

## Version History Summary

| Version | Date | Type | Description |
|---------|------|------|-------------|
| 0.6.4 | 2026-01-04 | PATCH | BatchDiscovery module |
| 0.7.0 | 2026-01-04 | MINOR | DatasetRegistry (yanked - hardcoded data) |
| 0.7.1 | 2026-01-04 | PATCH | Patient count fix (yanked) |
| 0.8.0 | 2026-01-04 | MAJOR | Config-driven refactor (yanked - rushed) |
| 0.8.1 | 2026-01-04 | PATCH | Stable config-driven release |
