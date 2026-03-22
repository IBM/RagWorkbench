---
name: Release Checklist
about: Checklist for preparing a new release
title: 'Release v[VERSION]'
labels: 'release'
assignees: ''
---

## Release Information

**Version**: v[VERSION]
**Target Date**: [DATE]
**Release Type**: [ ] Major [ ] Minor [ ] Patch [ ] Pre-release

## Pre-Release Checklist

### Code Quality
- [ ] All tests pass locally (`pytest`)
- [ ] Pre-commit checks pass (`pre-commit run --all-files`)
- [ ] No critical bugs or security issues
- [ ] Code coverage is acceptable
- [ ] Performance benchmarks reviewed (if applicable)

### Documentation
- [ ] README.md is up to date
- [ ] CHANGELOG.md updated with release notes
- [ ] API documentation updated (if applicable)
- [ ] Migration guide created (if breaking changes)
- [ ] Examples updated (if needed)

### Version Management
- [ ] Version updated in `pyproject.toml`
- [ ] Version follows semantic versioning
- [ ] CHANGELOG.md includes version section
- [ ] All version references updated

### Dependencies
- [ ] Dependencies reviewed and updated
- [ ] Security vulnerabilities addressed
- [ ] Compatibility tested with supported Python versions
- [ ] Optional dependencies verified

### Testing
- [ ] Unit tests pass
- [ ] Integration tests pass (if applicable)
- [ ] Manual testing completed
- [ ] Edge cases tested
- [ ] Backward compatibility verified (if applicable)

### Communication
- [ ] Release notes drafted
- [ ] Breaking changes documented
- [ ] Deprecation warnings added (if applicable)
- [ ] Contributors acknowledged

## Release Process

### 1. Prepare Release
```bash
# Run the prepare release script
./scripts/prepare_release.sh [VERSION]

# Or manually:
# 1. Update version in pyproject.toml
# 2. Update CHANGELOG.md
# 3. Commit changes
git add pyproject.toml CHANGELOG.md
git commit -m "Prepare release v[VERSION]"
git push origin main
```

### 2. Create and Push Tag
```bash
git tag -a v[VERSION] -m "Release version [VERSION]"
git push origin v[VERSION]
```

### 3. Monitor Release Workflow
- [ ] GitHub Actions workflow started
- [ ] Build job completed successfully
- [ ] PyPI publish job completed successfully
- [ ] GitHub release created successfully

### 4. Verify Release
- [ ] Package available on PyPI: https://pypi.org/project/ragworkbench/
- [ ] GitHub release created: https://github.com/IBM/RagWorkbench/releases
- [ ] Installation works: `pip install ragworkbench==[VERSION]`
- [ ] Basic functionality tested after installation

## Post-Release Tasks

- [ ] Announce release (if applicable)
- [ ] Update documentation site (if applicable)
- [ ] Close related issues
- [ ] Update project board (if applicable)
- [ ] Monitor for issues in the first 24-48 hours

## Rollback Plan

If issues are discovered:

1. **Yank the PyPI release** (if critical bug)
   - Go to https://pypi.org/project/ragworkbench/
   - Click "Manage" → "Releases" → "Yank"

2. **Delete the tag** (if needed)
   ```bash
   git tag -d v[VERSION]
   git push origin :refs/tags/v[VERSION]
   ```

3. **Create hotfix release** with incremented version

## Notes

<!-- Add any additional notes or context about this release -->

## Related Issues

<!-- Link to related issues or PRs -->

Closes #
