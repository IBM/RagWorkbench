# Release Process

This document describes the release process for RAGWorkbench.

## Overview

RAGWorkbench uses an automated release process powered by GitHub Actions. Releases are triggered by pushing version tags and automatically publish to PyPI and create GitHub releases.

## Release Types

### Stable Releases
- **Format**: `v1.0.0`, `v1.2.3`, etc.
- **Published to**: PyPI and GitHub Releases
- **Triggered by**: Tags matching `v*.*.*` (semantic versioning)

### Pre-releases
- **Format**: `v1.0.0-alpha.1`, `v1.0.0-beta.2`, `v1.0.0-rc.1`
- **Published to**: TestPyPI and GitHub Releases (marked as pre-release)
- **Triggered by**: Tags containing `alpha`, `beta`, or `rc`

## Prerequisites

Before creating a release, ensure:

1. **All tests pass**: Run `pytest` locally and ensure CI passes
2. **Code quality checks pass**: Run `pre-commit run --all-files`
3. **CHANGELOG.md is updated**: Add release notes for the new version
4. **Version is updated**: Update version in `pyproject.toml`
5. **Documentation is current**: Update README.md if needed

## PyPI Configuration

### Option 1: Trusted Publishing (Recommended)

Configure trusted publishing on PyPI to avoid managing API tokens:

1. Go to https://pypi.org/manage/account/publishing/
2. Add a new publisher with:
   - **PyPI Project Name**: `ragworkbench`
   - **Owner**: `IBM`
   - **Repository name**: `RagWorkbench`
   - **Workflow name**: `release.yaml`
   - **Environment name**: `pypi`

### Option 2: API Token

If not using trusted publishing:

1. Generate a PyPI API token at https://pypi.org/manage/account/token/
2. Add it as a repository secret named `PYPI_API_TOKEN`
3. Uncomment the `password` line in `.github/workflows/release.yaml`

For TestPyPI (pre-releases):
1. Generate a TestPyPI token at https://test.pypi.org/manage/account/token/
2. Add it as a repository secret named `TESTPYPI_API_TOKEN`

## Creating a Release

### Step 1: Update Version and Changelog

1. **Update version in pyproject.toml**:
   ```toml
   [project]
   name = "ragworkbench"
   version = "0.2.0"  # Update this
   ```

2. **Update CHANGELOG.md**:
   ```markdown
   ## [0.2.0] - 2024-XX-XX
   
   ### Added
   - New feature description
   
   ### Changed
   - Changed feature description
   
   ### Fixed
   - Bug fix description
   ```

3. **Commit changes**:
   ```bash
   git add pyproject.toml CHANGELOG.md
   git commit -m "Prepare release v0.2.0"
   git push origin main
   ```

### Step 2: Create and Push Tag

1. **Create an annotated tag**:
   ```bash
   git tag -a v0.2.0 -m "Release version 0.2.0"
   ```

2. **Push the tag**:
   ```bash
   git push origin v0.2.0
   ```

### Step 3: Monitor Release Process

1. Go to the [Actions tab](https://github.com/IBM/RagWorkbench/actions)
2. Watch the "Release" workflow execution
3. Verify all jobs complete successfully:
   - ✅ Build distribution
   - ✅ Publish to PyPI
   - ✅ Create GitHub Release

### Step 4: Verify Release

1. **Check PyPI**: https://pypi.org/project/ragworkbench/
2. **Check GitHub Release**: https://github.com/IBM/RagWorkbench/releases
3. **Test installation**:
   ```bash
   pip install ragworkbench==0.2.0
   ```

## Pre-release Process

For alpha, beta, or release candidate versions:

1. **Update version with pre-release suffix**:
   ```toml
   version = "0.2.0-beta.1"
   ```

2. **Create and push tag**:
   ```bash
   git tag -a v0.2.0-beta.1 -m "Beta release 0.2.0-beta.1"
   git push origin v0.2.0-beta.1
   ```

3. **Verify on TestPyPI**: https://test.pypi.org/project/ragworkbench/

## Hotfix Release Process

For urgent bug fixes:

1. **Create hotfix branch from tag**:
   ```bash
   git checkout -b hotfix/v0.1.1 v0.1.0
   ```

2. **Make fixes and commit**:
   ```bash
   git add .
   git commit -m "Fix critical bug"
   ```

3. **Update version to patch release**:
   ```toml
   version = "0.1.1"
   ```

4. **Update CHANGELOG.md**:
   ```markdown
   ## [0.1.1] - 2024-XX-XX
   
   ### Fixed
   - Critical bug description
   ```

5. **Merge to main and tag**:
   ```bash
   git checkout main
   git merge hotfix/v0.1.1
   git tag -a v0.1.1 -m "Hotfix release 0.1.1"
   git push origin main v0.1.1
   ```

## Rollback Process

If a release has issues:

1. **Delete the tag locally and remotely**:
   ```bash
   git tag -d v0.2.0
   git push origin :refs/tags/v0.2.0
   ```

2. **Delete the GitHub release** (if created)

3. **Yank the PyPI release** (if published):
   - Go to https://pypi.org/project/ragworkbench/
   - Click "Manage" → "Releases" → "Yank"

4. **Fix issues and create new release** with incremented version

## Versioning Guidelines

Follow [Semantic Versioning](https://semver.org/):

- **MAJOR** (1.0.0): Incompatible API changes
- **MINOR** (0.1.0): New functionality, backwards compatible
- **PATCH** (0.0.1): Backwards compatible bug fixes

### Pre-release Identifiers

- **alpha**: Early testing, unstable
- **beta**: Feature complete, testing phase
- **rc** (release candidate): Final testing before stable release

Examples:
- `v1.0.0-alpha.1` → `v1.0.0-alpha.2` → `v1.0.0-beta.1` → `v1.0.0-rc.1` → `v1.0.0`

## Troubleshooting

### Build Fails

- Check that `pyproject.toml` is valid
- Ensure all dependencies are properly specified
- Verify Python version compatibility

### PyPI Upload Fails

- Check PyPI credentials/trusted publishing configuration
- Verify package name is available on PyPI
- Ensure version doesn't already exist (can't overwrite)

### GitHub Release Fails

- Check repository permissions
- Verify `GITHUB_TOKEN` has write access
- Ensure tag exists and is pushed

## Best Practices

1. **Test before releasing**: Always run full test suite
2. **Update documentation**: Keep README and docs in sync
3. **Write clear release notes**: Help users understand changes
4. **Use semantic versioning**: Make version numbers meaningful
5. **Tag consistently**: Use annotated tags with messages
6. **Communicate**: Announce releases to users/contributors
7. **Monitor**: Watch for issues after release

## Release Checklist

Before creating a release, verify:

- [ ] All tests pass (`pytest`)
- [ ] Code quality checks pass (`pre-commit run --all-files`)
- [ ] Version updated in `pyproject.toml`
- [ ] CHANGELOG.md updated with release notes
- [ ] Documentation updated (if needed)
- [ ] Breaking changes documented (if any)
- [ ] Migration guide provided (if needed)
- [ ] Dependencies reviewed and updated
- [ ] Security vulnerabilities addressed
- [ ] Performance tested (if applicable)

## Support

For questions or issues with the release process:
- Open an issue: https://github.com/IBM/RagWorkbench/issues
- Contact maintainers: See AUTHORS in README.md