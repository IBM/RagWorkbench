#!/bin/bash
# Script to prepare a new release for RagWorkbench
# Usage: ./scripts/prepare_release.sh <version>
# Example: ./scripts/prepare_release.sh 0.2.0

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if version argument is provided
if [ -z "$1" ]; then
    echo -e "${RED}Error: Version number required${NC}"
    echo "Usage: $0 <version>"
    echo ""
    echo "Version format examples:"
    echo "  Stable releases:    0.2.0, 1.0.0, 2.1.3"
    echo "  Beta releases:      0.1.0-beta.1, 1.0.0-beta.2"
    echo "  Alpha releases:     0.1.0-alpha.1, 1.0.0-alpha.3"
    echo "  Release candidates: 1.0.0-rc.1, 2.0.0-rc.2"
    echo ""
    echo "Examples:"
    echo "  $0 0.2.0"
    echo "  $0 1.0.0-beta.1"
    echo "  $0 2.0.0-rc.1"
    exit 1
fi

NEW_VERSION=$1

# Validate version format (basic semver check)
if ! [[ $NEW_VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9.]+)?$ ]]; then
    echo -e "${RED}Error: Invalid version format${NC}"
    echo "Version must follow semantic versioning (e.g., 0.2.0, 1.0.0-beta.1)"
    exit 1
fi

echo -e "${GREEN}Preparing release v${NEW_VERSION}${NC}"
echo ""

# Check if on main branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "main" ]; then
    echo -e "${YELLOW}Warning: Not on main branch (current: ${CURRENT_BRANCH})${NC}"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo -e "${RED}Error: Uncommitted changes detected${NC}"
    echo "Please commit or stash your changes before preparing a release"
    exit 1
fi

# Pull latest changes
echo -e "${YELLOW}Pulling latest changes...${NC}"
git pull origin main

# Get current version from pyproject.toml
CURRENT_VERSION=$(grep -m 1 'version = ' pyproject.toml | cut -d'"' -f2)
echo "Current version: ${CURRENT_VERSION}"
echo "New version: ${NEW_VERSION}"
echo ""

# Update version in pyproject.toml
echo -e "${YELLOW}Updating version in pyproject.toml...${NC}"
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    sed -i '' "s/version = \"${CURRENT_VERSION}\"/version = \"${NEW_VERSION}\"/" pyproject.toml
else
    # Linux
    sed -i "s/version = \"${CURRENT_VERSION}\"/version = \"${NEW_VERSION}\"/" pyproject.toml
fi

# Check if CHANGELOG.md exists
if [ ! -f "CHANGELOG.md" ]; then
    echo -e "${RED}Error: CHANGELOG.md not found${NC}"
    exit 1
fi

# Check if version already exists in CHANGELOG
if grep -q "\[${NEW_VERSION}\]" CHANGELOG.md; then
    echo -e "${YELLOW}Warning: Version ${NEW_VERSION} already exists in CHANGELOG.md${NC}"
else
    # Get current date
    RELEASE_DATE=$(date +%Y-%m-%d)

    # Create temporary file with new version section
    echo -e "${YELLOW}Adding version ${NEW_VERSION} to CHANGELOG.md...${NC}"

    # Insert new version section after [Unreleased]
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' "/## \[Unreleased\]/a\\
\\
## [${NEW_VERSION}] - ${RELEASE_DATE}\\
\\
### Added\\
- \\
\\
### Changed\\
- \\
\\
### Fixed\\
- \\
" CHANGELOG.md
    else
        # Linux
        sed -i "/## \[Unreleased\]/a\\\\n## [${NEW_VERSION}] - ${RELEASE_DATE}\\n\\n### Added\\n- \\n\\n### Changed\\n- \\n\\n### Fixed\\n- \\n" CHANGELOG.md
    fi

    echo -e "${GREEN}✓ CHANGELOG.md updated${NC}"
    echo -e "${YELLOW}Please edit CHANGELOG.md to add release notes for version ${NEW_VERSION}${NC}"

    # Auto-fix any formatting issues introduced by sed
    if command -v uv &> /dev/null; then
        echo -e "${YELLOW}Auto-fixing CHANGELOG.md formatting...${NC}"
        uv run pre-commit run --files CHANGELOG.md || true
        echo -e "${GREEN}✓ CHANGELOG.md formatting fixed${NC}"
    fi
fi

# Run tests
echo ""
echo -e "${YELLOW}Running tests...${NC}"
if command -v pytest &> /dev/null; then
    pytest || {
        echo -e "${RED}Tests failed! Please fix before continuing.${NC}"
        exit 1
    }
    echo -e "${GREEN}✓ Tests passed${NC}"
else
    echo -e "${YELLOW}Warning: pytest not found, skipping tests${NC}"
fi

# Run pre-commit checks
echo ""
echo -e "${YELLOW}Running pre-commit checks...${NC}"
if command -v uv &> /dev/null; then
    uv run pre-commit run --all-files || {
        echo -e "${RED}Pre-commit checks failed! Please fix before continuing.${NC}"
        exit 1
    }
    echo -e "${GREEN}✓ Pre-commit checks passed${NC}"
else
    echo -e "${YELLOW}Warning: uv not found, skipping pre-commit checks${NC}"
fi

# Show git diff
echo ""
echo -e "${YELLOW}Changes to be committed:${NC}"
git diff pyproject.toml CHANGELOG.md

# Commit changes
echo ""
read -p "Commit these changes? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    git add pyproject.toml CHANGELOG.md
    git commit -m "Prepare release v${NEW_VERSION}"
    echo -e "${GREEN}✓ Changes committed${NC}"

    # Ask to push
    echo ""
    read -p "Push to origin? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git push origin main
        echo -e "${GREEN}✓ Changes pushed${NC}"
    fi
else
    echo -e "${YELLOW}Changes not committed. You can commit manually with:${NC}"
    echo "  git add pyproject.toml CHANGELOG.md"
    echo "  git commit -m 'Prepare release v${NEW_VERSION}'"
    exit 0
fi

# Instructions for creating tag
echo ""
echo -e "${GREEN}Release preparation complete!${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Review and edit CHANGELOG.md if needed"
echo "2. Create and push the release tag:"
echo -e "   ${GREEN}git tag -a v${NEW_VERSION} -m 'Release version ${NEW_VERSION}'${NC}"
echo -e "   ${GREEN}git push origin v${NEW_VERSION}${NC}"
echo ""
echo "3. The GitHub Actions workflow will automatically:"
echo "   - Build the package"
echo "   - Publish to PyPI"
echo "   - Create a GitHub release"
echo ""
echo "4. Monitor the release workflow at:"
echo "   https://github.com/IBM/RagWorkbench/actions"

# Made with Bob
