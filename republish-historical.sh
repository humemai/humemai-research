#!/bin/bash
set -e

# This script republishes all historical versions with their actual original code
# Each version will be checked out from its original commit, renamed, and published

# Get the repository root directory
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Array of versions and their original commits
declare -A versions=(
    ["1.0.0"]="4f51ab3"
    ["1.0.1"]="44ad918"
    ["1.0.2"]="537c6d1"
    ["1.0.3"]="9847692"
    ["1.0.4"]="0a5b358"
    ["1.1.0"]="39c0e60"
    ["1.1.1"]="bbc47d5"
    ["1.1.2"]="416d8f5"
    ["2.0.0"]="6b38fe4"
    ["2.0.2"]="5fdc3c1"
    ["2.1.0"]="c6019b3"
    ["2.2.0"]="4f8b39a"
    ["2.3.0"]="ce6aa57"
    ["2.4.0"]="679f8f9"
    ["2.4.1"]="813eb36"
    ["2.5.0"]="5c24c66"
    ["2.5.1"]="5de9feb"
    ["2.5.2"]="d338c5a"
    ["2.5.3"]="a07ffe6"
    ["2.5.6"]="9da74d0"
)

# Order matters - process in version order
ordered_versions=("1.0.0" "1.0.1" "1.0.2" "1.0.3" "1.0.4" "1.1.0" "1.1.1" "1.1.2" "2.0.0" "2.0.2" "2.1.0" "2.2.0" "2.3.0" "2.4.0" "2.4.1" "2.5.0" "2.5.1" "2.5.2" "2.5.3" "2.5.6")

# Save the current workflow file
echo "Saving current workflow file..."
cp .github/workflows/publish-pypi.yml /tmp/publish-pypi.yml.backup

for version in "${ordered_versions[@]}"; do
    commit="${versions[$version]}"
    post_version="${version}.post0"
    echo ""
    echo "========================================"
    echo "Processing v$post_version from commit $commit"
    echo "========================================"
    
    # Checkout the original commit
    git checkout $commit
    
        # Check if 'humemai' folder exists (old structure) and rename it properly
    if [ -d "humemai" ] && [ ! -d "humemai_research" ]; then
        echo "Renaming humemai/ to humemai_research/..."
        mv humemai humemai_research
        git add humemai_research
        git rm -r --cached humemai 2>/dev/null || true
    elif [ -d "humemai_research/humemai" ]; then
        echo "Fixing nested structure: moving humemai_research/humemai/ contents up..."
        # Move contents up one level
        mv humemai_research/humemai/* humemai_research/ 2>/dev/null || true
        mv humemai_research/humemai/.* humemai_research/ 2>/dev/null || true  
        rm -rf humemai_research/humemai
    fi
    
    # Update setup.cfg - change package name and URLs
    echo "Updating setup.cfg..."
    sed -i 's/^name = humemai$/name = humemai-research/' setup.cfg
    sed -i "s/^version = .*$/version = $post_version/" setup.cfg
    sed -i 's|github.com/humemai/humemai|github.com/humemai/humemai-research|g' setup.cfg
    
    # Remove old workflow file if it exists and add new one
    echo "Adding publish-pypi.yml workflow..."
    mkdir -p .github/workflows
    rm -f .github/workflows/publish-to-pypi.yaml
    cp /tmp/publish-pypi.yml.backup .github/workflows/publish-pypi.yml
    
    # Remove cassandra data files
    echo "Removing cassandra data files..."
    rm -rf cassandra_data/ examples/*/cassandra_data/ 2>/dev/null || true
    
    # Update all Python imports if humemai folder existed
    if [ -d "humemai_research" ]; then
        echo "Updating imports in Python files..."
        find . -path ./cassandra_data -prune -o -name "*.py" -type f -exec sed -i 's/from humemai/from humemai_research/g' {} \; 2>/dev/null || true
        find . -path ./cassandra_data -prune -o -name "*.py" -type f -exec sed -i 's/import humemai/import humemai_research/g' {} \; 2>/dev/null || true
    fi
    
    # Update README badge URLs
    echo "Updating README badges..."
    if [ -f "README.md" ]; then
        sed -i 's|badge.fury.io/py/humemai|badge.fury.io/py/humemai-research|g' README.md
    fi
    
    # Create a commit for this version
    git add -A
    git commit -m "Republish v$post_version with historical code to humemai-research on PyPI" || true
    
    # Tag it
    git tag -f v$post_version
    
    # Push the tag (which will trigger the workflow)
    git push origin refs/tags/v$post_version -f
    
    # Delete old release and create new one
    gh release delete v$post_version --yes 2>/dev/null || echo "No existing release to delete"
    gh release create v$post_version --title "v$post_version" --notes "Release v$post_version with original historical code, republished to \`humemai-research\` on PyPI."
    
    echo "✓ v$post_version processed and released"
    
    # Wait a bit between releases to avoid overwhelming the API
    sleep 15
done

# Return to main branch
echo ""
echo "========================================"
echo "All versions processed! Returning to main branch..."
git checkout main

echo ""
echo "Done! All 20 versions have been republished with their historical code."
echo "Check GitHub Actions: https://github.com/humemai/humemai-research/actions"
echo "Check PyPI: https://pypi.org/project/humemai-research/#history"
