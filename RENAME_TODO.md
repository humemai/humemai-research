# Rename TODO: humemai → humemai-research

This document tracks all changes needed to rename the package from `humemai` to `humemai-research`.

## Background

- **Old package name**: `humemai` (now reserved for the company SDK)
- **New package name**: `humemai-research` (for research code)
- **Old module name**: `humemai/`
- **New module name**: `humemai_research/`
- **Repository**: Already renamed to `humemai/humemai-research` on GitHub

---

## Tasks

### 1. Core Package Structure

- [ ] Rename module folder: `humemai/` → `humemai_research/`
- [ ] Update `setup.cfg`:
  - [ ] Change `name = humemai` → `name = humemai-research`
  - [ ] Update `url = https://github.com/humemai/humemai` → `https://github.com/humemai/humemai-research`
  - [ ] Update `Bug Tracker = https://github.com/humemai/humemai/issues` → `https://github.com/humemai/humemai-research/issues`

### 2. Code Updates

- [ ] Update all imports in source code:
  - [ ] Search and replace `from humemai` → `from humemai_research`
  - [ ] Search and replace `import humemai` → `import humemai_research`
- [ ] Update all imports in test files (`test/` directory)
- [ ] Update all imports in example files (`examples/` directory)
- [ ] Update any Jupyter notebooks (`*.ipynb`) that import the package

### 3. GitHub Workflows

- [ ] **`.github/workflows/docs.yaml`**:
  - [ ] Update clone URL: `humemai/humemai.git` → `humemai/humemai-research.git`
  - [ ] Update folder references: `cd humemai` → `cd humemai-research`
  - [ ] Update install command context
  - [ ] Update pdoc command: `pdoc ... humemai` → `pdoc ... humemai_research`

- [ ] **`.github/workflows/pytest.yaml`**:
  - [ ] Update clone URL: `humemai/humemai.git` → `humemai/humemai-research.git`
  - [ ] Update folder references: `cd humemai` → `cd humemai-research`
  - [ ] Update install command context

- [ ] **`.github/workflows/publish-to-pypi.yaml`**:
  - [ ] Add `environment: pypi` to the deploy job
  - [ ] Consider renaming file to `publish-pypi.yml` for consistency
  - [ ] Verify PyPI token secret is properly configured

### 4. Documentation

- [ ] Update `README.md`:
  - [ ] Installation instructions: `pip install humemai` → `pip install humemai-research`
  - [ ] Import examples: `from humemai` → `from humemai_research`
  - [ ] Any repository URLs
- [ ] Regenerate `_docs/` folder with new module name
- [ ] Update any other markdown files with references to old package name

### 5. PyPI Release Strategy

- [ ] Identify all existing GitHub releases and their version numbers
- [ ] For each release version:
  - [ ] Checkout the release tag
  - [ ] Apply the rename changes
  - [ ] Build the package: `python setup.py sdist bdist_wheel`
  - [ ] Upload to PyPI: `twine upload dist/*`
- [ ] Document version history in PyPI for the new package

### 6. Other Files

- [ ] Check `LICENSE` for any package name references
- [ ] Check any configuration files (`.flake8`, `.gitignore`, etc.)
- [ ] Update `Makefile` if it has any package-specific references
- [ ] Check for any hardcoded references in:
  - [ ] Docker files (if any)
  - [ ] Requirements files
  - [ ] CI/CD scripts

### 7. Testing & Validation

- [ ] Run local tests after changes: `make test`
- [ ] Verify package builds correctly: `python setup.py sdist bdist_wheel`
- [ ] Test install from local build: `pip install dist/humemai_research-*.whl`
- [ ] Verify imports work: `python -c "from humemai_research import ..."`
- [ ] Run GitHub Actions workflows manually to verify they pass
- [ ] Verify documentation builds and deploys correctly

### 8. Final Steps

- [ ] Commit all changes with clear message
- [ ] Create a new release/tag with updated package
- [ ] Verify PyPI upload works via GitHub Actions
- [ ] Update any external documentation pointing to the old package
- [ ] Announce the package rename to users (if applicable)

---

## Notes

- The `humemai` package name is now reserved for the company SDK
- All research code should use `humemai-research` going forward
- Module imports will use `humemai_research` (with underscore, not hyphen)
- PyPI package name is `humemai-research` (with hyphen)

---

## Commands for Reference

```bash
# Search for all imports to replace
grep -r "from humemai" --include="*.py" .
grep -r "import humemai" --include="*.py" .

# Build package
python setup.py sdist bdist_wheel

# Upload to PyPI
twine upload dist/*

# Test installation
pip install dist/humemai_research-*.whl
```
