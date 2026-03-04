# Release Notes – v0.7.20

Date: 2026-03-04
Branch: main
Tag: v0.7.20

## Overview

`v0.7.20` finalizes public snapshot visibility and release transparency.

## Changes

- README update for public snapshot positioning:
  - clarified project structure
  - updated status to include **Phase H complete**
  - switched test instructions from Makefile-based commands to direct `pytest`
- Public publish whitelist expanded:
  - includes representative tests for external readers
  - includes release process scripts (`scripts/publish_release.sh`, `scripts/publish_whitelist.txt`)
- Release notes advanced to `v0.7.20` and linked from README.

## Notes

- Publishing remains snapshot-based from private tags (no private git history transfer).
- Public demo notebook target remains `notebooks/team_demo.ipynb`.
