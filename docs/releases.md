# CI and releases

Three GitHub Actions workflows in `.github/workflows`:

| Workflow | Runs on | Does |
| --- | --- | --- |
| `ci.yml` | every push to main, every pull request, on demand | Tests, lint, web UI, installers |
| `release.yml` | a `vX.Y.Z` tag, every night, on demand | Builds the installers and publishes them as a release |
| `build.yml` | called by the other two | Builds the installer on Windows, macOS and Linux |

## CI

| Job | Checks |
| --- | --- |
| Tests | `pytest` on Windows, macOS and Linux with Python 3.14, and on Linux with Python 3.11 (the oldest `pyproject.toml` allows). PyTorch comes from its CPU index; tests that need real model weights skip |
| Lint | Ruff, errors only: syntax errors and undefined names |
| Web UI | `npm ci`, then `npm run build` (the TypeScript check and the Vite build) |
| Installers | `packaging/build.py` and the launcher's tests on each system, as in a release. Each installer is kept as a run artifact (`installer-windows`, `installer-macos`, `installer-linux`) |

The installers are built where they run: Windows Server 2025, macOS 15 on Apple silicon
(targeting macOS 12), and Ubuntu 22.04, so the Linux binary runs on glibc 2.35 and newer.

## Publishing a release

1. Set the version in `pyproject.toml` (`[project] version`).
2. Tag that commit and push the tag:

   ```bash
   git tag v0.2.0
   git push origin v0.2.0
   ```

The release workflow checks that the tag matches `pyproject.toml`, builds the three installers,
and publishes the release "Urakatsu 0.2.0" with them and notes generated from the commits.

## Nightly builds

Every night at 04:00 UTC, main is built and published as the `nightly` pre-release, replacing
the previous one, unless main has not moved since. Running the release workflow by hand on main
does the same at once.
