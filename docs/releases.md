# CI and releases

Four GitHub Actions workflows in `.github/workflows`:

| Workflow | Runs on | Does |
| --- | --- | --- |
| `ci.yml` | every push to main, every pull request, on demand | Tests, lint, web UI, installers |
| `release-pr.yml` | every push to main, on demand | Keeps the release PR open; merging it tags the release |
| `release.yml` | a `vX.Y.Z` tag, every night, on demand | Builds the installers and publishes them as a release |
| `build.yml` | called by `ci.yml` and `release.yml` | Builds the installer on Windows, macOS and Linux |

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

Releases work as in Koharu: [release-please](https://github.com/googleapis/release-please) keeps a
release PR open and [git-cliff](https://git-cliff.org) writes the changelog.

1. Every push to main updates the PR **chore(main): release X.Y.Z**. The version comes from the
   commits since the last release ([conventional commits](https://www.conventionalcommits.org)):
   `feat` bumps the minor version, `fix` and the rest the patch, and before 1.0 a breaking change
   (`feat!:` or a `BREAKING CHANGE:` footer) bumps the minor version too. The PR sets that version
   in every file that has one (`packaging/set_version.py`) and adds its entry to the top of
   `CHANGELOG.md`.
2. Merging the PR tags `vX.Y.Z` and creates the GitHub release with the same entry as its notes.
3. The release workflow then runs on the tag: it checks the tag against `pyproject.toml`, builds
   the three installers, attaches them, and names the release "Urakatsu X.Y.Z".

To change what a release says, edit the commit messages before it, or edit the release PR's
`CHANGELOG.md` entry and its description before merging (the description becomes the notes).

The repository must allow it: Settings > Actions > General > Workflow permissions >
"Allow GitHub Actions to create and approve pull requests".

A `vX.Y.Z` tag pushed by hand still works; the release workflow then writes the notes itself with
`git-cliff --current`.

### The changelog

`cliff.toml` (after Koharu's) groups the commits of each release by type: Features (`feat`), Bug
Fixes (`fix`), Refactor, Documentation, Performance, Styling, Testing, Miscellaneous Tasks
(`chore`, `ci`), Security, Revert, and Other for commits that are not conventional. A scope shows
in italics, `fix(catalog): ...` becomes "*(catalog)* ...". Release commits (`chore(main): release
...`) and dependency bumps (`chore(deps): ...`) are left out.

To preview the next release's notes:

```bash
npx git-cliff@2.14.2 --unreleased
```

release-please and git-cliff are pinned in `.github/release-please/package.json`. The adapter in
`.github/release-please/index.cjs` uses release-please internals, so test it on upgrades.

## Nightly builds

Every night at 04:00 UTC, main is built and published as the `nightly` pre-release, replacing
the previous one, unless main has not moved since. Running the release workflow by hand on main
does the same at once. Its notes list the changes since the last release.
