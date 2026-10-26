// The release PR, after Koharu's (.github/release-please/index.cjs in koharu-rs/koharu).
//
// On every push to main, release-please keeps one open PR "chore(main): release X.Y.Z" whose
// body and CHANGELOG.md entry are git-cliff's notes for the commits since the last release.
// Merging it tags vX.Y.Z and creates the GitHub release; release-pr.yml then starts
// release.yml on that tag, which builds the installers and attaches them.

const assert = require('node:assert/strict')
const { execFileSync } = require('node:child_process')
const { readFileSync } = require('node:fs')

const { runGitCliff } = require('git-cliff')
const { GitHub, Manifest, registerReleaseType } = require('release-please')
// BaseStrategy is internal: keep the release-please dependency pinned and test this adapter on upgrades.
const { BaseStrategy } = require('release-please/build/src/strategies/base')

// Every file packaging/set_version.py writes.
const VERSION_FILES = [
  'packaging/launcher/Cargo.lock',
  'packaging/launcher/Cargo.toml',
  'pyproject.toml',
  'urakatsu/__init__.py',
  'uv.lock',
  'web/package-lock.json',
  'web/package.json',
]

function command(program, args) {
  return execFileSync(program, args, { encoding: 'utf8', maxBuffer: 32 * 1024 * 1024 }).trim()
}

class Urakatsu extends BaseStrategy {
  async buildReleaseNotes(_commits, _version, tag, latestRelease) {
    assert(latestRelease?.sha, 'The previous release tag must be available')
    const { value: head } = await this.github
      .mergeCommitIterator(this.targetBranch, { maxResults: 1 })
      .next()
    assert.equal(
      command('git', ['rev-parse', 'HEAD']),
      head?.sha,
      'main changed; rerun with a fresh checkout',
    )
    assert.equal(
      command('git', ['status', '--porcelain']),
      '',
      'Release preparation requires a clean checkout',
    )

    // The virtual tag renders the next release without creating a tag before review.
    const { stdout } = await runGitCliff(
      [
        '--unreleased',
        '--tag',
        tag.toString(),
        '--strip',
        'all',
        `${latestRelease.sha}..HEAD`,
      ],
      { stdio: 'pipe' },
    )
    return stdout.trim()
  }

  async buildUpdates({ newVersion, changelogEntry }) {
    command('python3', ['packaging/set_version.py', newVersion.toString()])
    const paths = command('git', ['diff', '--name-only']).split('\n').filter(Boolean).sort()
    assert.deepEqual(paths, VERSION_FILES, 'set_version.py must update exactly the version files')
    const updates = paths.map((path) => {
      const content = readFileSync(path, 'utf8')
      return { path, createIfMissing: false, updater: { updateContent: () => content } }
    })
    const changelog = `${changelogEntry}\n\n\n${readFileSync('CHANGELOG.md', 'utf8')}`
    updates.push({
      path: 'CHANGELOG.md',
      createIfMissing: false,
      updater: { updateContent: () => changelog },
    })
    return updates
  }
}

// A custom release type keeps the version files' edits in one Python script.
registerReleaseType('urakatsu', (options) => new Urakatsu(options))

module.exports = async function releasePlease({ owner, repo, token }) {
  const github = await GitHub.create({ owner, repo, token })
  const manifest = await Manifest.fromManifest(
    github,
    'main',
    '.github/release-please/config.json',
    '.github/release-please/manifest.json',
  )
  const releases = (await manifest.createReleases()).filter(Boolean)
  assert(releases.length <= 1, 'Urakatsu has a single release')
  if (releases.length) return releases[0].tagName
  await manifest.createPullRequests()
  return ''
}
