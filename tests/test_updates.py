"""Update checks against GitHub releases (faked), channel switching and installing."""

from __future__ import annotations

import hashlib

import httpx
import pytest

from urakatsu import updates

INSTALLER = b"MZ fake installer"
NIGHTLY_COMMIT = "a" * 40


def release(channel: str, *, tag: str = "v0.2.0", commit: str = NIGHTLY_COMMIT, body: str = "", digest: bool = True) -> dict:
    name = "Urakatsu-Setup-0.2.0.exe"
    return {
        "tag_name": tag if channel == "stable" else "nightly",
        "target_commitish": commit if channel == "nightly" else "main",
        "published_at": "2026-10-24T04:00:00Z",
        "body": body,
        "html_url": f"https://github.com/karrot0/Urakatsu/releases/tag/{tag if channel == 'stable' else 'nightly'}",
        "assets": [
            {"name": "Urakatsu-0.2.0-linux-x86_64.tar.gz", "browser_download_url": "https://example.test/linux", "size": 1},
            {
                "name": name,
                "browser_download_url": "https://example.test/setup.exe",
                "size": len(INSTALLER),
                "digest": f"sha256:{hashlib.sha256(INSTALLER).hexdigest()}" if digest else None,
            },
        ],
    }


def github(stable: dict | None, nightly: dict | None) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/releases/latest"):
            return httpx.Response(200, json=stable) if stable else httpx.Response(404, json={})
        if request.url.path.endswith("/releases/tags/nightly"):
            return httpx.Response(200, json=nightly) if nightly else httpx.Response(404, json={})
        if request.url.host == "example.test":
            return httpx.Response(200, content=INSTALLER)
        return httpx.Response(404)

    return httpx.MockTransport(handle)


@pytest.fixture()
def fake(monkeypatch, tmp_path):
    """Windows, a stable 0.1.0 build, GitHub faked; returns a setter for the faked releases."""
    monkeypatch.setenv("URAKATSU_HOME", str(tmp_path))
    monkeypatch.setattr(updates.platform, "system", lambda: "Windows")
    build = updates.BuildInfo(version="0.1.0", channel="stable", commit="b" * 40, built="2026-10-20T00:00:00Z")
    monkeypatch.setattr(updates, "build_info", lambda: build)
    feeds = {"transport": github(release("stable"), release("nightly"))}
    real_client = httpx.Client
    monkeypatch.setattr(updates.httpx, "Client", lambda *a, **k: real_client(transport=feeds["transport"], **{key: v for key, v in k.items() if key != "transport"}))

    def set_feeds(stable, nightly):
        feeds["transport"] = github(stable, nightly)

    return set_feeds


def test_parses_both_channels_and_picks_this_systems_installer(fake):
    stable = updates.fetch_release("stable")
    assert (stable.id, stable.version, stable.channel) == ("v0.2.0", "0.2.0", "stable")
    assert stable.asset.name == "Urakatsu-Setup-0.2.0.exe" and stable.asset.sha256
    nightly = updates.fetch_release("nightly")
    assert nightly.commit == NIGHTLY_COMMIT and nightly.version == NIGHTLY_COMMIT[:7]


def test_nightly_commit_falls_back_to_the_notes(fake):
    fake(None, release("nightly", commit="main", body="Built from 1234abc on 2026-10-24."))
    assert updates.fetch_release("nightly").commit == "1234abc"
    assert updates.fetch_release("stable") is None  # no stable release yet


def test_is_newer_rules():
    stable_build = updates.BuildInfo(version="0.1.0", channel="stable", commit="b" * 40)
    newer = updates.Release(id="v0.2.0", channel="stable", version="0.2.0")
    same = updates.Release(id="v0.1.0", channel="stable", version="0.1.0")
    assert updates.is_newer(newer, stable_build) and not updates.is_newer(same, stable_build)
    assert updates._version_key("0.10.0") > updates._version_key("0.9.9") > updates._version_key("0.9.9rc1")
    nightly_build = updates.BuildInfo(version="0.1.0", channel="nightly", commit="c" * 40)
    assert updates.is_newer(updates.Release(id="n", channel="nightly", version="ddd", commit="d" * 40), nightly_build)
    assert not updates.is_newer(updates.Release(id="n", channel="nightly", version="ccc", commit="c" * 40), nightly_build)
    # Choosing the other channel offers its build, even an older version: that is the switch.
    assert updates.is_newer(same, nightly_build)


def test_status_switching_and_later(fake):
    updater = updates.Updater()
    assert updater.channel() == "stable"  # the running build's channel
    status = updater.check()
    assert status.available and not status.switching and not status.dismissed
    status = updater.dismiss()
    assert status.dismissed  # "Later" for v0.2.0
    status = updater.set_channel("nightly")
    assert status.channel == "nightly" and status.available and status.switching and not status.dismissed
    assert updates.Updater().channel() == "nightly"  # remembered in updates.json
    with pytest.raises(ValueError):
        updater.set_channel("beta")


def test_install_checks_the_digest_then_hands_off_to_the_desktop_app(fake, tmp_path):
    updater = updates.Updater()
    updater.check()
    assert not updater.status().can_install  # only the desktop app can run an installer
    with pytest.raises(ValueError):
        updater.install()
    ran = []
    updater.installer = ran.append
    assert updater.status().can_install
    updater._install(updater.status().latest.asset)  # what install() runs on its thread
    assert updater.install_state.status == "installing" and ran == [tmp_path / "updates" / "Urakatsu-Setup-0.2.0.exe"]
    assert ran[0].read_bytes() == INSTALLER

    tampered = updates.Asset(name="Urakatsu-Setup-0.2.0.exe", url="https://example.test/setup.exe", size=1, sha256="0" * 64)
    updater._install(tampered)
    assert updater.install_state.status == "failed" and "SHA-256" in updater.install_state.error
