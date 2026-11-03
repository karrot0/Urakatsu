"""Suwayomi Server: using a server that already runs elsewhere instead of Urakatsu's own."""

import json

import httpx
import pytest

from urakatsu.runtimes.packages import PackageManager
from urakatsu.sources import suwayomi
from urakatsu.sources.suwayomi import SuwayomiSystem


class FakeServer:
    """An always-on Suwayomi with basic authentication, answering GraphQL and thumbnails."""

    def __init__(self, url="http://nas.local:4567", user="me", password="secret"):
        self.url, self.user, self.password = url, user, password
        self.requests: list[tuple[str, object]] = []

    def _check(self, url: str, auth) -> httpx.Response | None:
        self.requests.append((url, auth))
        request = httpx.Request("GET", url)
        if not url.startswith(self.url + "/"):
            return httpx.Response(404, request=request)
        if auth is None or auth._auth_header != httpx.BasicAuth(self.user, self.password)._auth_header:
            return httpx.Response(401, request=request)
        return None

    def post(self, url, json=None, timeout=None, auth=None):
        refused = self._check(url, auth)
        if refused is not None:
            return refused
        if "aboutServer" in json["query"]:
            data = {"aboutServer": {"version": "v2.1.1867"}}
        else:
            data = {"fetchSourceManga": {"hasNextPage": False, "mangas": [{"id": 5, "title": "Series", "thumbnailUrl": "/api/v1/manga/5/thumbnail", "realUrl": None}]}}
        return httpx.Response(200, json={"data": data}, request=httpx.Request("POST", url))

    def get(self, url, timeout=None, follow_redirects=True, auth=None):
        if url.startswith(self.url):
            refused = self._check(url, auth)
            if refused is not None:
                return refused
        else:
            self.requests.append((url, auth))
        return httpx.Response(200, content=b"jpeg", headers={"content-type": "image/jpeg"}, request=httpx.Request("GET", url))


@pytest.fixture
def server(monkeypatch):
    fake = FakeServer()
    monkeypatch.setattr(suwayomi.httpx, "post", fake.post)
    monkeypatch.setattr(suwayomi.httpx, "get", fake.get)
    return fake


def _system(tmp_path):
    return SuwayomiSystem(tmp_path / "suwayomi", PackageManager(tmp_path / "runtimes", offline=True))


def test_a_remote_server_is_used_without_installing_or_starting_one(tmp_path, server, monkeypatch):
    system = _system(tmp_path)
    monkeypatch.setattr(system, "_launch", lambda: pytest.fail("a remote server must not start a local one"))
    status = system.connect("http://nas.local:4567/", "me", "secret")
    assert (status.state, status.server, status.username, status.version, status.runtime) == ("running", "http://nas.local:4567", "me", "v2.1.1867", None)
    assert status.connectable
    # The password stays on this computer; the status never carries it.
    assert "secret" not in status.model_dump_json()
    assert json.loads((tmp_path / "suwayomi" / "remote.json").read_text())["password"] == "secret"

    # Covers carry the server, so another server's manga 5 is not the same cached cover.
    found = system.search("1", "query")
    assert found.results[0].cover == "http://nas.local:4567/api/v1/manga/5/thumbnail"
    assert system.cover("1", found.results[0].cover).data == b"jpeg"

    # Stopping lets go of the server; it keeps running elsewhere.
    system.stop()
    assert system.status().state == "stopped"
    assert system.start().state == "running"
    # A new instance (Urakatsu restarted) finds the saved server.
    assert _system(tmp_path).status().server == "http://nas.local:4567"


def test_the_sign_in_never_goes_to_another_host(tmp_path, server):
    system = _system(tmp_path)
    system.connect("http://nas.local:4567", "me", "secret")
    system.cover("1", "https://cdn.example.com/cover.jpg")
    assert server.requests[-1] == ("https://cdn.example.com/cover.jpg", None)


def test_a_refused_sign_in_is_reported_and_the_address_kept(tmp_path, server):
    system = _system(tmp_path)
    status = system.connect("http://nas.local:4567", "me", "wrong")
    assert status.state == "error" and "did not accept" in status.message
    status = system.connect("http://nas.local:4567", "", None)
    assert status.state == "error" and "asks for a sign-in" in status.message
    assert status.server == "http://nas.local:4567"


def test_the_saved_password_stays_with_its_address(tmp_path, server):
    system = _system(tmp_path)
    system.connect("http://nas.local:4567", "me", "secret")
    # Saving again without a password keeps it for the same address and user...
    assert system.connect("http://nas.local:4567", "me", None).state == "running"
    # ...but never sends it to a new address.
    server.url = "http://other.local:4567"
    system.connect("http://other.local:4567", "me", None)
    assert system.remote().password == ""


def test_no_address_goes_back_to_the_own_server(tmp_path, server):
    system = _system(tmp_path)
    system.connect("http://nas.local:4567", "me", "secret")
    status = system.connect(None)
    assert status.server is None and status.state == "not_installed" and status.runtime == "suwayomi-server"
    assert not (tmp_path / "suwayomi" / "remote.json").exists()


def test_an_address_must_be_http():
    with pytest.raises(ValueError):
        suwayomi.RemoteServer(url="nas.local:4567")
    assert suwayomi.RemoteServer(url="https://nas.local/api/graphql").url == "https://nas.local"
