"""`aegis serve` must not expose /v1/* unauthenticated by accident."""

import pytest

from aegis import cli


@pytest.fixture
def served(monkeypatch):
    """Record uvicorn.run calls instead of starting a server."""
    import uvicorn

    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda app, host, port: calls.append(host))
    monkeypatch.delenv("AEGIS_API_KEY", raising=False)
    return calls


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.20"])
def test_public_bind_without_key_is_refused(served, host, capsys):
    assert cli.main(["serve", "--host", host]) == 2
    assert served == []
    assert "refusing" in capsys.readouterr().err


@pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.2", "localhost", "::1"])
def test_loopback_without_key_serves(served, host):
    assert cli.main(["serve", "--host", host]) == 0
    assert served == [host]


def test_public_bind_with_key_serves(served, monkeypatch):
    monkeypatch.setenv("AEGIS_API_KEY", "k" * 32)
    assert cli.main(["serve", "--host", "0.0.0.0"]) == 0
    assert served == ["0.0.0.0"]


def test_explicit_opt_out_serves_with_warning(served, capsys):
    assert cli.main(["serve", "--host", "0.0.0.0", "--allow-unauthenticated"]) == 0
    assert served == ["0.0.0.0"]
    assert "WARNING" in capsys.readouterr().err
