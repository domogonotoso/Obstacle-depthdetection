"""Downloads must never replace cached checkpoints with corrupt responses."""
import hashlib
import io

import pytest

from app import model_assets


def test_reject_corrupt_cache_without_network(tmp_path, monkeypatch):
    monkeypatch.setitem(model_assets.ASSETS, "test", ("weights.pth", "https://example.invalid", "0" * 64, 1024))
    (tmp_path / "weights.pth").write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        model_assets.ensure_asset("test", tmp_path)


def test_failed_download_does_not_leave_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setitem(model_assets.ASSETS, "test", ("weights.pth", "https://example.invalid", "0" * 64, 1024))
    monkeypatch.setattr(model_assets.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"wrong"))
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        model_assets.ensure_asset("test", tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_verified_download_is_cached(tmp_path, monkeypatch):
    content = b"official-test-bytes"
    digest = hashlib.sha256(content).hexdigest()
    monkeypatch.setitem(model_assets.ASSETS, "test", ("weights.pth", "https://example.invalid", digest, 1024))
    monkeypatch.setattr(model_assets.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(content))
    path = model_assets.ensure_asset("test", tmp_path)
    monkeypatch.setattr(model_assets.urllib.request, "urlopen", lambda *a, **k: pytest.fail("cached file downloaded again"))
    assert model_assets.ensure_asset("test", tmp_path) == path
    assert path.read_bytes() == content
