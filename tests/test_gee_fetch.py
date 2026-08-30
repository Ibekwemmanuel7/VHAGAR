"""Earth Engine patch fetcher: the worker must be picklable (module-level, not a
closure) and ``fetch_patches`` must raise when EVERY request fails instead of
silently yielding nothing. The ProcessPoolExecutor is stubbed with an in-process
fake so the logic is testable without ``ee`` or real subprocesses."""
from __future__ import annotations

import concurrent.futures as _cf
import pickle

import numpy as np
import pytest

from vhagar.io import gee


def test_worker_is_module_level_and_picklable():
    # A nested closure could not be pickled; the module-level worker can, which is
    # what ProcessPoolExecutor needs under the spawn/forkserver start methods.
    assert pickle.loads(pickle.dumps(gee._fetch_patch_worker)) is gee._fetch_patch_worker


class _FakeFuture:
    def __init__(self, fn, *args):
        self._fn, self._args = fn, args

    def result(self):
        return self._fn(*self._args)


class _FakePool:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def submit(self, fn, *args):
        return _FakeFuture(fn, *args)


def _req(uid):
    return gee.PatchRequest(uid=uid, bounds=(0, 0, 1, 1), crs="EPSG:4326", bands=("a",))


@pytest.fixture
def _inproc(monkeypatch):
    monkeypatch.setattr(_cf, "ProcessPoolExecutor", _FakePool)
    monkeypatch.setattr(_cf, "as_completed", lambda futs: list(futs))
    monkeypatch.setattr(gee, "initialize", lambda **k: None)


def _factory_ok():
    return object()


def test_fetch_patches_raises_when_all_fail(_inproc, monkeypatch):
    def boom(image, req):
        raise RuntimeError("quota exceeded")

    monkeypatch.setattr(gee, "fetch_patch", boom)
    with pytest.raises(RuntimeError, match="all 3 GEE patch requests failed"):
        list(gee.fetch_patches(_factory_ok, [_req("a"), _req("b"), _req("c")]))


def test_fetch_patches_yields_and_skips_partial_failures(_inproc, monkeypatch):
    def sometimes(image, req):
        if req.uid == "bad":
            raise RuntimeError("one bad chip")
        return np.zeros((1, 2, 2), dtype=np.float32)

    monkeypatch.setattr(gee, "fetch_patch", sometimes)
    got = dict(gee.fetch_patches(_factory_ok, [_req("ok1"), _req("bad"), _req("ok2")]))
    assert set(got) == {"ok1", "ok2"}        # the bad chip is skipped, not fatal
