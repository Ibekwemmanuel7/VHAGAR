"""Diurnal climatology reducer: streaming per-pixel, per-hour mean and variance.

The properties that matter: the online statistics equal the direct numpy ones,
NaN samples are excluded per pixel, merging shards equals a single pass, and a
saved accumulator round-trips. All offline.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from vhagar.archive.climatology import DiurnalClimatology

T15 = datetime(2026, 8, 3, 15, 0, 0, tzinfo=UTC)


def _frames():
    """Three 2x2 frames for one channel, with a NaN planted at pixel (1, 0)."""
    return np.array(
        [
            [[300.0, 310.0], [290.0, 320.0]],
            [[302.0, 308.0], [np.nan, 322.0]],
            [[298.0, 312.0], [294.0, 318.0]],
        ]
    )


def test_mean_and_std_match_numpy_nan_aware():
    frames = _frames()
    clim = DiurnalClimatology(["C07"], shape=(2, 2))
    for f in frames:
        clim.update_frame(15, {"C07": f})

    assert np.allclose(clim.mean("C07")[15], np.nanmean(frames, axis=0), equal_nan=True)
    # sample std, ddof=1, computed only where at least two samples exist
    expected = np.nanstd(frames, axis=0, ddof=1)
    assert np.allclose(clim.std("C07")[15], expected, equal_nan=True)


def test_per_pixel_count_reflects_only_valid_samples():
    frames = _frames()
    clim = DiurnalClimatology(["C07"], shape=(2, 2))
    for f in frames:
        clim.update_frame(15, {"C07": f})
    counts = clim.count("C07")[15]
    assert counts[0, 0] == 3
    assert counts[1, 0] == 2  # one NaN at this pixel
    assert clim.mean("C07")[15][1, 0] == pytest.approx(292.0)  # (290 + 294) / 2


def test_a_bin_never_updated_reads_as_nan():
    clim = DiurnalClimatology(["C07"], shape=(2, 2))
    clim.update_frame(15, {"C07": np.full((2, 2), 300.0)})
    assert np.all(np.isnan(clim.mean("C07")[3]))       # untouched bin
    assert np.all(clim.count("C07")[3] == 0)


def test_variance_is_nan_with_fewer_than_two_samples():
    clim = DiurnalClimatology(["C07"], shape=(1, 1))
    clim.update_frame(0, {"C07": np.array([[305.0]])})
    assert np.isnan(clim.variance("C07")[0, 0, 0])     # only one sample
    assert clim.mean("C07")[0, 0, 0] == pytest.approx(305.0)


def test_different_hours_land_in_different_bins():
    clim = DiurnalClimatology(["C07"], shape=(1, 1), n_bins=24)
    assert clim.bin_for(datetime(2026, 8, 3, 15, 10, tzinfo=UTC)) == 15
    assert clim.bin_for(datetime(2026, 8, 3, 6, 0, tzinfo=UTC)) == 6


def test_finer_bins_must_tile_the_day():
    DiurnalClimatology(["C07"], shape=(1, 1), n_bins=48)  # 30-minute bins, ok
    DiurnalClimatology(["C07"], shape=(1, 1), n_bins=96)  # 15-minute bins, ok
    with pytest.raises(ValueError, match="1440"):
        DiurnalClimatology(["C07"], shape=(1, 1), n_bins=7)


def test_merge_equals_a_single_pass_over_the_combined_stream():
    frames = _frames()
    single = DiurnalClimatology(["C07"], shape=(2, 2))
    for f in frames:
        single.update_frame(15, {"C07": f})

    a = DiurnalClimatology(["C07"], shape=(2, 2))
    a.update_frame(15, {"C07": frames[0]})
    b = DiurnalClimatology(["C07"], shape=(2, 2))
    for f in frames[1:]:
        b.update_frame(15, {"C07": f})
    merged = a.merge(b)

    assert np.allclose(merged.count("C07"), single.count("C07"))
    assert np.allclose(merged.mean("C07")[15], single.mean("C07")[15], equal_nan=True)
    assert np.allclose(merged.std("C07")[15], single.std("C07")[15], equal_nan=True)


def test_merge_rejects_a_different_layout():
    a = DiurnalClimatology(["C07"], shape=(2, 2))
    b = DiurnalClimatology(["C07", "C14"], shape=(2, 2))
    with pytest.raises(ValueError, match="different layout"):
        a.merge(b)


def test_save_and_load_round_trip(tmp_path):
    frames = _frames()
    clim = DiurnalClimatology(["C07", "C14"], shape=(2, 2))
    for f in frames:
        clim.update_frame(15, {"C07": f, "C14": f - 5.0})
    path = clim.save(tmp_path / "clim.npz")
    back = DiurnalClimatology.load(path)

    assert back.channels == clim.channels
    assert back.shape == clim.shape
    for c in ("C07", "C14"):
        assert np.allclose(back.mean(c)[15], clim.mean(c)[15], equal_nan=True)
        assert np.allclose(back.count(c), clim.count(c))


def test_update_from_a_cmip_stack_uses_the_utc_hour_bin():
    """The io-facing path: a stack folds into the bin of its scan start."""
    pytest.importorskip("xarray")
    from _fixtures import _synthetic_cmip

    from vhagar.io.cmip_reader import decode_cmip, stack_channels
    from vhagar.io.goes_reader import _clear_nav_cache

    _clear_nav_cache()
    chans = [
        decode_cmip(_synthetic_cmip(n=40, channel=b), satellite=18, channel=b)
        for b in ("C07", "C14")
    ]
    stack = stack_channels(chans)
    stack.scan_start = T15  # pin the bin

    clim = DiurnalClimatology(["C07", "C14"], shape=(40, 40))
    clim.update(stack)
    # the background is 300 K and valid, so its bin-15 count is 1 there
    assert clim.count("C07")[15][0, 0] == 1
    assert np.all(clim.count("C07")[3] == 0)
