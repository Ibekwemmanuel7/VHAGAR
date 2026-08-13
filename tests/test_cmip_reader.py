"""ABI CMIP thermal-channel decoding.

Runs entirely offline against a synthetic CMIP granule, so CI never depends on
S3. The CMIP granule shares its grid with the synthetic FDC granule, which lets
us prove the navigation cache is reused across products, the whole reason the
radiance tier is affordable in wall clock.
"""

from __future__ import annotations

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

from _fixtures import _synthetic_cmip, _synthetic_fdc  # noqa: E402

from vhagar.io.cmip_reader import (  # noqa: E402
    CMIP_CHANNELS,
    decode_cmip,
)
from vhagar.io.goes_reader import decode_fdc  # noqa: E402
from vhagar.physics.planck import brightness_temperature, planck_radiance  # noqa: E402

# ------------------------------------------------------- decoding --------


def test_decode_produces_bt_and_geometry_on_the_grid():
    c = decode_cmip(_synthetic_cmip(), satellite=18, channel="C07")
    assert c.bt_k.shape == c.lat.shape == c.view_zenith_deg.shape
    assert c.band == "C07"
    assert c.wavelength_um == CMIP_CHANNELS["C07"]
    assert np.isfinite(c.lat).all() and np.isfinite(c.lon).all()
    assert np.all(c.view_zenith_deg >= 0.0)
    assert np.all(c.true_pixel_area_m2 > 0.0)
    assert c.scan_start.year >= 2000


def test_cmi_is_treated_as_brightness_temperature():
    """CMI is already kelvin, so a value round-trips through planck at the band
    centre. This is the assumption the whole decoder rests on."""
    c = decode_cmip(_synthetic_cmip(), satellite=18, channel="C07")
    warm = c.bt_k[10, 12]
    assert warm == pytest.approx(330.0)
    rad = planck_radiance(c.wavelength_um, warm)
    assert float(brightness_temperature(c.wavelength_um, rad)) == pytest.approx(330.0, abs=1e-6)


def test_saturated_mir_pixel_is_censored_not_passed_through():
    """ABI Ch7 saturates near 400 K. The pixel is a real hot source, so it is
    flagged and its value dropped rather than used as a temperature."""
    c = decode_cmip(_synthetic_cmip(), satellite=18, channel="C07")
    assert c.saturated[14, 16], "the 450 K pixel must be marked saturated"
    assert np.isnan(c.bt_k[14, 16]), "a saturated value must not survive as a number"
    assert c.n_saturated == 1


def test_window_channel_does_not_censor_at_the_mir_threshold():
    """Only the MIR channel saturates within real scene temperatures; a window
    channel at 450 K would be nonphysical but must not be silently censored by
    the MIR rule."""
    c = decode_cmip(_synthetic_cmip(channel="C14"), satellite=18, channel="C14")
    assert c.n_saturated == 0
    assert c.bt_k[14, 16] == pytest.approx(450.0)


def test_fill_and_bad_dqf_become_nan():
    c = decode_cmip(_synthetic_cmip(), satellite=18, channel="C07")
    assert np.isnan(c.bt_k[18, 20]), "fill value must be NaN"
    assert np.isnan(c.bt_k[22, 24]), "out-of-range DQF must be NaN"
    # a good pixel with DQF 0 survives
    assert np.isfinite(c.bt_k[10, 12])


def test_valid_mask_counts_usable_pixels():
    c = decode_cmip(_synthetic_cmip(), satellite=18, channel="C07")
    # background is all 300 K and usable; the planted bad ones are excluded.
    assert c.valid_mask().sum() == c.bt_k.size - 3  # saturated, fill, bad-DQF


def test_unknown_channel_is_rejected():
    with pytest.raises(ValueError, match="unknown channel"):
        decode_cmip(_synthetic_cmip(), satellite=18, channel="C02")


# ------------------------------------------------- bbox and navigation ---


def test_bbox_crop_reduces_the_grid():
    full = decode_cmip(_synthetic_cmip(n=60), satellite=18, channel="C07")
    lat_mid = float(np.nanmedian(full.lat))
    lon_mid = float(np.nanmedian(full.lon))
    small = decode_cmip(
        _synthetic_cmip(n=60),
        satellite=18,
        channel="C07",
        bbox=(lon_mid - 0.3, lat_mid - 0.3, lon_mid + 0.3, lat_mid + 0.3),
    )
    assert small.bt_k.size < full.bt_k.size
    assert np.isfinite(small.lat).all()


def test_navigation_cache_is_shared_across_channels_and_with_fdc():
    """The efficiency argument for the whole radiance tier: geometry is computed
    once for a grid and reused by every channel, and by FDC, on that grid."""
    from vhagar.io.goes_reader import _NAV_CACHE_STATS, _clear_nav_cache

    _clear_nav_cache()
    decode_fdc(_synthetic_fdc(n=40), satellite=18)          # miss: builds the grid
    decode_cmip(_synthetic_cmip(n=40, channel="C07"), satellite=18, channel="C07")  # hit
    decode_cmip(_synthetic_cmip(n=40, channel="C14"), satellite=18, channel="C14")  # hit
    assert _NAV_CACHE_STATS["misses"] == 1, "the same grid must be built only once"
    assert _NAV_CACHE_STATS["hits"] == 2


def test_cmip_geometry_is_identical_to_fdc_on_the_same_grid():
    """Same grid, same geometry: the shared cache is not just fast, it is the
    same arrays, so FDC detections and CMIP pixels co-register exactly."""
    from vhagar.io.goes_reader import _clear_nav_cache

    _clear_nav_cache()
    f = decode_fdc(_synthetic_fdc(n=40), satellite=18)
    c = decode_cmip(_synthetic_cmip(n=40), satellite=18, channel="C07")
    assert c.lat is f.lat
    assert c.true_pixel_area_m2 is f.true_pixel_area_m2


# ------------------------------------------------- multi-channel stacks --

from datetime import UTC, datetime, timedelta  # noqa: E402

from vhagar.io.cmip_reader import (  # noqa: E402
    group_cmip_keys_by_timestamp,
    open_cmip_stack,
    stack_channels,
)


def _cmip_key(channel: str, when: datetime) -> str:
    """Build a realistic CMIP S3 key for a channel at a given scan time."""
    doy = when.timetuple().tm_yday
    stamp = f"{when.year:04d}{doy:03d}{when.hour:02d}{when.minute:02d}{when.second:02d}0"
    name = f"OR_ABI-L2-CMIPC-M6{channel}_G18_s{stamp}_e{stamp}_c{stamp}.nc"
    return f"ABI-L2-CMIPC/{when.year}/{doy:03d}/{when.hour:02d}/{name}"


def test_stack_channels_shares_one_geometry_and_keys_bt_by_band():
    from vhagar.io.goes_reader import _clear_nav_cache

    _clear_nav_cache()
    chans = [
        decode_cmip(_synthetic_cmip(n=40, channel=b), satellite=18, channel=b)
        for b in ("C07", "C14", "C15")
    ]
    stack = stack_channels(chans)
    assert stack.bands == ("C07", "C14", "C15")
    assert set(stack.bt_k) == {"C07", "C14", "C15"}
    # geometry held once, and it is the shared cached array
    assert stack.lat is chans[0].lat
    assert stack.shape == (40, 40)


def test_stack_bt_difference_is_a_plain_coregistered_subtraction():
    from vhagar.io.goes_reader import _clear_nav_cache

    _clear_nav_cache()
    c07 = decode_cmip(_synthetic_cmip(n=40, channel="C07"), satellite=18, channel="C07")
    c14 = decode_cmip(_synthetic_cmip(n=40, channel="C14"), satellite=18, channel="C14")
    stack = stack_channels([c07, c14])
    diff = stack.bt_difference("C07", "C14")
    # both backgrounds are 300 K, so the difference is 0 where both are valid
    assert np.nanmax(np.abs(diff)) == pytest.approx(0.0)


def test_stack_rejects_channels_on_different_grids():
    from vhagar.io.goes_reader import _clear_nav_cache

    _clear_nav_cache()
    a = decode_cmip(_synthetic_cmip(n=40), satellite=18, channel="C07")
    b = decode_cmip(_synthetic_cmip(n=60), satellite=18, channel="C14")  # different grid
    with pytest.raises(ValueError, match="different grid|does not match"):
        stack_channels([a, b])


def test_stacking_nothing_raises():
    with pytest.raises(ValueError, match="no channels"):
        stack_channels([])


def test_grouping_pairs_channels_within_tolerance():
    """The five channel files of one timestep scan a few seconds apart; they must
    still group into one stack."""
    t0 = datetime(2026, 8, 3, 15, 0, 0, tzinfo=UTC)
    keys = {
        "C07": [_cmip_key("C07", t0)],
        "C14": [_cmip_key("C14", t0 + timedelta(seconds=1))],
        "C15": [_cmip_key("C15", t0 + timedelta(seconds=2))],
    }
    groups = group_cmip_keys_by_timestamp(keys, satellite=18)
    assert len(groups) == 1
    assert set(groups[0]) == {"C07", "C14", "C15"}


def test_grouping_drops_incomplete_timesteps():
    """A timestep missing a band is dropped, so no stack is ever built with a
    hole that would bias a band difference."""
    t0 = datetime(2026, 8, 3, 15, 0, 0, tzinfo=UTC)
    t1 = t0 + timedelta(minutes=5)
    keys = {
        "C07": [_cmip_key("C07", t0), _cmip_key("C07", t1)],
        "C14": [_cmip_key("C14", t0), _cmip_key("C14", t1)],
        "C15": [_cmip_key("C15", t0)],  # missing at t1
    }
    groups = group_cmip_keys_by_timestamp(keys, satellite=18)
    assert len(groups) == 1  # only t0 is complete
    assert set(groups[0]) == {"C07", "C14", "C15"}


def test_grouping_does_not_merge_two_distinct_timesteps():
    t0 = datetime(2026, 8, 3, 15, 0, 0, tzinfo=UTC)
    t1 = t0 + timedelta(minutes=5)  # well beyond the 2-minute tolerance
    keys = {
        "C07": [_cmip_key("C07", t0), _cmip_key("C07", t1)],
        "C14": [_cmip_key("C14", t0), _cmip_key("C14", t1)],
    }
    groups = group_cmip_keys_by_timestamp(keys, satellite=18)
    assert len(groups) == 2
    assert all(set(g) == {"C07", "C14"} for g in groups)


def test_open_cmip_stack_uses_the_grouping_and_stacker(monkeypatch):
    """open_cmip_stack opens each band's key and stacks them. Patch the network
    read so this stays offline."""
    import vhagar.io.cmip_reader as reader

    def fake_open(key, satellite, channel, bbox=None, anon=True):
        return decode_cmip(_synthetic_cmip(n=40, channel=channel), satellite=satellite,
                           channel=channel)

    monkeypatch.setattr(reader, "open_cmip", fake_open)
    t0 = datetime(2026, 8, 3, 15, 0, 0, tzinfo=UTC)
    group = {"C07": _cmip_key("C07", t0), "C14": _cmip_key("C14", t0)}
    stack = open_cmip_stack(group, satellite=18)
    assert stack.bands == ("C07", "C14")
    assert stack.shape == (40, 40)
