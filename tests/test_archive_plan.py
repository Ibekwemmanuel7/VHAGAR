"""Archive sizing. The arithmetic that decides whether Step 2 is affordable."""

from __future__ import annotations

import pytest

from vhagar.archive.plan import (
    STANDARD_PLANS,
    ArchivePlan,
    recommend_plan,
    two_tier_plan,
)


def test_tile_is_48_goes_pixels_across():
    """96 km tile at 2 km GOES resolution. Tiles are small; cadence is not."""
    assert ArchivePlan("x", 1, 1, 15, 1).pixels_per_tile_side == 48


def test_cadence_is_the_steepest_cost_gradient():
    """5-minute sampling costs 3x what 15-minute does, for no climatology gain."""
    slow = ArchivePlan("slow", 100, 2, 15, 5).cost()
    fast = ArchivePlan("fast", 100, 2, 5, 5).cost()
    assert fast.disk_gb == pytest.approx(3 * slow.disk_gb, rel=1e-6)
    assert fast.granule_reads == pytest.approx(3 * slow.granule_reads, rel=1e-6)


def test_granule_reads_are_independent_of_tile_count():
    """The whole argument for granule-outer-loop. Per-tile iteration would
    multiply S3 requests by the tile count and turn hours into weeks."""
    few = ArchivePlan("few", 10, 2, 15, 5).cost()
    many = ArchivePlan("many", 500, 2, 15, 5).cost()
    assert few.granule_reads == many.granule_reads
    assert many.disk_gb == pytest.approx(50 * few.disk_gb, rel=1e-6)


def test_per_band_products_multiply_reads():
    """ABI CMIP ships one file per channel, the most-missed sizing factor."""
    multiband = ArchivePlan("fdc", 100, 1, 15, 5, one_file_per_band=False).cost()
    perband = ArchivePlan("cmip", 100, 1, 15, 5, one_file_per_band=True).cost()
    assert perband.granule_reads == 5 * multiband.granule_reads
    assert perband.disk_gb == pytest.approx(multiband.disk_gb), "disk is unchanged"


def test_download_traffic_dwarfs_disk():
    """The binding constraint is the wire, not the disk. Sizing on disk alone
    is how people commit to a month of downloading by accident."""
    c = ArchivePlan("standard", 200, 2, 15, 5).cost()
    assert c.download_tb * 1000 > 10 * c.disk_gb


def test_scaling_to_a_disk_budget_cuts_tiles_not_years():
    plan = ArchivePlan("big", 300, 3, 15, 5)
    fitted = plan.scaled_to_disk(50.0)
    assert fitted.cost().disk_gb <= 50.0
    assert fitted.n_tiles < plan.n_tiles
    assert fitted.years == plan.years, "years must survive; they are the statistics"
    assert fitted.cadence_min == plan.cadence_min


def test_scaling_is_a_noop_when_it_already_fits():
    plan = ArchivePlan("small", 10, 1, 15, 3)
    assert plan.scaled_to_disk(1000.0) is plan


def test_two_tier_reserves_a_meaningful_high_cadence_slice():
    clim, hi = two_tier_plan(100.0)
    assert hi.cadence_min == 5, "the high-cadence tier must actually be 5-minute"
    assert clim.cadence_min == 15
    total = clim.cost().disk_gb + hi.cost().disk_gb
    assert total <= 100.0
    assert hi.cost().disk_gb >= 8.0


def test_two_tier_survives_a_tiny_budget():
    clim, hi = two_tier_plan(12.0)
    assert clim.n_tiles >= 20 and hi.n_tiles >= 20


def test_naive_plan_is_an_order_of_magnitude_worse():
    by_name = {p.name: p for p in STANDARD_PLANS}
    naive = by_name["naive: everything at 5 min"].cost()
    sane = by_name["generous climatology"].cost()
    assert naive.disk_gb > 2.5 * sane.disk_gb


def test_recommendation_is_human_readable():
    text = recommend_plan(100.0)
    assert "GB disk" in text and "TB wire" in text and "OUTER loop" in text


# ---------------------------------------------------------------------------
# Storage mode. FDC is a sparse product; sizing it as a dense raster overstates
# disk by three orders of magnitude and hides the fact that the detection tier
# is nearly free.
# ---------------------------------------------------------------------------


def test_sparse_storage_is_orders_of_magnitude_smaller_than_dense():
    dense = ArchivePlan("d", 500, 3, 5, 1, storage="dense")
    sparse = ArchivePlan("s", 500, 3, 5, 1, storage="sparse")
    assert sparse.cost().disk_gb < dense.cost().disk_gb / 100


def test_sparse_disk_scales_linearly_with_detection_rate():
    base = ArchivePlan("s", 100, 1, 15, 1, storage="sparse", detection_rate=1e-5)
    busy = ArchivePlan("s", 100, 1, 15, 1, storage="sparse", detection_rate=1e-4)
    assert busy.cost().disk_gb == pytest.approx(10 * base.cost().disk_gb)


def test_sparse_storage_does_not_double_count_compression():
    """Row bytes are already post-encoding, so the raster ratio must not apply."""
    plan = ArchivePlan("s", 100, 1, 15, 1, storage="sparse", compression=4.0)
    c = plan.cost()
    assert c.disk_gb == pytest.approx(c.raw_gb)


def test_storage_mode_does_not_change_wire_or_wall_clock():
    """You read the same granules either way. Only what you keep differs."""
    dense = ArchivePlan("d", 500, 3, 5, 1, storage="dense").cost()
    sparse = ArchivePlan("s", 500, 3, 5, 1, storage="sparse").cost()
    assert sparse.granule_reads == dense.granule_reads
    assert sparse.download_tb == pytest.approx(dense.download_tb)


def test_unknown_storage_mode_is_rejected():
    with pytest.raises(ValueError, match="storage"):
        ArchivePlan("x", 1, 1, 15, 1, storage="compressed-ish").cost()


def test_fdc_tier_is_cheap_on_the_wire_relative_to_radiance():
    """The finding that reshapes Step 2: detections are nearly free to fetch.

    Compared like for like, same tiles, years and cadence. The reference plans
    deliberately differ on all three, so comparing them directly would measure
    the plan choices rather than the products.
    """
    common = dict(n_tiles=500, years=3, cadence_min=5)
    fdc = ArchivePlan(
        "fdc", n_bands=1, granule_mb=0.32, one_file_per_band=False,
        storage="sparse", **common,
    ).cost()
    radiance = ArchivePlan("rad", n_bands=5, granule_mb=4.5, **common).cost()
    assert fdc.download_tb < radiance.download_tb / 50
    assert fdc.disk_gb < radiance.disk_gb / 1000


def test_measured_radiance_size_keeps_the_full_plan_under_five_terabytes():
    """Guards the v0.9 correction: 4.5 MB granules, not the 20 MB I assumed.

    If someone restores the old default this fails loudly, because the whole
    'a full backfill is a weekend, not a month' conclusion rests on it.
    """
    from vhagar.archive.plan import three_tier_plan

    total = sum(p.cost().download_tb for p in three_tier_plan(100.0))
    assert total < 5.0


def test_abi_products_are_distinct_prefixes():
    from vhagar.archive.plan import ABI_PRODUCTS

    assert ABI_PRODUCTS["FDC"] != ABI_PRODUCTS["CMIP"]
    assert all(v.startswith("ABI-L2-") for v in ABI_PRODUCTS.values())


def test_measure_granule_rejects_unknown_product():
    pytest.importorskip("s3fs")  # measure_granule reaches the S3 stack (archive extra)
    from vhagar.archive.plan import measure_granule

    with pytest.raises(ValueError, match="product"):
        measure_granule(product="L1b")
