"""The leakage tests. If these fail, nothing downstream is trustworthy."""

from __future__ import annotations

from datetime import date

import pytest

from vhagar.eval import splits as S


def make_units(n: int = 60) -> list[S.SplitUnit]:
    out = []
    for i in range(n):
        out.append(
            S.SplitUnit(
                uid=f"u{i:03d}",
                lon=-125.0 + (i % 10) * 4.0,
                lat=32.0 + (i // 10) * 3.0,
                when=date(2017 + i % 5, 7, 1 + i % 20),
                group=f"fire{i // 6:02d}",
                ecoregion=f"eco{i % 4}",
                continent="NA" if i % 3 else "EU",
                tile_id=f"conus/x{i % 7:04d}_y{i % 5:04d}",
            )
        )
    return out


def test_random_split_is_unavailable():
    with pytest.raises(NotImplementedError, match="Random splits are not supported"):
        S.random_split([])


@pytest.mark.parametrize("by", ["group", "ecoregion", "continent", "tile_id"])
def test_leave_one_group_out_is_disjoint(by):
    m = S.leave_one_group_out(make_units(), by=by)
    S.verify_no_overlap(m)
    assert m.n_folds >= 2


def test_leave_one_group_out_never_shares_a_group():
    units = make_units()
    by_uid = {u.uid: u.group for u in units}
    m = S.leave_one_group_out(units, by="group")
    for fold in m.folds:
        train_groups = {by_uid[u] for u in fold["train"]}
        test_groups = {by_uid[u] for u in fold["test"]}
        assert not (train_groups & test_groups), "a fire event appeared in train and test"


def test_spatial_block_split_is_disjoint_and_blocks_are_whole():
    units = make_units()
    m = S.spatial_block_split(units, n_folds=4, block_degrees=5.0, seed=7)
    S.verify_no_overlap(m)

    def block(u):
        return (int(u.lon // 5.0), int(u.lat // 5.0))

    by_uid = {u.uid: block(u) for u in units}
    for fold in m.folds:
        train_blocks = {by_uid[u] for u in fold["train"]}
        test_blocks = {by_uid[u] for u in fold["test"]}
        assert not (train_blocks & test_blocks), "a spatial block was split across train/test"


def test_leave_year_out_is_chronologically_disjoint():
    units = make_units()
    by_uid = {u.uid: u.when.year for u in units}
    m = S.leave_year_out(units, n_test_years=1, n_val_years=1)
    S.verify_no_overlap(m)
    for fold in m.folds:
        test_years = {by_uid[u] for u in fold["test"]}
        train_years = {by_uid[u] for u in fold["train"]}
        assert not (test_years & train_years)
        # Validation must PRECEDE the test block (no future leakage into model
        # selection): every val year is strictly earlier than every test year.
        val_years = {by_uid[u] for u in fold.get("val", [])}
        if val_years:
            assert max(val_years) < min(test_years)
            assert not (val_years & test_years)


def test_verify_no_overlap_detects_leakage():
    bad = S.SplitManifest(scheme="broken", folds=[{"train": ["a", "b"], "test": ["b"]}])
    with pytest.raises(AssertionError, match="in both"):
        S.verify_no_overlap(bad)


def test_manifest_roundtrip_and_fingerprint(tmp_path):
    m = S.leave_year_out(make_units())
    p = m.to_json(tmp_path / "m.json")
    m2 = S.SplitManifest.from_json(p)
    assert m2.fingerprint() == m.fingerprint()
    assert m2.n_folds == m.n_folds


def test_duplicate_uids_rejected():
    u = S.SplitUnit("dup", 0.0, 0.0, date(2020, 1, 1))
    with pytest.raises(ValueError, match="duplicate"):
        S.leave_year_out([u, u])


def test_spatial_block_rejects_too_many_folds():
    units = [S.SplitUnit(f"u{i}", 0.1 * i, 0.0, date(2020, 1, 1)) for i in range(5)]
    with pytest.raises(ValueError, match="spatial blocks"):
        S.spatial_block_split(units, n_folds=5, block_degrees=90.0)
