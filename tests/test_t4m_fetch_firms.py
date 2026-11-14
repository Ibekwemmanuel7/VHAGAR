"""Contract tests for bounded historical FIRMS requests."""

from datetime import date
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _script_module():
    path = Path(__file__).parents[1] / "scripts" / "t4m_fetch_firms.py"
    spec = spec_from_file_location("t4m_fetch_firms", path)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_firms_date_batches_are_contiguous_and_bounded():
    script = _script_module()
    batches = list(script._date_batches(date(2020, 9, 7), date(2020, 9, 18)))
    assert batches == [
        (date(2020, 9, 7), date(2020, 9, 11)),
        (date(2020, 9, 12), date(2020, 9, 16)),
        (date(2020, 9, 17), date(2020, 9, 18)),
    ]
