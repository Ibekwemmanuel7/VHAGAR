import numpy as np

from vhagar.models.rothermel import fbfm13_to_fbfm40_proxy, params_for_codes


def test_fbfm13_adapter_preserves_nonburnable_and_maps_fuel_groups():
    source = np.array([1, 2, 4, 8, 10, 11, 13, 91, 99])
    proxy = fbfm13_to_fbfm40_proxy(source)
    assert proxy.tolist() == [101, 121, 141, 181, 163, 201, 204, 91, 99]
    *_, burnable = params_for_codes(proxy)
    assert burnable.tolist() == [True, True, True, True, True, True, True, False, False]
