import numpy as np

from vhagar.models.spread import dynamic_anisotropic_forecast


def test_dynamic_solver_advances_one_frame_at_a_time() -> None:
    ros = np.ones((2, 31, 31))
    wind = np.zeros_like(ros)
    direction = np.zeros_like(ros)
    seed = np.zeros((31, 31), dtype=bool)
    seed[15, 15] = True
    burned, arrival = dynamic_anisotropic_forecast(ros, wind, direction, seed, [0.1, 0.1], dx=1)
    assert burned[15, 15]
    assert burned[15, 25]
    assert not burned[15, 30]
    assert 0 < arrival[15, 25] <= 0.2


def test_dynamic_solver_rejects_mismatched_time_axis() -> None:
    field = np.ones((2, 5, 5))
    with np.testing.assert_raises_regex(ValueError, "frame_hours"):
        dynamic_anisotropic_forecast(field, field, field, np.zeros((5, 5), bool), [1])
