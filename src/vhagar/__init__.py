"""VHAGAR, multi-sensor wildfire intelligence.

Four tasks, one platform:

    T1  detection      active fire, near-real-time, GEO + LEO fusion
    T2  burned_area    extent and severity from Sentinel-2 / Landsat / SAR
    T3  danger         ignition probability and fire danger indices
    T4  spread         12-72 h fire progression

The public API is deliberately small. Heavy optional dependencies (torch,
rasterio, earthengine-api) are imported lazily inside the modules that need
them, so ``import vhagar`` works in a minimal environment.
"""

from __future__ import annotations

__version__ = "0.1.0"

TASKS = ("detection", "burned_area", "danger", "spread")

__all__ = ["TASKS", "__version__"]
