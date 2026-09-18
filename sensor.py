"""The one place the sensor's grid shape is defined.

Both the simulated LiDAR and the real Ouster OS1-64 produce scans on the
same (64 rings x 1024 azimuth columns) grid -- that match is the whole
reason a model trained in simulation can be fed real scans at all (see
ros2_inference/). This shape was previously restated in ten separate
modules under three different names, which is nine chances for a mismatch
nothing would catch.

Deliberately NOT imported by tests/conftest.py: a fixture that builds its
test data from the same constant it validates against can't catch that
constant changing, so the test suite keeps its own independent copy.
"""
from __future__ import annotations

N_RINGS = 64
N_COLS = 1024
