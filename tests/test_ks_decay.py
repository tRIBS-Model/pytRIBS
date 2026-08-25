"""Tests for compute_ks_decay: recovery of the Ivanov et al. (2004) decay parameter f."""
import numpy as np
import pytest

from pytRIBS.shared.inout import InOut
from pytRIBS.soil.soil import SoilProcessor

NODATA = -9999.0
DEPTHS_MM = [1.0, 100.0, 400.0, 800.0]


def ivanov_ks(k0, f, z):
    """Ivanov et al. (2004) eqn 17: mean Ks over depth z given surface K0."""
    z = np.asarray(z, dtype=float)
    return k0 * (f * z) / (np.exp(f * z) - 1.0)


def test_compute_ks_decay_recovers_known_f(make_ascii_raster, tmp_path):
    k0 = 10.0  # mm/hr at the surface
    f_slow, f_fast = 0.005, 0.02  # 1/mm

    # Pixel layout:  (0,0) slow decay | (0,1) uniform profile (hits the floor)
    #                (1,0) nodata     | (1,1) fast decay
    grid_input = []
    for depth in DEPTHS_MM:
        data = np.array([
            [ivanov_ks(k0, f_slow, depth), k0],
            [k0, ivanov_ks(k0, f_fast, depth)],
        ])
        if depth == 100.0:
            data[1, 0] = NODATA  # poison one layer of pixel (1,0)
        path = make_ascii_raster(f'ks_{int(depth)}mm.asc', data, nodata=NODATA)
        grid_input.append({'depth': depth, 'path': path})

    out = str(tmp_path / 'f.asc')
    SoilProcessor().compute_ks_decay(grid_input, output=out)

    f_grid = InOut.read_ascii(out)['data']

    assert f_grid[0, 0] == pytest.approx(f_slow, rel=0.15)
    assert f_grid[1, 1] == pytest.approx(f_fast, rel=0.15)
    # Uniform Ks profile must be floored at min_f, not left at ~0
    assert f_grid[0, 1] == pytest.approx(1e-4, abs=1e-5)
    # Nodata in any input layer propagates nodata to the output
    assert f_grid[1, 0] == pytest.approx(NODATA)
