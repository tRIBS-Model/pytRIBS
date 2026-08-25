"""Round-trip tests for the ESRI ASCII raster reader/writer in InOut."""
import numpy as np
import pytest

from pytRIBS.shared.inout import InOut


def test_ascii_round_trip(make_ascii_raster):
    data = [[1.5, 2.25], [3.75, 4.0]]
    path = make_ascii_raster('grid.asc', data, cellsize=100.0)

    raster = InOut.read_ascii(path)

    assert raster['data'].tolist() == data
    assert raster['profile']['nodata'] == -9999.0
    assert raster['profile']['width'] == 2
    assert raster['profile']['height'] == 2
    # Cell size survives the header rewrite in write_ascii
    assert raster['profile']['transform'].a == pytest.approx(100.0)


def test_ascii_write_replaces_nan_with_nodata(make_ascii_raster):
    data = [[1.0, np.nan], [np.nan, 4.0]]
    path = make_ascii_raster('grid_nan.asc', data)

    raster = InOut.read_ascii(path)

    assert raster['data'][0, 1] == pytest.approx(-9999.0)
    assert raster['data'][1, 0] == pytest.approx(-9999.0)
    assert raster['data'][0, 0] == pytest.approx(1.0)
