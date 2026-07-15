import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from pytRIBS.shared.inout import InOut


@pytest.fixture
def make_ascii_raster(tmp_path):
    """Factory that writes a small ESRI ASCII grid and returns its path.

    Grids are placed in UTM zone 12N around a synthetic Arizona-like origin so
    that any code touching the CRS or transform sees realistic values.
    """

    def _make(name, data, nodata=-9999.0, cellsize=100.0):
        data = np.asarray(data, dtype='float32')
        profile = {
            'driver': 'AAIGrid',
            'dtype': 'float32',
            'count': 1,
            'nodata': nodata,
            'width': data.shape[1],
            'height': data.shape[0],
            'crs': rasterio.crs.CRS.from_epsg(32612),
            'transform': from_origin(400000.0, 3900000.0, cellsize, cellsize),
        }
        path = str(tmp_path / name)
        InOut.write_ascii({'data': data, 'profile': profile}, path)
        return path

    return _make
