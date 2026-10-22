"""Tests for GenerateMesh.process_level: wavelet points must fall in their coefficient's block."""
import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point, box

from pytRIBS.mesh.mesh import GenerateMesh

X0, Y0 = 400000.0, 3900000.0  # UTM 12N origin of the synthetic DEM
RES = 5.0


@pytest.fixture
def odd_sized_dem(tmp_path):
    """A rough 37 x 53 DEM (not a multiple of 2**level at any level > 0) plus the
    watershed, stream and outlet shapefiles GenerateMesh expects."""
    rng = np.random.default_rng(0)
    rows, cols = np.mgrid[0:37, 0:53]
    z = 2000 - 2.0 * rows + 5 * np.sin(cols / 3.0) + rng.normal(0, 1.5, rows.shape)
    dem = tmp_path / "dem.tif"
    with rasterio.open(dem, "w", driver="GTiff", dtype="float32", count=1, width=53, height=37,
                       crs="EPSG:26912", transform=from_origin(X0, Y0, RES, RES), nodata=-9999.0) as dst:
        dst.write(z.astype("float32"), 1)
    paths = {"dem": dem}
    xmax, ymin = X0 + 53 * RES, Y0 - 37 * RES
    for name, geom in (("watershed", box(X0, ymin, xmax, Y0)),
                       ("streams", LineString([((X0 + xmax) / 2, Y0 - 5), ((X0 + xmax) / 2, ymin + 5)])),
                       ("outlet", Point((X0 + xmax) / 2, ymin + 5))):
        paths[name] = tmp_path / f"{name}.shp"
        gpd.GeoDataFrame(geometry=[geom], crs="EPSG:26912").to_file(paths[name])
    return paths


def test_points_fall_in_their_coefficient_block(odd_sized_dem):
    p = odd_sized_dem
    gm = GenerateMesh(str(p["dem"]), str(p["watershed"]), str(p["streams"]), str(p["outlet"]),
                      None, "curvature")
    assert gm.maxlevel >= 3
    for level in range(1, gm.maxlevel + 1):
        cell = 2 ** level
        v, h, d = (gm.wavelet_packet[k * level].data for k in "vhd")
        norm = np.maximum.reduce([np.abs(v), np.abs(h), np.abs(d)]) / gm.normalizing_coeff
        sig = np.argwhere(norm > 2 ** (-level + 1) * 0.0)          # threshold 0: every block
        pts = np.asarray(gm.process_level(level, 0.0))
        assert len(pts) == len(sig)
        # points sit on cell corners (x_grid / y_grid are the cells' upper-left corners)
        col = np.rint((pts[:, 0] - X0) / RES).astype(int)
        row = np.rint((Y0 - pts[:, 1]) / RES).astype(int)
        assert (row // cell == sig[:, 0]).all(), f"level {level}: row outside its block"
        assert (col // cell == sig[:, 1]).all(), f"level {level}: column outside its block"
        # the last (partial) block row/column is reached, not squeezed out
        assert row.max() // cell == v.shape[0] - 1 and col.max() // cell == v.shape[1] - 1
