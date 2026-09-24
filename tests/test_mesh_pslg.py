"""Tests for MeshFromPSLG: every stream segment in the final TIN must be Delaunay."""
import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point, box

from pytRIBS.mesh.mesh import MeshFromPSLG, _edge_opposite_angle_sums

X0, Y0 = 400000.0, 3900000.0  # UTM 12N origin of the synthetic basin
STREAM_X = X0 + 150.0


def test_edge_opposite_angle_sums():
    # Unit square split on a diagonal: co-circular, opposite angles 90 + 90
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    edges, sums = _edge_opposite_angle_sums(sq, [[0, 1, 2], [0, 2, 3]])
    assert edges.tolist() == [[0, 2]]
    assert sums[0] == pytest.approx(180.0)

    # Squash the far corners toward the shared edge: both opposite angles turn obtuse
    kite = np.array([[0, 0], [1, 0.1], [2, 0], [1, -0.1]], float)
    _, sums = _edge_opposite_angle_sums(kite, [[0, 1, 2], [0, 2, 3]])
    assert sums[0] > 180.0


def _valley_basin(tmp_path, berm=0.0):
    """A 300 m square basin drained by one straight N-S stream, with an interior
    point 1 m off the stream that encroaches its segment (clearance disabled).
    ``berm`` raises the DEM by that many metres south of Y0 + 10, under the stream
    terminal and the outlet boundary node, like a bank sampled off the channel."""
    res = 5.0
    ny = nx = 80  # covers the basin plus the 30 m boundary buffer
    left, top = X0 - 50.0, Y0 + 350.0
    xs = left + res * (np.arange(nx) + 0.5)
    ys = top - res * (np.arange(ny) + 0.5)
    gx, gy = np.meshgrid(xs, ys)
    dem = 100.0 + 0.05 * np.abs(gx - STREAM_X) + 0.02 * (gy - Y0)
    dem = (dem + np.where(gy < Y0 + 10.0, berm, 0.0)).astype('float32')
    dem_path = str(tmp_path / 'dem.tif')
    with rasterio.open(dem_path, 'w', driver='GTiff', height=ny, width=nx, count=1,
                       dtype='float32', crs='EPSG:26912',
                       transform=from_origin(left, top, res, res)) as dst:
        dst.write(dem, 1)

    crs = 'EPSG:26912'
    watershed = gpd.GeoDataFrame(geometry=[box(X0, Y0, X0 + 300, Y0 + 300)], crs=crs)
    stream = gpd.GeoDataFrame(
        geometry=[LineString([(STREAM_X, Y0 + 290), (STREAM_X, Y0 + 5)])], crs=crs)
    outlet = gpd.GeoDataFrame(geometry=[Point(STREAM_X, Y0 + 5)], crs=crs)

    g = X0 + 12.5 + 25.0 * np.arange(12)  # 25 m grid, never on the stream line
    gxx, gyy = np.meshgrid(g, g - X0 + Y0)
    interior = np.column_stack([gxx.ravel(), gyy.ravel()])
    interior = np.vstack([interior, [STREAM_X + 1.0, Y0 + 150.0]])
    return interior, watershed, stream, dem_path, outlet


@pytest.fixture
def valley_basin(tmp_path):
    return _valley_basin(tmp_path)


def _mesh(valley_basin):
    interior, watershed, stream, dem_path, outlet = valley_basin
    pslg = MeshFromPSLG(interior, watershed, stream, dem_path, outlet=outlet,
                        boundary_buffer_dist=30.0, boundary_spacing=40.0,
                        stream_point_spacing=30.0, stream_clearance_radius=0.0,
                        mesh_quality_opts='q15', separate_parallel_streams=True)
    pslg.generate()
    return pslg


def test_encroached_stream_segment_is_split(valley_basin, monkeypatch):
    # Without conforming, the point 1 m off the stream leaves a non-Delaunay stream edge
    monkeypatch.setattr(MeshFromPSLG, 'MAX_CONFORM_PASSES', 0)
    raw = _mesh(valley_basin)
    assert raw.n_non_delaunay_edges > 0
    monkeypatch.undo()

    pslg = _mesh(valley_basin)
    assert pslg.n_non_delaunay_edges == 0
    _, sums = _edge_opposite_angle_sums(pslg.vertices[:, :2], pslg.triangles)
    assert (sums <= 180.0 + MeshFromPSLG.DELAUNAY_TOL_DEG).all()

    # The fix adds stream nodes on the stream rather than removing the interior point
    is_stream = pslg.node_codes == 3
    assert is_stream.sum() > (raw.node_codes == 3).sum()
    assert np.allclose(pslg.vertices[is_stream, 0], STREAM_X, atol=1e-6)
    near = np.linalg.norm(pslg.vertices[:, :2] - [STREAM_X + 1.0, Y0 + 150.0], axis=1)
    assert near.min() < 1e-6

    # New stream nodes take part in monotonic descent: z rises strictly upstream (+y)
    s = pslg.vertices[is_stream]
    s = s[np.argsort(s[:, 1])]
    assert (np.diff(s[:, 2]) > 0).all()


def test_outlet_node_below_stream_terminal(tmp_path):
    # The berm puts the terminal and the outlet node ~5 m above the channel. tRIBS
    # FillLakes then fails to drain the terminal into the outlet (see SM_041 hang),
    # so both must be lowered: outlet < terminal < upstream stream node.
    pslg = _mesh(_valley_basin(tmp_path, berm=5.0))
    xyz, codes = pslg.vertices, pslg.node_codes
    outlet = xyz[codes == 2][0]
    stream = xyz[codes == 3]
    stream = stream[np.argsort(stream[:, 1])]      # south (downstream) first
    terminal, upstream = stream[0], stream[1]

    assert outlet[2] < terminal[2] < upstream[2]
    # Both were pulled down off the berm (~105 m) to just below the upstream channel
    assert terminal[2] < 101.0 and outlet[2] < 101.0
    assert (np.diff(stream[:, 2]) > 0).all()


def test_outlet_node_left_alone_when_already_lowest(valley_basin):
    # Without a berm the outlet node samples lower valley floor, so it keeps its DEM value
    pslg = _mesh(valley_basin)
    outlet = pslg.vertices[pslg.node_codes == 2][0]
    assert outlet[2] == pytest.approx(pslg._sample_elevations([outlet[:2]])[0])
