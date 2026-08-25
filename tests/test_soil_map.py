"""Tests for create_soil_map: USDA texture classification from sand/clay grids."""
from pytRIBS.shared.inout import InOut
from pytRIBS.soil.soil import SoilProcessor


def test_create_soil_map_from_isric_grids(make_ascii_raster, tmp_path):
    # ISRIC SoilGrids delivers sand/clay in g/kg; create_soil_map converts to %.
    # Pixels: (0,0) 92% sand / 3% clay -> sand; (0,1) 40/20 -> loam;
    # (1,0) and (1,1) 20/60 -> clay.
    sand = [[920.0, 400.0], [200.0, 200.0]]
    clay = [[30.0, 200.0], [600.0, 600.0]]
    grid_input = [
        {'type': 'sand', 'path': make_ascii_raster('sand.asc', sand)},
        {'type': 'clay', 'path': make_ascii_raster('clay.asc', clay)},
    ]
    out = str(tmp_path / 'soil_classes.soi')

    soil_list = SoilProcessor().create_soil_map(grid_input, output=out)

    # Classes present are renumbered sequentially from 1 in USDA order
    assert [d['Texture'] for d in soil_list] == ['sand', 'loam', 'clay']
    assert [d['ID'] for d in soil_list] == [1, 2, 3]

    raster = InOut.read_ascii(out)
    assert raster['data'].tolist() == [[1, 2], [3, 3]]


def test_create_soil_map_from_solus_grids(make_ascii_raster, tmp_path):
    # SOLUS delivers sandtotal/claytotal already in % mass - no unit scaling.
    sand = [[92.0, 40.0], [20.0, 20.0]]
    clay = [[3.0, 20.0], [60.0, 60.0]]
    grid_input = [
        {'type': 'sandtotal', 'path': make_ascii_raster('sandtotal.asc', sand)},
        {'type': 'claytotal', 'path': make_ascii_raster('claytotal.asc', clay)},
    ]
    out = str(tmp_path / 'soil_classes_solus.soi')

    soil_list = SoilProcessor().create_soil_map(grid_input, output=out)

    assert [d['Texture'] for d in soil_list] == ['sand', 'loam', 'clay']
    raster = InOut.read_ascii(out)
    assert raster['data'].tolist() == [[1, 2], [3, 3]]
