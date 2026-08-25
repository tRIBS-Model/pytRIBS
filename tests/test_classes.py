"""Construction and input-file round-trip tests for the top-level pytRIBS classes."""
import pytest

from pytRIBS.classes import Land, Met, Mesh, Model, Soil


@pytest.fixture
def input_file(tmp_path):
    """A tRIBS .in file written by Model with a handful of options set."""
    m = Model()
    m.options['startdate']['value'] = '06/01/2015/00/00'
    m.options['runtime']['value'] = '240'
    m.options['outfilename']['value'] = 'results/test'
    m.options['soiltablename']['value'] = 'data/model/soil/soils.sdt'
    m.options['soilmapname']['value'] = 'data/model/soil/soil_classes.soi'
    m.options['landtablename']['value'] = 'data/model/land/land.ldt'
    m.options['hydrometstations']['value'] = 'data/model/met/meteor/met.sdf'
    m.options['gaugestations']['value'] = 'data/model/met/precip/precip.sdf'
    m.options['pointfilename']['value'] = 'data/model/mesh/mesh.points'
    path = str(tmp_path / 'test.in')
    m.write_input_file(path)
    return path


def test_model_input_file_round_trip(input_file):
    m = Model(input_file=input_file)
    assert m.options['startdate']['value'] == '06/01/2015/00/00'
    assert m.options['runtime']['value'] == '240'
    assert m.options['outfilename']['value'] == 'results/test'


def test_default_construction():
    # Every top-level class must construct without an input file
    assert Soil().soilmapname['value'] is None
    assert Land().landtablename['value'] is None
    assert Met().hydrometstations['value'] is None
    assert Mesh().pointfilename['value'] is None


def test_soil_from_input_file(input_file):
    s = Soil(input_file=input_file)
    assert s.soiltablename['value'] == 'data/model/soil/soils.sdt'
    assert s.soilmapname['value'] == 'data/model/soil/soil_classes.soi'


def test_land_from_input_file(input_file):
    land = Land(input_file=input_file)
    assert land.landtablename['value'] == 'data/model/land/land.ldt'


def test_met_from_input_file(input_file):
    met = Met(input_file=input_file)
    assert met.hydrometstations['value'] == 'data/model/met/meteor/met.sdf'
    assert met.gaugestations['value'] == 'data/model/met/precip/precip.sdf'


def test_mesh_from_input_file(input_file):
    mesh = Mesh(input_file=input_file)
    assert mesh.pointfilename['value'] == 'data/model/mesh/mesh.points'
