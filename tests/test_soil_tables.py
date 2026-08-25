"""Round-trip tests for the soil reclassification table (*.sdt) readers/writers."""
import os

from pytRIBS.soil.soil import SoilProcessor

SOIL_TYPES = [
    {'ID': '1', 'Ks': '175.3', 'thetaS': '0.35', 'thetaR': '0.05', 'm': '1.64',
     'PsiB': '-14.6', 'f': '0.0005', 'As': '1.0', 'Au': '1.0', 'n': '0.35',
     'ks': '2.5', 'Cs': '1200000.0', 'Texture': 'sand'},
    {'ID': '2', 'Ks': '6.9', 'thetaS': '0.40', 'thetaR': '0.09', 'm': '0.41',
     'PsiB': '-17.3', 'f': '0.0005', 'As': '1.0', 'Au': '1.0', 'n': '0.40',
     'ks': '2.0', 'Cs': '1300000.0', 'Texture': 'loam'},
]

PARAM_KEYS = ['ID', 'Ks', 'thetaS', 'thetaR', 'm', 'PsiB', 'f', 'As', 'Au',
              'n', 'ks', 'Cs']


def test_soil_table_round_trip(tmp_path):
    path = str(tmp_path / 'soils.sdt')
    SoilProcessor.write_soil_table(SOIL_TYPES, path)

    back = SoilProcessor().read_soil_table(file_path=path)

    assert len(back) == len(SOIL_TYPES)
    for written, read in zip(SOIL_TYPES, back):
        for key in PARAM_KEYS:
            assert read[key] == written[key]


def test_soil_table_texture_sidecar_round_trip(tmp_path):
    path = str(tmp_path / 'soils.sdt')
    SoilProcessor.write_soil_table(SOIL_TYPES, path, textures=True)

    sidecar = str(tmp_path / 'soils_textures.csv')
    assert os.path.exists(sidecar), "texture sidecar *_textures.csv not written"

    back = SoilProcessor().read_soil_table(textures=True, file_path=path)
    assert [s['Texture'] for s in back] == ['sand', 'loam']
