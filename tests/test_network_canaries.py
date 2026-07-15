"""Live-service canary tests.

These make small, real requests against the external services pytRIBS depends
on, so that an API change is caught by the scheduled CI run. They are excluded
from the default test run; run them with:  pytest -m network
"""
import os

import numpy as np
import pytest
import rasterio

from pytRIBS.classes import Soil
from pytRIBS.met.met import MetProcessor

pytestmark = pytest.mark.network

# ~2 x 2 km box in northern Arizona (UTM zone 12N); all four services cover it.
BBOX_UTM12 = [400000.0, 3895000.0, 402000.0, 3897000.0]
EPSG = 32612


def _utm_soil():
    soil = Soil()
    soil.meta['EPSG'] = EPSG
    return soil


def _read_valid_values(path):
    with rasterio.open(path) as src:
        data = src.read(1).astype(float)
        nodata = src.nodata
    if nodata is not None:
        data = data[data != nodata]
    return data[np.isfinite(data)]


def test_isric_soilgrids_wcs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    files = _utm_soil().get_soil_grids(
        BBOX_UTM12, depths=['0-5cm'], soil_vars=['sand'], stats=['mean'])
    assert files == ['sand_0-5cm_mean.tif']

    values = _read_valid_values(tmp_path / 'sg250' / files[0])
    assert values.size > 0, "ISRIC returned an empty/all-nodata grid"
    # ISRIC sand is g/kg: 0-1000
    assert np.all(values >= 0) and np.all(values <= 1000)


def test_solus_cog_reads(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    files = _utm_soil().get_solus_grids(
        BBOX_UTM12, depths=['0-5cm'], variables=['sandtotal'])
    assert files == ['sandtotal_0-5cm_mean.tif']

    path = tmp_path / 'solus' / files[0]
    assert path.exists(), "SOLUS download produced no file"
    values = _read_valid_values(path)
    assert values.size > 0, "SOLUS returned an empty/all-nodata grid"


def test_polaris_tiles(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    files = _utm_soil().get_polaris_grids(
        BBOX_UTM12, depths=['0-5cm'], variables=['sand'], stats=['mean'])
    assert files == ['sand_0-5cm_mean.tif']

    path = tmp_path / 'polaris' / files[0]
    assert path.exists(), "POLARIS download produced no file"
    values = _read_valid_values(path)
    assert values.size > 0, "POLARIS returned an empty/all-nodata grid"
    # POLARIS sand is percent
    assert np.all(values >= 0) and np.all(values <= 100)


def _has_earthdata_credentials():
    if os.environ.get('EARTHDATA_USERNAME') and os.environ.get('EARTHDATA_PASSWORD'):
        return True
    netrc_path = os.path.expanduser('~/.netrc')
    if os.path.exists(netrc_path):
        with open(netrc_path) as f:
            return 'urs.earthdata.nasa.gov' in f.read()
    return False


@pytest.mark.skipif(not _has_earthdata_credentials(),
                    reason="no Earthdata Login credentials (env or .netrc)")
def test_giovanni_nldas_timeseries():
    df = MetProcessor().get_nldas_point(
        centroids=[(-112.10, 35.25)], begin='2015-06-01', end='2015-06-02',
        epsg=4326)

    # get_nldas_point swallows per-variable fetch errors, so a missing column
    # is the failure signal for a broken variable endpoint.
    expected = {'prcp', 'temp', 'humidity', 'psurf', 'wind_u', 'wind_v', 'rsds'}
    assert expected.issubset(df.columns), (
        f"Giovanni returned only {sorted(set(df.columns))}")
    assert len(df) >= 24, "expected at least a day of hourly NLDAS records"

    # Plausibility gates on the raw NLDAS units
    assert df['temp'].between(230, 340).all()          # K
    assert df['psurf'].between(50000, 110000).all()    # Pa
    assert (df['prcp'] >= 0).all()
    assert df['humidity'].between(0, 0.05).all()       # kg/kg
