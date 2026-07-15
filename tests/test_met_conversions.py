"""Tests for the NLDAS -> tRIBS met conversions and station file round trips.

The unit conversions (Pa->hPa, K->degC, specific humidity->RH, 10 m->2 m wind
scaling) live in convert_and_write_nldas_timeseries, upstream of the shared
writers. Golden values below were frozen from a hand-checked run (July 2026);
they guard both the conversion formulas and the pandas plumbing around them.
"""
import numpy as np
import pandas as pd
import pytest

from pytRIBS.met.met import MetProcessor
from pytRIBS.shared.inout import InOut

# One synthetic NLDAS station: constant, hand-checkable forcing values.
NLDAS_INPUT = {
    'psurf': 100000.0,   # Pa      -> PA 1000.00 hPa
    'wind_u': 3.0,       # m/s     -> |U| = 5 m/s at 10 m
    'wind_v': 4.0,       #            scaled to 3.739 m/s at 2 m
    'temp': 293.15,      # K       -> TA 20.00 degC
    'humidity': 0.010,   # kg/kg   -> RH 69.66 %
    'rsds': 500.0,       # W/m2    -> IS unchanged
    'prcp': 2.5,         # mm/hr   -> R unchanged
}
STATION_COORDS = [[-112.0, 400000.0, 33.5, 3900000.0, 1500.0]]
GMT = -7


@pytest.fixture
def written_stations(tmp_path):
    idx = pd.date_range('2015-06-01 00:00', periods=24, freq='h', name='time')
    df = pd.DataFrame(NLDAS_INPUT, index=idx)

    met_sdf = str(tmp_path / 'met.sdf')
    precip_sdf = str(tmp_path / 'precip.sdf')
    MetProcessor().convert_and_write_nldas_timeseries(
        [df], STATION_COORDS, gmt=GMT, met_path=met_sdf, precip_path=precip_sdf)
    return met_sdf, precip_sdf


def test_nldas_met_conversions(written_stations):
    met_sdf, _ = written_stations
    stations = InOut.read_sdf(met_sdf)
    assert len(stations) == 1
    assert stations[0]['x'] == pytest.approx(400000.0)
    assert stations[0]['y'] == pytest.approx(3900000.0)
    assert stations[0]['elevation'] == pytest.approx(1500.0)
    assert stations[0]['file_path'].endswith('met_NLDAS_2015-2015.mdf')

    mdf = InOut.read_met_station(stations[0]['file_path'])
    row = mdf.iloc[0]
    assert row['PA'] == pytest.approx(1000.00, abs=0.01)   # Pa -> hPa
    assert row['TA'] == pytest.approx(20.00, abs=0.01)     # K -> degC
    assert row['RH'] == pytest.approx(69.66, abs=0.5)      # q -> RH
    assert row['US'] == pytest.approx(3.739, abs=0.01)     # 10 m -> 2 m wind
    assert row['IS'] == pytest.approx(500.0)               # passthrough
    assert row['XC'] == pytest.approx(9999.99)             # fill value
    assert row['TS'] == pytest.approx(9999.99)             # fill value
    # GMT offset shifts the first UTC midnight timestamp to local time
    assert row['date'] == pd.Timestamp('2015-05-31 17:00:00')


def test_nldas_precip_conversions(written_stations):
    _, precip_sdf = written_stations
    stations = InOut.read_sdf(precip_sdf)
    assert len(stations) == 1
    assert stations[0]['file_path'].endswith('precip_NLDAS_2015-2015.mdf')

    pdf = InOut.read_precip_station(stations[0]['file_path'])
    assert np.allclose(pdf['R'], 2.5)
    assert pdf['date'].iloc[0] == pd.Timestamp('2015-05-31 17:00:00')


def test_met_station_round_trip(tmp_path):
    df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=5, freq='h'),
        'PA': 950.0, 'RH': 50.0, 'US': 2.0, 'TA': 25.0, 'IS': 600.0,
    })
    path = str(tmp_path / 'station.mdf')
    InOut.write_met_station(df.copy(), path)

    back = InOut.read_met_station(path)
    assert len(back) == 5
    for col in ('PA', 'RH', 'US', 'TA', 'IS'):
        assert np.allclose(back[col], df[col].iloc[0])
    # XC/TS auto-filled when absent
    assert np.allclose(back['XC'], 9999.99)
    assert np.allclose(back['TS'], 9999.99)
    assert back['date'].iloc[-1] == df['date'].iloc[-1]


def test_precip_station_round_trip(tmp_path):
    df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=5, freq='h'),
        'R': [0.0, 1.2, 3.4, 0.0, 0.5],
    })
    path = str(tmp_path / 'precip.mdf')
    InOut.write_precip_station(df.copy(), path)

    back = InOut.read_precip_station(path)
    assert back['R'].tolist() == df['R'].tolist()
    assert back['date'].iloc[0] == df['date'].iloc[0]


def test_observation_validation_missing_column():
    df = pd.DataFrame({'date': pd.date_range('2020-01-01', periods=3, freq='h'),
                       'PA': 950.0, 'RH': 50.0, 'US': 2.0, 'TA': 25.0})  # no IS
    stations = [{'name': 'stn1', 'x': 0.0, 'y': 0.0, 'elevation': 100.0, 'data': df}]
    with pytest.raises(ValueError, match="IS"):
        MetProcessor._validate_observation_stations(
            stations, MetProcessor.MET_OBS_COLUMNS, 'met')


def test_observation_validation_duplicate_names():
    df = pd.DataFrame({'date': pd.date_range('2020-01-01', periods=3, freq='h'),
                       'R': 1.0})
    station = {'name': 'stn1', 'x': 0.0, 'y': 0.0, 'elevation': 100.0, 'data': df}
    with pytest.raises(ValueError, match="unique"):
        MetProcessor._validate_observation_stations(
            [station, dict(station)], MetProcessor.PRECIP_OBS_COLUMNS, 'precip')
