"""Known-answer tests for the ROSETTA -> tRIBS soil parameter workflow.

These exist to catch silent behavior changes in the rosetta-soil dependency.
Two layers of defense here:

1. Physical/literature range checks (the hard gate): a unit-convention change
   in either direction pushes Ksat orders of magnitude out of these windows.
2. Golden values frozen from rosetta-soil 0.3.x (July 2026) with a loose
   tolerance, to flag subtler drift in the predictions themselves.
"""
import numpy as np
import pytest

from pytRIBS.soil.soil import SoilProcessor

# Rows: sand %, silt %, clay %, bulk density g/cm3
TEXTURES = {
    'sand':      [92.0, 5.0, 3.0, 1.60],
    'loam':      [40.0, 40.0, 20.0, 1.40],
    'clay':      [20.0, 20.0, 60.0, 1.30],
    'silt_loam': [20.0, 65.0, 15.0, 1.35],
}
IDX = {name: divmod(i, 2) for i, name in enumerate(TEXTURES)}

# (theta_r, theta_s, Ks mm/hr, psib mm, m) per texture, frozen from
# rosetta-soil 0.3.x. Loose rel tolerance: catches convention changes and
# retrained-model drift without failing on numerical noise.
GOLDEN = {
    'sand':      (0.0503, 0.3518, 175.30, -145.93, 1.6351),
    'loam':      (0.0858, 0.4036, 6.90, -172.68, 0.4148),
    'clay':      (0.1374, 0.5088, 7.78, -63.61, 0.2623),
    'silt_loam': (0.0775, 0.4159, 11.16, -416.17, 0.5263),
}

# Literature-informed plausibility windows for Ks (mm/hr). Deliberately
# generous: they should only fail on order-of-magnitude breakage.
KS_WINDOWS = {
    'sand':      (50.0, 700.0),
    'loam':      (1.0, 40.0),
    'clay':      (0.2, 40.0),
    'silt_loam': (1.0, 60.0),
}


@pytest.fixture(scope='module')
def params():
    pixel_data = np.array(list(TEXTURES.values()))
    theta_r, theta_s, ks, psib, m = SoilProcessor._rosetta_to_tribs_params(
        pixel_data, (2, 2))
    return {'theta_r': theta_r, 'theta_s': theta_s, 'ks': ks,
            'psib': psib, 'm': m}


def test_model_code_is_ssc_bd(params):
    # 4-input rows (sand/silt/clay + bulk density) must select ROSETTA model 3
    assert np.all(params['theta_r'][2] == 3)
    assert np.all(params['theta_s'][2] == 3)
    assert np.all(params['ks'][2] == 3)


def test_moisture_contents_physical(params):
    theta_r, theta_s = params['theta_r'][0], params['theta_s'][0]
    assert np.all(theta_r > 0.0) and np.all(theta_r < 0.25)
    assert np.all(theta_s > 0.25) and np.all(theta_s < 0.65)
    assert np.all(theta_r < theta_s)
    # Clay holds more water at saturation than sand
    assert theta_s[IDX['clay']] > theta_s[IDX['sand']]


def test_ks_within_literature_windows(params):
    ks = params['ks'][0]
    assert np.all(np.isfinite(ks))
    for name, (lo, hi) in KS_WINDOWS.items():
        val = ks[IDX[name]]
        assert lo < val < hi, (
            f"{name} Ks = {val:.4g} mm/hr outside plausible window ({lo}, {hi}). "
            f"A unit-convention change in rosetta-soil is the usual suspect.")
    # Sand must drain much faster than fine-textured soils
    assert ks[IDX['sand']] > 5 * ks[IDX['loam']]
    assert ks[IDX['sand']] > 5 * ks[IDX['clay']]


def test_psib_negative_and_plausible(params):
    psib = params['psib'][0]
    assert np.all(psib < 0), "psib must be negative (tRIBS convention)"
    assert np.all(psib > -2000) and np.all(psib < -20), (
        f"air-entry pressures {psib.ravel()} mm outside plausible +/- window")


def test_pore_size_index_plausible(params):
    m = params['m'][0]
    assert np.all(m > 0.1) and np.all(m < 3.0)
    # Coarse soils have a much broader pore-size distribution index
    assert m[IDX['sand']] > m[IDX['loam']] > 0
    assert m[IDX['sand']] > m[IDX['clay']] > 0


def test_stdev_layers_positive(params):
    for key in ('theta_r', 'theta_s', 'ks'):
        stdev = params[key][1]
        assert np.all(np.isfinite(stdev))
        assert np.all(stdev > 0)


def test_golden_values(params):
    for name, (g_tr, g_ts, g_ks, g_psib, g_m) in GOLDEN.items():
        i = IDX[name]
        assert params['theta_r'][0][i] == pytest.approx(g_tr, rel=0.25)
        assert params['theta_s'][0][i] == pytest.approx(g_ts, rel=0.25)
        assert params['ks'][0][i] == pytest.approx(g_ks, rel=0.25)
        assert params['psib'][0][i] == pytest.approx(g_psib, rel=0.25)
        assert params['m'][0][i] == pytest.approx(g_m, rel=0.25)


def test_six_input_rows_use_full_model():
    # Adding TH33/TH1500 water-retention points must select ROSETTA model 5
    pixel_data = np.array([
        [40.0, 40.0, 20.0, 1.40, 0.27, 0.12],   # loam
        [92.0, 5.0, 3.0, 1.60, 0.10, 0.04],     # sand
    ])
    theta_r, theta_s, ks, psib, m = SoilProcessor._rosetta_to_tribs_params(
        pixel_data, (1, 2))
    assert np.all(theta_r[2] == 5)
    # Same plausibility gates as the 4-input model
    assert np.all(theta_r[0] < theta_s[0])
    assert np.all(ks[0] > 0.2) and np.all(ks[0] < 700)
    assert np.all(psib[0] < -20) and np.all(psib[0] > -2000)
    assert np.all(m[0] > 0.1) and np.all(m[0] < 3.0)
