This pull request adds a test suite and continuous-integration coverage to pytRIBS. Until now the GitHub Actions workflow only installed the package and imported it. Multiple failure modes are now covered by tests, in two tiers: fast unit tests that run on every pull request, and scheduled live-service tests that run weekly against the real external APIs.

### Unit tests (run on every PR)

* **`test_rosetta_soil_params.py`:** direct regression guard. Feeds four textbook textures (sand, loam, clay, silt loam) through `_rosetta_to_tribs_params` and asserts (1) literature-informed plausibility windows on Ks per texture, physical bounds on theta_r/theta_s/psib/m, and correct texture ordering and (2) "golden values" frozen from `rosetta-soil` 0.3.x at a loose ±25% tolerance to flag subtler prediction drift. Also verifies the ROSETTA model-code selection.
* **`test_soil_map.py`:** USDA texture classification through `create_soil_map`, covering both the ISRIC (g/kg) and SOLUS (% mass) input conventions and the sequential class renumbering.
* **`test_ks_decay.py`:** generates synthetic Ks-with-depth grids from a known decay parameter and asserts `compute_ks_decay` recovers it, floors uniform profiles at `min_f`, and propagates nodata.
* **`test_met_conversions.py`:** hand-checked "golden values" for the NLDAS unit conversions (Pa to hPa, K to degC, specific humidity to RH, 10 m to 2 m wind log-profile scaling), the GMT timestamp shift, `.mdf`/`.sdf` round trips, and the observation-station validation errors.
* **`test_soil_tables.py` / `test_ascii_io.py`:** round trips for the `.sdt` soil table and the ESRI ASCII raster writer/reader, including NaN-to-nodata replacement and cell-size preservation.
* **`test_evaluate_metrics.py`:** known-answer checks for NSE, KGE, RMSE, and percent bias.
* **`test_classes.py`:** constructs every top-level class bare and from a written `.in` file, locking in the constructor fix and the input-file write/read round trip.

### Live-service canaries (weekly, `pytest -m network`)

Small real requests with plausibility gates on the responses, so an upstream API change is flagged by a failed-workflow email instead of by a broken user run:

* ISRIC SoilGrids WCS, SOLUS100 COG windowed reads, and POLARIS tile downloads over a ~2×2 km test domain, each asserting a readable, non-empty grid with in-range values.
* NASA Giovanni NLDAS point timeseries, asserting all seven forcing variables return with physically plausible values. Because `get_nldas_point` swallows per-variable fetch errors, a missing column is the failure signal.

### Continuous integration

* **`install_and_test.yml`:** now installs `.[test]` and runs `pytest` (network tests excluded by marker) after the import smoke test, on Python 3.11 and 3.13. Because dependencies are unpinned, every PR run also exercises the newest dependency releases.
* **`canary.yml` (new):** scheduled Mondays 12:00 UTC plus manual dispatch. Installs the latest release of every dependency and runs the full unit suite, so a breaking dependency release is caught within a week even with no pushes. The Giovanni test authenticates via the `EARTHDATA_USERNAME`/`EARTHDATA_PASSWORD` repository secrets.

### Technical changes

* **`tests/`** (new): nine test modules plus a `conftest.py` fixture that writes small georeferenced ESRI ASCII grids for the raster-based tests.
* **`pytRIBS/classes.py`:** fixed the broken `input_file` path in the `Soil`, `Land`, `Met`, and `Mesh` constructors described above; each now also carries a populated `options` dictionary, consistent with `Results`.
* **`pyproject.toml`:** added a `test` extra (`pytest>=8.0`) and pytest configuration — `testpaths`, the `network` marker, and `addopts = "-m 'not network'"` so the default `pytest` invocation never touches the network (an explicit `-m network` overrides it).
* **`.github/workflows/install_and_test.yml`:** unit test step added.
* **`.github/workflows/canary.yml`:** new scheduled canary workflow.

### Running the tests

```bash
pip install -e ".[test]"

pytest              # fast unit suite (network canaries excluded)
pytest -m network   # live-service canaries (Giovanni needs Earthdata credentials)
```

