# pytRIBS: Version 1.0.0

**pytRIBS** is a pre-to-post processing Python package designed to allow users to set up, simulate, and analyze
TIN-based Real-time Integrated Basin Simulator ([tRIBS](https://github.com/tRIBS-Model/tRIBS)) model runs through a
Python interface. It provides a modular workflow that mirrors the structure of a tRIBS simulation, spanning data
preprocessing, model configuration, execution, and post-processing.

pytRIBS v1.0.0 is the first stable release, aligning the package with the tRIBS v6.0.0 model release. Because it targets
the reworked v6.0.0 input structure, this release is **not backwards compatible** with files produced for earlier tRIBS
versions.

## Design

pytRIBS has a modular design that follows the stages of a tRIBS model simulation. The core of the package is organized
into five preprocessing classes and two simulation classes, each of which draws from a set of shared base classes:

**Preprocessing**
* `Project` — manages the project directory structure and metadata.
* `Soil` — soil data processing and generation of soil input files.
* `Land` — land use / land cover processing and generation of land use input files.
* `Mesh` — TIN mesh generation from watershed and stream network data.
* `Met` — meteorological forcing data download and preparation.

**Simulation**
* `Model` — model setup, input file generation, and simulation execution.
* `Results` — post-processing, analysis, plotting, and calibration aids for simulation output.

## Installation

pytRIBS requires **Python 3.11+**. Install from source with:

```bash
git clone https://github.com/tRIBS-Model/pytRIBS.git
cd pytRIBS
pip install .
```

## Quick Start

```python
from pytRIBS.classes import Project, Model, Results

# Set up a project directory structure
proj = Project(base_dir="my_basin", name="my_basin", epsg=32612)

# Configure and run a model from an existing input file
model = Model(input_file="my_basin.in")

# Analyze the simulation output
results = Results(input_file="my_basin.in")
```

For complete, end-to-end examples, two companion repositories are available:

* [**pytRIBS examples**](https://github.com/tRIBS-Model/pytRIBS-examples) — the reference workflow for learning pytRIBS,
  building a model of an example watershed from scratch. Covers the updated v1.0.0 input structure, the new snow
  parameter files (`.spf`), and the new mesh, spin-up, and plotting tools.
* [**tRIBS Workshop Sandbox**](https://github.com/tRIBS-Model/tRIBS-Workshop-Sandbox) — developed for classes and
  workshops but usable by anyone. This example drives a model with observational data and demonstrates the calibration
  workflows.

## Documentation

* API documentation is available on [Read the Docs](https://pytribs.readthedocs.io/en/latest/).
* For new tRIBS users, see the [tRIBS documentation](https://tribshms.readthedocs.io/en/latest/).

## Release/Version Notes

pytRIBS uses semantic versioning. We record updates of major, minor, and patch versions [here](https://github.com/tRIBS-Model/pytRIBS/blob/main/CHANGELOG.md).

## Contributing

Please open an [issue](https://github.com/tRIBS-Model/pytRIBS/issues) to report bugs or request features. Because much of
the functionality here has had limited testing, users are encouraged to verify package behavior for their own
applications.

## License

pytRIBS is released under the GNU General Public License v2. See [LICENSE](https://github.com/tRIBS-Model/pytRIBS/blob/main/LICENSE) for details.
