<!--- CHANGELOG.md --->
# Changelog
All notable changes to this project are documented in this file.

pytRIBS uses [semantic versioning](https://semver.org/).

## [1.0.0] - Unreleased
pytRIBS v1.0.0 is the first stable release of the package, moving out of beta and aligning pytRIBS with the tRIBS v6.0.0 model release. Because it targets the reworked v6.0.0 input structure, this release is not backwards compatible with files produced for earlier tRIBS versions. This version standardizes how pytRIBS reads and writes the soil, land use, and gridded parameter files to the new v6.0.0 format, adds support for the new snow parameter file (`.spf`), and updates the data-processing workflows that generate these files. **For examples of the updated v6.0.0 input structure, the new snow parameter files (.spf) or application of the new tools described below, see our [example repository](https://github.com/tRIBS-Model/pytRIBS-examples).**

The v1.0.0 changes listed below are abbreviated. For specific details refer to the tRIBS Wiki or the pull request links associated with each change.

### Added
* **Snow Parameter File (`.spf`):** Snow physics constants can now be read from and written to a dedicated `.spf` file, and are loaded automatically when referenced in the main input file. ([#34](https://github.com/tRIBS-Model/pytRIBS/pull/34))
* **Root Zone Depth Parameter:** Added the `RZD_m` parameter to the land use table (`.ldt`). ([#34](https://github.com/tRIBS-Model/pytRIBS/pull/34))
* **Simplified Soil Data Table Creation:** Previously if a user had their own soil ID map the process to generate a soil data table (`.sdt`) was cumbersome. A new function, `create_soil_table_from_map()` was added to simplify this process. ([#35](https://github.com/tRIBS-Model/pytRIBS/pull/35))
* **Full TIN Mesh Generation:** Previously pytRIBS could generate a `*.points` file (x, y, z) that tRIBS read in and triangulated internally. The new method builds the TIN within pytRIBS instead using unconstrained delaunay triangulation with breaklines defining the watershed boundary and streamlines. This method provides users much more control of mesh generation through the breaklines and addiitonal triangle quality options. Manually editing of the mesh to resolve errors in tRRIBS is vastly reduced. ([#39](https://github.com/tRIBS-Model/pytRIBS/pull/39)), ([#45](https://github.com/tRIBS-Model/pytRIBS/pull/45))
* **Station Mesh Generation:** A common tRIBS exercise is to setup a model for single observational station for calibration, getting familiar with the model, etc. Generating the `*.points` for this exercise was a manualy process within GIS. A new workflow has been added to pytRIBS that generates the `*.points` based on a variable number of user inputs. Users must provide coordinates and elevation, they can optionally provide slope and aspect but the preffered method is to provide a DEM which is used to compute slope and aspect internally based on a representative 50m radius for the hillslope scale. ([#39](https://github.com/tRIBS-Model/pytRIBS/pull/39))
* **Spin-up Tool:** Previously spin-up simulation were a manual process that used only the initial groundwater table as the starting condition. Using the reworked restart mechanism in tRIBS v6.0.0 two methods have been added to pytRIBS for generating all required files for running a spin-up simulation with a restart file written at the final timestep. ([#41](https://github.com/tRIBS-Model/pytRIBS/pull/41))
* **Standard Plots:** Add multiple new functions for making standard plots of the time series and spatial output files. The idea being to quickly make a plot, not advanced figure creation. The output of the new functions do provide their matplotlib `Axes` which could be used for further tweaking. Refer to the PR description for details on the multiple plotting functions added. ([#42](https://github.com/tRIBS-Model/pytRIBS/pull/42))
* **Simple Calibration Tools:** Past versions of pytRIBS had methods for computing various preformance metrics now a workflow was developed that organizes the input data into an orderly time-series and computes the metrics. At this time the methods are limited to streamflow and snow water equivalent time-series. ([#43](https://github.com/tRIBS-Model/pytRIBS/pull/43))
* **Forcing from User Data:** Previously pytRIBS didn't have full capabilities to write the full suite of forcing input files if a user had their own data. A new workflow has been added to handle the generation of these files. Note that the workflow does not handle any unit conversions. ([#44](https://github.com/tRIBS-Model/pytRIBS/pull/44))

### Changed & Refactored
* **Standardized File Formats:** Reworked the soil (`.sdt`), land use (`.ldt`), and grid data (`.gdf`) readers and writers to the new single-header, comma-delimited v6.0.0 format. ([#34](https://github.com/tRIBS-Model/pytRIBS/pull/34))
* **Soil Textures:** Soil textures are now written to a `*_textures.csv` sidecar file instead of an extra column in the `.sdt`. ([#34](https://github.com/tRIBS-Model/pytRIBS/pull/34))
* **Static Land Use Grids:** Grid-file path validation now recognizes the new static gridded land use option (`OPTLANDUSE = 2`). ([#34](https://github.com/tRIBS-Model/pytRIBS/pull/34))
* **Solar Position Calculations:** Variables for computing the solar position have been moved from the station data files into the main input file as keywords. ([#35](https://github.com/tRIBS-Model/pytRIBS/pull/35))
* **Forcing Data File Formats:** Reworked the precipitation/meteorological data files (`.sdf` and `.mdf`) readers and writers to the new single-header, comma-delimited v6.0.0 format. ([#35](https://github.com/tRIBS-Model/pytRIBS/pull/35))
* **Output File Formats:** Updated the `.pixel`, `.qout`, `.mrf`, and spatial output map readers for the new v6.0.0 format of a single CSV header line followed by comma-delimited rows. The `.mrf` reader no longer expects a separate units row. ([#40](https://github.com/tRIBS-Model/pytRIBS/pull/40))
* **Node-List Files:** Updated the node-list (`.nol`) reader and writer (`NODEOUTPUTLIST`, `HYDRONODELIST`, `OUTLETNODELIST`) to the new v6.0.0 CSV format: a single `ID` or `X,Y` header line followed by one node per row. `write_node_file` now supports writing coordinate (`X,Y`) lists via the `coords` argument. ([#40](https://github.com/tRIBS-Model/pytRIBS/pull/40))
* **Input File Generation:** The main tRIBS input file generation code has been completely reworked to write a more user-readable file with descriptions of the different options and sectioning. The input file keywords being removed in tRIBS v6.0.0 have also been removed from pytRIBS as well. After this release older input files generated by pytRIBS will not be compatible (input files can be manually edited with a small number of changes to compatible). ([#38](https://github.com/tRIBS-Model/pytRIBS/pull/38))
* **Nodata Elevation in Point Extraction:** Depending on the provided DEM, `extract_points_from_significant_details` would extract nodata values as elevation values. this issue disrupted the wavlet decomposition. ([#45](https://github.com/tRIBS-Model/pytRIBS/pull/45))

### Removed
* **Legacy Land Use Parameters:** Removed the Gray (1970) interception parameters (`a`, `b1`) from the land use table. ([#34](https://github.com/tRIBS-Model/pytRIBS/pull/34))
* **Parallel Output Merging:** tRIBS v6.0.0 now writes the same consolidated spatial, Voronoi, and integrated output files in parallel mode as in serial mode. The per-processor-rank merging logic (`merge_parallel_voi`, the `parallelmode` branches in `get_spatial_files` and `get_invariant_properties`) has been removed. ([#40](https://github.com/tRIBS-Model/pytRIBS/pull/40))

## [0.7.3] - 2026-06-17
This release introduces the support for Python 3.13. The code was tested using a full example model setup but not every pytRIBS function was tested. Please open an issue if you come across other problems.

### Changed
* Remove pynldas2 dependency and add additional dependency imports that were originally included with the pynldas2 package.
* Remove import of pytz package and replace with python's built in zoneinfo tool.
* Remove PyVista dependencies and pytRIBS functions that used PyVista. PyVista is a very heavy package and the pytRIBS code that used it were no longer functioning or regularly used.
* Small chanhes to multiple workflows to handle the shift to Numpy 2.X.

## [0.7.2] - 2026-04-02
This release fixes an issue users on certain versions of Python and introduces GitHub Actions.

### Added
* Added GitHub Actions workflow to verify installation across Python 3.10, 3.11, and 3.12 on ubuntu-latest.

### Changed
* Relaxed `earthaccess` requirement from `~=0.15.1` to `>=0.14.0` to resolve a dependency conflict for users on Python 3.10.

## [0.7.1] - 2026-02-03
This release is a small update to add a missing dependency required for downloading and processing the NLDAS-2 elevation raster.

## [0.7.0] - 2026-02-03
This release introduces a set of relatively small changes that fix existing points of confusion or bugs in the code. Additionally, updates to the meteorological workflow to handle changes to the NASA API for downloading NLDAS-2 data.

### Added
* Added new optional input to the run_soil_workflow for downloading POLARIS gridded soil data rather than the ISRIC dataset. Can be controlled with the `source` argument but defaults to ISRIC if not specified. This dataset follows the same general workflow but does not require applying ROSETTA3 like with the ISRIC data. ([#26](https://github.com/tRIBS-Model/pytRIBS/pull/26))

### Changed / Improved
* **Spatial Outputs** ([#28](https://github.com/tRIBS-Model/pytRIBS/pull/28))
    * Addressed limitation of pytRIBS workflow only able to process tRIBS spatial outputs if the model was ran in parallel mode.
    * Renamed merge_parallel_spatial_files to get_spatial_files to reflect its expanded capability.
    * The workflow will now automatically detect from the input file if model was ran in serial or parallel mode for processing the outputs.
* **NLDAS-2 Data Download** ([#27](https://github.com/tRIBS-Model/pytRIBS/pull/29))
    * Refactored `get_nldas_point` to account for changes to NASA API.
    * Removed use of `pynldas2` dependency and added new `earthaccess` dependency that handles the API token for accessing NLDAS-2 data. Note that an earthdata account is now required to download the data.
* **Windspeed Correction** ([#29](https://github.com/tRIBS-Model/pytRIBS/pull/29))
    * Updated code related to converting 10m windspeeds from NLDAS-2 data to 2m height required by tRIBS.
    * All values for the parameters in the conversion now follow the FAO-56 / ASCE standard constants for a standard reference surface of short grass.
* **Hydraulic Conductivity Decay**
    * Updated the method for calculating the hydraulic conductivity decay coefficient to better represent its purpose as the decay rate of the surface soil
* **Technical Cleanup**
    * Loosen package dependencies in `pyproject.toml` to resolve version conflicts.
    * Remove redundant code and improved class initialization in `met.py` to make the code more effective as a standalone workflow.

## [0.6.0] - 2025-11-20
* Fixed bug in reading landuse table (can only use for model or land class though).
* Added optional input to write_ascii() that allows user to specify number of decimal places in output raster.
* Added new function, grid_geodataframe(), in the shared class that is called from the results object. The tool ingests a GDF containing a the voronoi polygon geometry with a spatial output attached and rasterizes that into a data dictionary for file writing.
* Updated generate_meshbuild_input_file() to handle additonal input options for mesh data in newest version of MeshBuilder software.
* Restored public API for Land and Soil helper methods.
* Updated run_docker.py to use the latest branch of tRIBS and removed hardcoding of "OPTLUINTERP" in the input file.
* Fixed a unit conversion erro in how tRIBS soil parameters are calculated from the ROSETTA3 outputs.
* Fixed a bug in convert_to_datetime function that incorrectly reads the starting date from the tRIBS input file.
* Added the function write_geotiff that follows the same functionality as the existing write_ascii function.
* Modified the get_soil_grids function so that the ISRIC soil data is download in its native WGS84 CRS then is reprojected locally. Changed due to recent update to in ISRIC api.

## [0.5.0] - 2024-07-13
* Added in unsupervised classification function for NAIP image and Tree hieght rasters in Land Class
* Finalized Mesh Class, with dependence on a Preprocessing Class (DEM and GIS analysis) and MeshGeneration Class
* Model class can be initialized with combination of Met, Soil, Land, and Mesh classes as well as an input file
* Soil workflow update: input is now shapely polygon, not geopandas geodataframe
* Added in function to find centroid of watershed
* Updated docker workflows for both tRIBS and MeshBuilder
* Added in build and source code for read the docs--needs fine-tuning

## [0.4.0] - 2024-07-11
* added in functionality for met class, can now download and subset NLDAS-2 data with watershed shapefile
* changed key_word in Model.options dictionary to keyword
* Updated Met Class including methods to download and merge NLDAS-2 data.
* Changed waterbalance clacs to use ThetaS instead of porosity following tRIBS
* Converted geo to meta, and added Meta class.
* Added new function in read.py to read in *_Outlet.qout files

## [0.3.0] - 2024-05-03
* Removed tmodel/tresults, replaced with classes
* added new classes Soil, Mesh, Met, Land
* renamed mixins folder to shared
* created results/visualize.py
* created soil/soil.py --moved soil related content from preprocess to here.
* updated create_soil_map to return a soil table in .sdt format.
* updated read/write soil tables to include options for including texture.

## [0.2.0] - 2024-04-25
This minor update includes:
* updates to the infile_mixin, with updates for model documentation
* addition of Paul Tol's colormaps (https://personal.sron.nl/~pault/)
* In shared mixin:
  * added processor # to the attribute voronoi
  * added plot_mesh()
  * fixed other syntax bugs
* model.inout.py
  * added read added write_point_file()
  * fixed syntax bugs in several functions
* Fixed several bugs in preporcess.py and waterbalance.py
* Added create_animation() to Results()

## Return to [README](README.md)
