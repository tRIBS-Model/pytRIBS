# pytRIBS
A pre-to-post processing python package designed to allow users to setup, simulate, and analyze TIN-based Real-time Integrated Basin Simulator (tRIBS) model runs through a python interface.
Note this packages is currently under development and is subject to further changes. Additionally, much of the functionality here has had limited testing, consequently responsibility is on the user to verify package functionality. 

## Documentation
pytRIBS documentation is available [here.](https://pytribs.readthedocs.io/en/latest/)

## Examples
A full tRIBS model setup, simulation, and analysis is provided [here.](https://zenodo.org/records/13988020)

## Known Issues
* As of October 2025 the package, pynldas2 for downloading NLDAS2 timeseries is no longer functional. This breaks the meteorological workflow in pytRIBS for time being until the package is updated to work with the new API.

## Release/Version Notes
PytRIBS uses semantic versioning. Currently, we are in the initial development phase--anything MAY change at any time and
this package SHOULD NOT be considered stable.

## Version 0.7.0 (02/03/2026)
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
      
### Verison 0.6.0 (11/20/2025)
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
### Verison 0.5.0 (07/13/2024)
* Added in unsupervised classification function for NAIP image and Tree hieght rasters in Land Class
* Finalized Mesh Class, with dependence on a Preprocessing Class (DEM and GIS analysis) and MeshGeneration Class
* Model class can be initialized with combination of Met, Soil, Land, and Mesh classes as well as an input file
* Soil workflow update: input is now shapely polygon, not geopandas geodataframe
* Added in function to find centroid of watershed
* Updated docker workflows for both tRIBS and MeshBuilder
* Added in build and source code for read the docs--needs fine-tuning
### Version 0.4.0 (07/11/2024)
* added in functionality for met class, can now download and subset NLDAS-2 data with watershed shapefile
* changed key_word in Model.options dictionary to keyword
* Updated Met Class including methods to download and merge NLDAS-2 data.
* Changed waterbalance clacs to use ThetaS instead of porosity following tRIBS
* Converted geo to meta, and added Meta class.
* Added new function in read.py to read in *_Outlet.qout files
### Version 0.3.0 (5/03/2024)
* Removed tmodel/tresults, replaced with classes
* added new classes Soil, Mesh, Met, Land
* renamed mixins folder to shared
* created results/visualize.py
* created soil/soil.py --moved soil related content from preprocess to here.
* updated create_soil_map to return a soil table in .sdt format.
* updated read/write soil tables to include options for including texture.
### Version 0.2.0 (4/25/2024)
This minor update includes:
* updates to the infile_mixin, with updates for 
model documentation
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
