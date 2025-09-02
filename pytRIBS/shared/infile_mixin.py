class Infile:
    """
    Mixin for .in file parameters and definitions shared by both pytRIBS Classes Model & Results.
    """

    @staticmethod
    def create_input_file():
        """
        Creates a dictionary with tRIBS input options assigne to attribute input_options.

        This function loads a dictionary of the necessary variables for a tRIBS input file. And is called upon
        initialization. The dictionary is assigned as instance variable:input_options to the Class Simulation. Note
        the templateVars file will need to be udpated if additional keywords are added to the .in file.

        Each subdictionary has a tags key. With the tag indicating the role of the given option or variable in the model
        simulation.

        Tags:
        time    - parameters related to model simulation times and time steps
        mesh    - options for reading mesh
        flow    - flow routing parameters
        hydro    - options and physical parameters for hydrological components of modle
        spatial   - input raster files and tables for bedrock, groundwater, landuse, and soil properties.
        meterological    -  options for meterological data
        output    - options and paths for model outputs
        forecast - suite of options for forecast mode
        stochastic    - suite of options for stochastic mode
        restart   - options for restart functionality
        parallel  - options for parallel functionality
        """
        input_file = {

            # ==================================================================================================
            # TIME PARAMETERS
            # ==================================================================================================

            "startdate": {"keyword": "STARTDATE:", "describe": "Starting time (MM/DD/YYYY/HH/MM)", "value": None,
                          "tags": ["time"]},
            "runtime": {"keyword": "RUNTIME:", "describe": "Simulation length in hours", "value": None,
                        "tags": ["time"]},
            "rainsearch": {"keyword": "RAINSEARCH:", "describe": "Rainfall search interval (hours)", "value": 24,
                           "tags": ["time"]},
            "timestep": {"keyword": "TIMESTEP:", "describe": "Unsaturated zone computational time step (mins)",
                         "value": 3.75, "tags": ["time"]},
            "gwstep": {"keyword": "GWSTEP:", "describe": "Saturated zone computational time step (mins)",
                       "value": 30.0, "tags": ["time"]},
            "metstep": {"keyword": "METSTEP:", "describe": "Meteorological data time step (mins)", "value": 60.0,
                        "tags": ["time"]},
            "etistep": {"keyword": "ETISTEP:", "describe": "ET, interception and snow time step (hours)", "value": 1,
                        "tags": ["time"]},
            "rainintrvl": {"keyword": "RAININTRVL:", "describe": "Time interval in rainfall input (hours)",
                           "value": 1, "tags": ["time"]},
            "intstormmax": {"keyword": "INTSTORMMAX:", "describe": "Interstorm interval (hours)", "value": 10000,
                            "tags": ["time"]},
            # ==================================================================================================
            # MESH PARAMETERS
            # ==================================================================================================
            "optmeshinput": {"keyword": "OPTMESHINPUT:", "describe": "Mesh input data option\n" + \
                                                                      "1  tMesh data\n" + \
                                                                      "2  Point file\n" + \
                                                                      "3  ArcGrid (random)\n" + \
                                                                      "4  ArcGrid (hex)\n" + \
                                                                      "5  Arc/Info *.net\n" + \
                                                                      "6  Arc/Info *.lin,*.pnt\n" + \
                                                                      "7  Scratch\n" + \
                                                                      "8  Point Triangulator", "value": 8,
                             "tags": ["mesh"]},
            "inputdatafile": {"keyword": "INPUTDATAFILE:", "describe": "tMesh input file base name for Mesh files",
                              "value": None, "tags": ["mesh"]},
            "inputtime": {"keyword": "INPUTTIME:", "describe": "Deprecated, will be removed", "value": None, "tags": ["mesh"]},
            "arcinfofilename": {"keyword": "ARCINFOFILENAME:", "describe": "tMesh input file base name Arc files",
                                "value": None, "tags": ["mesh"]},
            "pointfilename": {"keyword": "POINTFILENAME:", "describe": "tMesh input file name Points files",
                              "value": None, "tags": ["mesh"]},
            # ==================================================================================================
            # ROUTING PARAMETERS
            # ==================================================================================================

            "baseflow": {"keyword": "BASEFLOW:", "describe": "Baseflow discharge (m3/s)", "value": 0.2,
                         "tags": ["flow"]},
            "velocitycoef": {"keyword": "VELOCITYCOEF:", "describe": "Discharge-velocity coefficient", "value": 1.2,
                             "tags": ["flow"]},
            "kinemvelcoef": {"keyword": "KINEMVELCOEF:", "describe": "Kinematic routing velocity coefficient",
                             "value": 3, "tags": ["flow"]},
            "velocityratio": {"keyword": "VELOCITYRATIO:", "describe": "Stream to hillslope velocity coefficient",
                              "value": 60, "tags": ["flow"]},
            "flowexp": {"keyword": "FLOWEXP:", "describe": "Nonlinear discharge coefficient", "value": 0.3,
                        "tags": ["flow"]},
            "channelroughness": {"keyword": "CHANNELROUGHNESS:", "describe": "Uniform channel roughness value",
                                 "value": 0.15, "tags": ["flow"]},
            "channelwidth": {"keyword": "CHANNELWIDTH:", "describe": "Uniform channel width  (meters)", "value": 12,
                             "tags": ["flow"]},
            "channelwidthcoeff": {"keyword": "CHANNELWIDTHCOEFF:",
                                  "describe": "Coefficient in width-area relationship", "value": 2.33,
                                  "tags": ["flow"]},
            "channelwidthexpnt": {"keyword": "CHANNELWIDTHEXPNT:", "describe": "Exponent in width-area relationship",
                                  "value": 0.54, "tags": ["flow"]},
            "channelwidthfile": {"keyword": "CHANNELWIDTHFILE:", "describe": "Filename that contains channel widths",
                                 "value": None, "tags": ["flow"]},
            "widthinterpolation": {"keyword": "WIDTHINTERPOLATION:",
                                   "describe": "Option for interpolating width values", "value": 0, "tags": ["flow"]},

            # ==================================================================================================
            # HYDROLOGIC PARAMETERS
            # ==================================================================================================

            "optevapotrans": {"keyword": "OPTEVAPOTRANS:", "describe": "Option for evapoTranspiration scheme\n" + \
                                                                        "0  Inactive evapotranspiration\n" + \
                                                                        "1  Penman-Monteith method\n" + \
                                                                        "2  Deardorff method\n" + \
                                                                        "3  Priestley-Taylor method\n" + \
                                                                        "4  Pan evaporation measurements", "value": 1,
                              "tags": ["hydro"]},
            "optintercept": {"keyword": "OPTINTERCEPT:", "describe": "Option for interception scheme\n" + \
                                                                      "0  Inactive interception\n" + \
                                                                      "1  Canopy storage method\n" + \
                                                                      "2  Canopy water balance method", "value": 2,
                             "tags": ["hydro"]},
            "optlanduse": {"keyword": "OPTLANDUSE:", "describe": "Option for static or dynamic land cover\n" + \
                                                                  "0  Static land cover maps\n" + \
                                                                  "1  Dynamic updating of land cover maps", "value": 0,
                           "tags": ["hydro"]},
            "optluinterp": {"keyword": "OPTLUINTERP:", "describe": "Option for interpolation of land cover\n" + \
                                                                    "0  Constant (previous) values between land cover\n" + \
                                                                    "1  Linear interpolation between land cover",
                            "value": 0, "tags": ["hydro"]},
            "optsoiltype": {"keyword": "OPTSOILTYPE:", "describe": "Option for soil input type: 0 soil table, 1 soil "
                                                                    "rasters\n See SCGRID, SOILTABLENAME, and SOILMAPNAME",
                            "value": 0, "tags": ["hydro"]},
            "gfluxoption": {"keyword": "GFLUXOPTION:", "describe": "Option for ground heat flux\n" + \
                                                                    "0  Inactive ground heat flux\n" + \
                                                                    "1  Temperature gradient method\n" + \
                                                                    "2  Force_Restore method", "value": 2,
                             "tags": ["hydro"]},
            "optbedrock": {"keyword": "OPTBEDROCK:", "describe": "Option for uniform or variable depth\n"
                                                                  "0 Uniform bedrock depth\n"
                                                                  "1 Spatial grid of bedrock depth\n"
                                                                  "See DEPTHTOBEDROCK and BEDROCKFILE\n", "value": 0,
                           "tags": ["hydro"]},
            "optgroundwater": {"keyword": "OPTGROUNDWATER","describe": "Option for groundwater module, 0 off, 1 on",
                                "value": 1, "tags": ['hydro']},

            "optgwfile": {"keyword": "OPTGWFILE:", "describe": "Option for groundwater initial file\n" + \
                                                                "0 Resample ASCII grid file in GWATERFILE\n" + \
                                                                "1 Read in Voronoi polygon file with GW levels",
                          "value": 0, "tags": ["hydro"]},

            "optrunon": {"keyword": "OPTRUNON:", "describe": "Option for runon in overland flow paths [IN DEVELOPMENT]: 0 off, 1 on", "value": 0,
                         "tags": ["hydro"]},
            "optreservoir": {"keyword": "OPTRESERVOIR:", "describe": 'Option for leve pool routing: 0 off, 1 on',
                             "value": 0, "tags": ["hydro"]},
            ### added
            "respolygonid": {"keyword": "RESPOLYGONID:", "describe": "Path to file of node IDs representing reservoirs",
                             "value":None, "tags": ["hydro"]},
            "resdata": {"keyword": "RESDATA:", "describe": "Path to file of elevation-discharge-storage information for each type of reservoir",
                        "value":None,"tags": ["hydro"]},

            "optsnow": {"keyword": "OPTSNOW:", "describe": "Enable single layer snow module : 0 off, 1 on", "value": 1,
                        "tags": ["hydro"]},
            "minsntemp": {"keyword": "MINSNTEMP:", "describe": "Minimum snow temperature", "value": -50.0,
                          "tags": ["hydro"]},
            "snliqfrac": {"keyword": "SNLIQFRAC:", "describe": "Maximum fraction of liquid water in snowpack",
                          "value": 0.065, "tags": ["hydro"]},
            "depthtobedrock": {"keyword": "DEPTHTOBEDROCK:", "describe": "Uniform depth to bedrock (meters), see OPTBEDROCK",
                               "value": 15, "tags": ["hydro"]},
            "optpercolation": {"keyword": "OPTPERCOLATION:", "describe": "Option to for percolation losses from channels\n"
                               "0 off\n1 on"

                , "value": 0,
                               "tags": ["hydro"]},
            "channelconductivity": {"keyword": "CHANNELCONDUCTIVITY:", "describe": "Conductivity in channel", "value": 0,
                                    "tags": ["hydro"]},
            "transientconductivity": {"keyword": "TRANSIENTCONDUCTIVITY:", "describe": "Conductivity during transient period",
                                      "value": 0, "tags": ["hydro"]},
            "transienttime": {"keyword": "TRANSIENTTIME:", "describe": "Time until transient period ends", "value": 0,
                              "tags": ["hydro"]},
            "channelporosity": {"keyword": "CHANNELPOROSITY:", "describe": "Porosity in channel", "value": 0,
                                "tags": ["hydro"]},
            "chanporeindex": {"keyword": "CHANPOREINDEX:", "describe": "Channel pore index in channel", "value": 0,
                              "tags": ["hydro"]},
            "chanpsib": {"keyword": "CHANPSIB:", "describe": "Matric potential in channel", "value": 0, "tags": ["hydro"]},
            # ==================================================================================================
            # GRID RESAMPLE
            # ==================================================================================================

            "lugrid": {"keyword": "LUGRID:", "describe": "Land cover grid data file (*.gdf)", "value": None,
                       "tags": ["spatial"]},
            "scgrid": {"keyword": "SCGRID:", "describe": "Soil grid data file (*.gdf). Note OPTSOILTYPE must = 1 if "
                                                          "inputing soil grids", "value": None,
                       "tags": ["spatial"]},
            "soiltablename": {"keyword": "SOILTABLENAME:", "describe": "Soil parameter reference table (*.sdt)",
                              "value": None, "tags": ["spatial"]},
            "soilmapname": {"keyword": "SOILMAPNAME:", "describe": "Soil texture ASCII grid (*.soi)", "value": None,
                            "tags": ["spatial"]},
            "landtablename": {"keyword": "LANDTABLENAME:", "describe": "Land use parameter reference table",
                              "value": None, "tags": ["spatial"]},
            "landmapname": {"keyword": "LANDMAPNAME:", "describe": "Land use ASCII grid (*.lan)", "value": None,
                            "tags": ["spatial"]},
            "gwaterfile": {"keyword": "GWATERFILE:", "describe": "Ground water ASCII grid (*iwt)", "value": None,
                           "tags": ["spatial"]},
            "bedrockfile": {"keyword": "BEDROCKFILE:", "describe": "Bedrock depth ASCII grid (*.brd), see OPTBEDROCK", "value": None,
                            "tags": ["spatial"]},
            "demfile": {"keyword": "DEMFILE:", "describe": "DEM ASCII grid for sky and land view factors (*.dem)",
                        "value": None, "tags": ["spatial"]},

            # ==================================================================================================
            # METEROLOGICAL DATA
            # ==================================================================================================
            "rainsource": {"keyword": "RAINSOURCE:", "describe": "Rainfall data source option\n" +
                                                                  "1  Stage III radar\n" +
                                                                  "2  WSI radar\n" +
                                                                  "3  Rain gauges", "value": 3, "tags": ["meterological"]},
            "rainfile": {"keyword": "RAINFILE:", "describe": "Base name of the radar ASCII grid", "value": None,
                         "tags": ["meterological"]},
            "rainextension": {"keyword": "RAINEXTENSION:", "describe": "Extension for the radar ASCII grid",
                              "value": None, "tags": ["meterological"]},
            "metdataoption": {"keyword": "METDATAOPTION:", "describe": "Option for meteorological data\n" + \
                                                                        "0  Inactive meteorological data\n" + \
                                                                        "1  Weather station point data\n" +
                                                                        "2  Gridded meteorological data", "value": 1,
                              "tags": ["meterological"]},
            "hydrometstations": {"keyword": "HYDROMETSTATIONS:",
                                 "describe": "Hydrometeorological station file (*.sdf)", "value": None, "tags": ["meterological"]},
            "hydrometgrid": {"keyword": "HYDROMETGRID:", "describe": "Hydrometeorological grid data file (*.gdf)",
                             "value": None, "tags": ["meterological"]},
            "hydrometconvert": {"keyword": "HYDROMETCONVERT:",
                                "describe": "Hydrometeorological data conversion file (*.mdi)", "value": None,
                                "tags": ["meterological"]},
            "hydrometbasename": {"keyword": "HYDROMETBASENAME:",
                                 "describe": "Hydrometeorological data BASE name (*.mdf)", "value": None,
                                 "tags": ["meterological"]},
            "gaugestations": {"keyword": "GAUGESTATIONS:", "describe": "Rain Gauge station file (*.sdf)",
                              "value": None, "tags": ["meterological"]},
            "gaugeconvert": {"keyword": "GAUGECONVERT:", "describe": "Rain Gauge data conversion file (*.mdi)",
                             "value": None, "tags": ["meterological"]},
            "gaugebasename": {"keyword": "GAUGEBASENAME:", "describe": "Rain Gauge data BASE name (*.mdf)",
                              "value": None, "tags": ["meterological"]},
            "outhydroextension": {"keyword": "OUTHYDROEXTENSION:", "describe": "Extension for hydrograph output",
                                  "value": "mrf", "tags": ["meterological"]},
            "ribshydoutput": {"keyword": "RIBSHYDOUTPUT:", "describe": "Compatibility with RIBS User Interphase,\nDepracted and will be removed.",
                              "value": 0, "tags": ["meterological"]},
            "convertdata": {"keyword": "CONVERTDATA:", "describe": "Option to convert met data format", "value": 0,
                            "tags": ["meterological"]},
            "templapse": {"keyword": "TEMPLAPSE:", "describe": "Temperature lapse rate", "value": -0.0065,
                          "tags": ["meterological"]},
            "preclapse": {"keyword": "PRECLAPSE:", "describe": "Precipitation lapse rate", "value": 0,
                          "tags": ["meterological"]},
            "hillalbopt": {"keyword": "HILLALBOPT:", "describe": "Option for albedo of surrounding hillslopes\n" + \
                                                                  "0  Snow albedo for hillslopes\n" + \
                                                                  "1  Land-cover albedo for hillslopes\n" + \
                                                                  "2  Dynamic albedo for hillslopes", "value": 0,
                           "tags": ["meterological"]},
            "optradshelt": {"keyword": "OPTRADSHELT:", "describe": "Option for local and remote radiation sheltering" +
                                                                    "0  Local controls on shortwave radiation\n" + \
                                                                    "1  Remote controls on diffuse shortwave\n" + \
                                                                    "2  Remote controls on entire shortwave\n" + \
                                                                    "3  No sheltering", "value": 0, "tags": ["meterological"]},
            "tlinke": {"keyword": "TLINKE:", "describe": "Atmospheric turbidity parameter", "value": 2.5,
                       "tags": ["meterological"]},
            # ==================================================================================================
            # OUTPUT DATA
            # ==================================================================================================
            "outfilename": {"keyword": "OUTFILENAME:", "describe": "Base name of the tMesh and variable",
                            "value": None, "tags": ["output"]},
            "outhydrofilename": {"keyword": "OUTHYDROFILENAME:", "describe": "Base name for hydrograph output",
                                 "value": None, "tags": ["output"]},
            "optspatial": {"keyword": "OPTSPATIAL:", "describe": "Enable dynamic spatial output", "value": 0,
                           "tags": ["output"]},
            "optinterhydro": {"keyword": "OPTINTERHYDRO:", "describe": "Enable intermediate hydrograph output",
                              "value": 0, "tags": ["output"]},
            "optheader": {"keyword": "OPTHEADER:", "describe": "Enable headers in output files", "value": 1,
                          "tags": ["output"]},
            "optviz": {"keyword": "OPTVIZ:", "describe": "Option to write binary output files for visualization\n" + \
                                                          "0  Do NOT write binary output files for viz\n" + \
                                                          "1  Write binary output files for viz", "value": 0,
                       "tags": ["output"]},
            "outvizfilename": {"keyword": "OUTVIZFILENAME:", "describe": "Filename for viz binary files",
                               "value": None, "tags": ["output"]},
            "nodeoutputlist": {"keyword": "NODEOUTPUTLIST:",
                               "describe": "Filename with Nodes for Dynamic Output (*.nol)", "value": None,
                               "tags": ["output"]},
            "hydronodelist": {"keyword": "HYDRONODELIST:",
                              "describe": "Filename with Nodes for HydroModel Output (*.nol)", "value": None,
                              "tags": ["output"]},
            "outletnodelist": {"keyword": "OUTLETNODELIST:",
                               "describe": "Filename with Interior Nodes for  Output (*.nol)", "value": None,
                               "tags": ["output"]},
            "opintrvl": {"keyword": "OPINTRVL:", "describe": "Output interval (hours)", "value": 1, "tags": ["output"]},
            "spopintrvl": {"keyword": "SPOPINTRVL:", "describe": "Spatial output interval (hours)", "value": 50000,
                           "tags": ["output"]},
            # ==================================================================================================
            # FORECAST MODE
            # ==================================================================================================
            "forecastmode": {"keyword": "FORECASTMODE:", "describe": "Rainfall Forecasting Mode Option", "value": 0,
                             "tags": ["forecast"]},
            # TODO need to update model mode descriptions
            "forecasttime": {"keyword": "FORECASTTIME:", "describe": "Forecast Time (hours from start)", "value": 0,
                             "tags": ["forecast"]},
            "forecastleadtime": {"keyword": "FORECASTLEADTIME:", "describe": "Forecast Lead Time (hours) ",
                                 "value": 0, "tags": ["forecast"]},
            "forecastlength": {"keyword": "FORECASTLENGTH:", "describe": "Forecast Window Length (hours)", "value": 0,
                               "tags": ["forecast"]},
            "forecastfile": {"keyword": "FORECASTFILE:", "describe": "Base name of the radar QPF grids",
                             "value": None, "tags": ["forecast"]},
            "climatology": {"keyword": "CLIMATOLOGY:", "describe": "Rainfall climatology (mm/hr)", "value": 0,
                            "tags": ["forecast"]},
            "raindistribution": {"keyword": "RAINDISTRIBUTION:", "describe": "Distributed or MAP radar rainfall",
                                 "value": 0, "tags": ["forecast"]},
            # ==================================================================================================
            # STOCHASTIC MODE
            # ==================================================================================================
            "stochasticmode": {"keyword": "STOCHASTICMODE:", "describe": "Stochastic Climate Mode Option", "value": 0,
                               "tags": ["stochastic"]},
            "pmean": {"keyword": "PMEAN:", "describe": "Mean rainfall intensity (mm/hr)	", "value": 0,
                      "tags": ["stochastic"]},
            "stdur": {"keyword": "STDUR:", "describe": "Mean storm duration (hours)", "value": 0,
                      "tags": ["stochastic"]},
            "istdur": {"keyword": "ISTDUR:", "describe": "Mean time interval between storms (hours)", "value": 0,
                       "tags": ["stochastic"]},
            "seed": {"keyword": "SEED:", "describe": "Random seed", "value": 0, "tags": ["stochastic"]},
            "period": {"keyword": "PERIOD:", "describe": "Period of variation (hours)", "value": 0,
                       "tags": ["stochastic"]},
            "maxpmean": {"keyword": "MAXPMEAN:", "describe": "Maximum value of mean rainfall intensity (mm/hr)",
                         "value": 0, "tags": ["stochastic"]},
            "maxstdurmn": {"keyword": "MAXSTDURMN:", "describe": "Maximum value of mean storm duration (hours)",
                           "value": 0, "tags": ["stochastic"]},
            "maxistdurmn": {"keyword": "MAXISTDURMN:", "describe": "Maximum value of mean interstorm period (hours)",
                            "value": 0, "tags": ["stochastic"]},
            "weathertablename": {"keyword": "WEATHERTABLENAME:", "describe": "File with Stochastic Weather Table",
                                 "value": None, "tags": ["stochastic"]},
            # ==================================================================================================
            # RESTART MODE
            # ==================================================================================================
            "restartmode": {"keyword": "RESTARTMODE:", "describe": "Restart Mode Option\n" + \
                                                                    "0 No reading or writing of restart\n" + \
                                                                    "1 Write files (only for initial runs)\n" + \
                                                                    "2 Read file only (to start at some specified time)\n" + \
                                                                    "Read a restart file and continue to write",
                            "value": 0, "tags": ["restart"]},
            "restartintrvl": {"keyword": "RESTARTINTRVL:", "describe": "Time set for restart output (hours)",
                              "value": None, "tags": ["restart"]},
            "restartdir": {"keyword": "RESTARTDIR:", "describe": "Path of directory for restart output",
                           "value": None, "tags": ["restart"]},
            "restartfile": {"keyword": "RESTARTFILE:", "describe": "Actual file to restart a run", "value": None,
                            "tags": ["restart"]},
            # ==================================================================================================
            # PARALLEL MODE
            # ==================================================================================================
            "parallelmode": {"keyword": "PARALLELMODE:", "describe": "Parallel or Serial Mode Option\n" + \
                                                                      "0  Run in serial mode\n" + \
                                                                      "1  Run in parallel mode",
                             "value": 0, "tags": ["parallel"]},
            "graphoption": {"keyword": "GRAPHOPTION:", "describe": "Graph File Type Option\n" + \
                                                                    "0  Default partitioning of the graph\n" + \
                                                                    "1  Reach-based partitioning\n" + \
                                                                    "2  Inlet/outlet-based partitioning", "value": 0,
                            "tags": ["parallel"]},
            "graphfile": {"keyword": "GRAPHFILE:", "describe": "Reach connectivity filename (graph file option 1,2)",
                          "value": None, "tags": ["parallel"]}
        }

        return input_file
