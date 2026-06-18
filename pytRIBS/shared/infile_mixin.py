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
        restart   - options for restart functionality
        parallel  - options for parallel functionality

        Section/subsection fields organize how write_input_file groups and formats output.
        Entries without a section field are written at the end under "Additional Options".
        """
        input_file = {

            # ==================================================================================================
            # TIME PARAMETERS
            # ==================================================================================================

            "startdate": {"keyword": "STARTDATE:", "describe": "Starting time (MM/DD/YYYY/HH/MM)", "value": None,
                          "tags": ["time"], "section": 1, "subsection": "Time Variables"},
            "runtime": {"keyword": "RUNTIME:", "describe": "Simulation length in hours", "value": None,
                        "tags": ["time"], "section": 1, "subsection": "Time Variables"},
            "timestep": {"keyword": "TIMESTEP:", "describe": "Unsaturated zone computational time step (mins)",
                         "value": 3.75, "tags": ["time"], "section": 1, "subsection": "Time Variables"},
            "gwstep": {"keyword": "GWSTEP:", "describe": "Saturated zone computational time step (mins)",
                       "value": 30.0, "tags": ["time"], "section": 1, "subsection": "Time Variables"},
            "metstep": {"keyword": "METSTEP:", "describe": "Meteorological data time step (mins)", "value": 60.0,
                        "tags": ["time"], "section": 1, "subsection": "Time Variables"},
            "rainintrvl": {"keyword": "RAININTRVL:", "describe": "Time interval in rainfall input (hours)",
                           "value": 1, "tags": ["time"], "section": 1, "subsection": "Time Variables"},
            "opintrvl": {"keyword": "OPINTRVL:", "describe": "Output interval (hours)", "value": 1,
                         "tags": ["output"], "section": 1, "subsection": "Time Variables"},
            "spopintrvl": {"keyword": "SPOPINTRVL:", "describe": "Spatial output interval (hours)", "value": 8760,
                           "tags": ["output"], "section": 1, "subsection": "Time Variables"},
            "intstormmax": {"keyword": "INTSTORMMAX:", "describe": "Interstorm interval (hours)", "value": 50,
                            "tags": ["time"], "section": 1, "subsection": "Time Variables"},
            "rainsearch": {"keyword": "RAINSEARCH:", "describe": "Rainfall search interval (hours)", "value": 24,
                           "tags": ["time"], "section": 1, "subsection": "Time Variables"},

            # ==================================================================================================
            # ROUTING PARAMETERS
            # ==================================================================================================

            "baseflow": {"keyword": "BASEFLOW:", "describe": "Baseflow discharge (m3/s)", "value": 0.2,
                         "tags": ["flow"], "section": 1, "subsection": "Routing Variables"},
            "kinemvelcoef": {"keyword": "KINEMVELCOEF:", "describe": "Kinematic routing velocity coefficient",
                             "value": 3, "tags": ["flow"], "section": 1, "subsection": "Routing Variables"},
            "flowexp": {"keyword": "FLOWEXP:", "describe": "Nonlinear discharge coefficient", "value": 0.3,
                        "tags": ["flow"], "section": 1, "subsection": "Routing Variables"},
            "channelroughness": {"keyword": "CHANNELROUGHNESS:", "describe": "Uniform channel roughness value",
                                 "value": 0.15, "tags": ["flow"], "section": 1, "subsection": "Routing Variables"},
            "channelwidth": {"keyword": "CHANNELWIDTH:", "describe": "Uniform channel width  (meters)", "value": 12,
                             "tags": ["flow"], "section": 1, "subsection": "Routing Variables"},
            "channelwidthcoeff": {"keyword": "CHANNELWIDTHCOEFF:",
                                  "describe": "Coefficient in width-area relationship", "value": 2.33,
                                  "tags": ["flow"], "section": 1, "subsection": "Routing Variables"},
            "channelwidthexpnt": {"keyword": "CHANNELWIDTHEXPNT:", "describe": "Exponent in width-area relationship",
                                  "value": 0.54, "tags": ["flow"], "section": 1, "subsection": "Routing Variables"},
            "channelwidthfile": {"keyword": "CHANNELWIDTHFILE:", "describe": "Filename that contains channel widths",
                                 "value": None, "tags": ["flow"], "section": 1, "subsection": "Routing Variables"},

            # ==================================================================================================
            # MODEL RUN OPTIONS
            # ==================================================================================================

            "optmeshinput": {"keyword": "OPTMESHINPUT:", "describe": ("Mesh input data option\n"
                                                                     "1  tMesh data\n"
                                                                     "2  Point Triangulator"),
                             "value": 2, "tags": ["mesh"], "section": 2},

            "rainsource": {"keyword": "RAINSOURCE:", "describe": ("Rainfall data source option\n"
                                                                   "1  Gridded Rainfall\n"
                                                                   "2  Rain gauges"),
                           "value": 2, "tags": ["meterological"], "section": 2},

            "optevapotrans": {"keyword": "OPTEVAPOTRANS:", "describe": ("Option for evapoTranspiration scheme\n"
                                                                        "0  Inactive evapotranspiration\n"
                                                                        "1  Penman-Monteith method\n"
                                                                        "2  Pan evaporation measurements"),
                              "value": 1, "tags": ["hydro"], "section": 2},

            "optsnow": {"keyword": "OPTSNOW:", "describe": ("Option for snow module\n"
                                                            "0  Inactive snow module\n"
                                                            "1  Single layer snow module"),
                        "value": 1, "tags": ["hydro"], "section": 2},

            "hillalbopt": {"keyword": "HILLALBOPT:", "describe": ("Option for albedo of surrounding hillslopes\n"
                                                                  "0  Snow albedo for hillslopes\n"
                                                                  "1  Land-cover albedo for hillslopes\n"
                                                                  "2  Dynamic albedo for hillslopes"),
                           "value": 0, "tags": ["meterological"], "section": 2},

            "optradshelt": {"keyword": "OPTRADSHELT:", "describe": ("Option for local and remote radiation sheltering\n"
                                                                    "0  Local controls on shortwave radiation\n"
                                                                    "1  Remote controls on diffuse shortwave\n"
                                                                    "2  Remote controls on entire shortwave\n"
                                                                    "3  No sheltering"),
                            "value": 0, "tags": ["meterological"], "section": 2},

            "optintercept": {"keyword": "OPTINTERCEPT:", "describe": ("Option for interception scheme\n"
                                                                      "0  Inactive interception\n"
                                                                      "1  Canopy water balance method"),
                             "value": 1, "tags": ["hydro"], "section": 2},

            "optlanduse": {"keyword": "OPTLANDUSE:", "describe": ("Option for static or dynamic land cover\n"
                                                                  "0  ID Map with Lookup Table (LANDTABLENAME, LANDMAPNAME)\n"
                                                                  "1  Dynamic updating of gridded parameter maps, ID Map & Table for non-gridded parameters (LUGRID)\n"
                                                                  "2  Static gridded parameter maps, ID Map & Table for non-gridded parameters (LUGRID)"),
                           "value": 0, "tags": ["hydro"], "section": 2},

            "optluinterp": {"keyword": "OPTLUINTERP:", "describe": ("Option for interpolation of land cover\n"
                                                                    "0  Constant (previous) values between land cover\n"
                                                                    "1  Linear interpolation between land cover"),
                            "value": 0, "tags": ["hydro"], "section": 2},

            "gfluxoption": {"keyword": "GFLUXOPTION:", "describe": ("Option for ground heat flux\n"
                                                                    "0  Inactive ground heat flux\n"
                                                                    "1  Temperature gradient method\n"
                                                                    "2  Force-Restore method"),
                            "value": 2, "tags": ["hydro"], "section": 2},

            "metdataoption": {"keyword": "METDATAOPTION:", "describe": ("Option for meteorological data\n"
                                                                        "0  Inactive meteorological data\n"
                                                                        "1  Weather station point data\n"
                                                                        "2  Gridded meteorological data"),
                              "value": 1, "tags": ["meterological"], "section": 2},

            "optbedrock": {"keyword": "OPTBEDROCK:", "describe": ("Option for uniform or variable depth\n"
                                                                  "0  Uniform bedrock depth (DEPTHTOBEDROCK)\n"
                                                                  "1  Spatial grid of bedrock depth (BEDROCKFILE)"),
                           "value": 0, "tags": ["hydro"], "section": 2},

            "widthinterpolation": {"keyword": "WIDTHINTERPOLATION:", "describe": ("Option for interpolating width values\n"
                                                                                  "0  Interpolate between measured and observed\n"
                                                                                  "1  Interpolate only between measured"),
                                   "value": 0, "tags": ["flow"], "section": 2},

            "optgwfile": {"keyword": "OPTGWFILE:", "describe": ("Option for groundwater initial file\n"
                                                                "0  Resample ASCII grid file in GWATERFILE\n"
                                                                "1  Read in Voronoi polygon file with GW levels"),
                          "value": 0, "tags": ["hydro"], "section": 2},

            "optgroundwater": {"keyword": "OPTGROUNDWATER:", "describe": ("Option for groundwater module\n"
                                                                          "0  Inactive groundwater module\n"
                                                                          "1  Active groundwater module"),
                               "value": 1, "tags": ["hydro"], "section": 2},

            "optspatial": {"keyword": "OPTSPATIAL:", "describe": ("Option for writing spatial outputs\n"
                                                                  "0  Inactive spatial outputs\n"
                                                                  "1  Active spatial outputs at interval SPOPINTRVL"),
                           "value": 1, "tags": ["hydro"], "section": 2},

            "optrunon": {"keyword": "OPTRUNON:", "describe": ("Option for runon in overland flow paths\n"
                                                              "0  Inactive runon module\n"
                                                              "1  Active runon module [IN DEVELOPMENT]"),
                         "value": 0, "tags": ["hydro"], "section": 2},

            "optreservoir": {"keyword": "OPTRESERVOIR:", "describe": ("Option for level pool reservoir routing\n"
                                                                      "0  Inactive reservoir routing module\n"
                                                                      "1  Active level pool reservoir routing module"),
                             "value": 0, "tags": ["hydro"], "section": 2},

            "optsoiltype": {"keyword": "OPTSOILTYPE:", "describe": ("Option for soil input type\n"
                                                                    "0  ID Map with Lookup Table (SOILTABLENAME, SOILMAPNAME)\n"
                                                                    "1  Static gridded parameter maps, ID Map & Table for non-gridded parameters (SCGRID)"),
                            "value": 0, "tags": ["hydro"], "section": 2},

            "optpercolation": {"keyword": "OPTPERCOLATION:", "describe": ("Option for percolation losses from channels\n"
                                                                          "0  Inactive channel transmission losses\n"
                                                                          "1  Constant loss method\n"
                                                                          "2  Transient loss method\n"
                                                                          "3  Green-Ampt loss method [IN DEVELOPMENT]"),
                               "value": 0, "tags": ["hydro"], "section": 2},

            # ==================================================================================================
            # MESH / INPUT FILES
            # ==================================================================================================

            "inputdatafile": {"keyword": "INPUTDATAFILE:", "describe": "tMesh input file base name for Mesh files, see OPTMESHINPUT",
                              "value": None, "tags": ["mesh"], "section": 3, "subsection": "Mesh Generation"},
            "pointfilename": {"keyword": "POINTFILENAME:", "describe": "tMesh input file name Points files, see OPTMESHINPUT",
                              "value": None, "tags": ["mesh"], "section": 3, "subsection": "Mesh Generation"},

            # ==================================================================================================
            # GRID RESAMPLE
            # ==================================================================================================

            "soiltablename": {"keyword": "SOILTABLENAME:", "describe": "Soil parameter reference table (*.sdt), see OPTSOILTYPE",
                              "value": None, "tags": ["spatial"], "section": 3, "subsection": "Resampling Grids"},
            "soilmapname": {"keyword": "SOILMAPNAME:", "describe": "Soil texture ASCII grid (*.soi), see OPTSOILTYPE", "value": None,
                            "tags": ["spatial"], "section": 3, "subsection": "Resampling Grids"},
            "landtablename": {"keyword": "LANDTABLENAME:", "describe": "Land use parameter reference table, see OPTLANDUSE",
                              "value": None, "tags": ["spatial"], "section": 3, "subsection": "Resampling Grids"},
            "landmapname": {"keyword": "LANDMAPNAME:", "describe": "Land use ASCII grid (*.lan), see OPTLANDUSE", "value": None,
                            "tags": ["spatial"], "section": 3, "subsection": "Resampling Grids"},
            "gwaterfile": {"keyword": "GWATERFILE:", "describe": "Ground water ASCII grid (*iwt), see OPTGWFILE", "value": None,
                           "tags": ["spatial"], "section": 3, "subsection": "Resampling Grids"},
            "demfile": {"keyword": "DEMFILE:", "describe": "DEM ASCII grid for sky and land view factors (*.dem), see OPTRADSHELT",
                        "value": None, "tags": ["spatial"], "section": 3, "subsection": "Resampling Grids"},
            "rainfile": {"keyword": "RAINFILE:", "describe": "Base name of the radar ASCII grid, see RAINSOURCE", "value": None,
                         "tags": ["meterological"], "section": 3, "subsection": "Resampling Grids"},
            "rainextension": {"keyword": "RAINEXTENSION:", "describe": "Extension for the radar ASCII grid, see RAINSOURCE",
                              "value": None, "tags": ["meterological"], "section": 3, "subsection": "Resampling Grids"},
            "raindistribution": {"keyword": "RAINDISTRIBUTION:", "describe": "Precipitation distributed as provided or mean areal precipitation, see RAINSOURCE",
                                 "value": 0, "tags": ["meterological"], "section": 3, "subsection": "Resampling Grids"},
            "depthtobedrock": {"keyword": "DEPTHTOBEDROCK:", "describe": "Uniform depth to bedrock (meters), see OPTBEDROCK",
                               "value": 15, "tags": ["hydro"], "section": 3, "subsection": "Resampling Grids"},
            "bedrockfile": {"keyword": "BEDROCKFILE:", "describe": "Bedrock depth ASCII grid (*.brd), see OPTBEDROCK",
                            "value": None, "tags": ["spatial"], "section": 3, "subsection": "Resampling Grids"},
            "lugrid": {"keyword": "LUGRID:", "describe": "Land cover grid data file (*.gdf), see OPTLANDUSE", "value": None,
                       "tags": ["spatial"], "section": 3, "subsection": "Resampling Grids"},
            "scgrid": {"keyword": "SCGRID:", "describe": "Soil cover grid data file (*.gdf), see OPTSOILTYPE", "value": None,
                       "tags": ["spatial"], "section": 3, "subsection": "Resampling Grids"},

            # ==================================================================================================
            # METEROLOGICAL DATA
            # ==================================================================================================

            "tlinke": {"keyword": "TLINKE:", "describe": "Atmospheric turbidity parameter", "value": 2.5,
                       "tags": ["meterological"], "section": 3, "subsection": "Meteorological Variables"},
            "templapse": {"keyword": "TEMPLAPSE:", "describe": "Temperature lapse rate", "value": -0.0065,
                          "tags": ["meterological"], "section": 3, "subsection": "Meteorological Variables"},
            "preclapse": {"keyword": "PRECLAPSE:", "describe": "Precipitation lapse rate", "value": 0,
                          "tags": ["meterological"], "section": 3, "subsection": "Meteorological Variables"},
            "utcoffset": {"keyword": "UTCOFFSET:", "describe": "UTC (GMT) offset of the watershed centroid, used for solar position calculations (hours)",
                          "value": None, "tags": ["time"], "section": 3, "subsection": "Meteorological Variables"},
            "centroidlat": {"keyword": "CENTROIDLAT:", "describe": "Latitude of the watershed centroid, used for solar position calculations (decimal degrees)",
                            "value": None, "tags": ["solar"], "section": 3, "subsection": "Meteorological Variables"},
            "centroidlong": {"keyword": "CENTROIDLONG:", "describe": "Longitude of the watershed centroid, used for solar position calculations (decimal degrees)",
                             "value": None, "tags": ["solar"], "section": 3, "subsection": "Meteorological Variables"},
            "hydrometstations": {"keyword": "HYDROMETSTATIONS:",
                                 "describe": "Hydrometeorological station file (*.sdf), see METDATAOPTION", "value": None,
                                 "tags": ["meterological"], "section": 3, "subsection": "Meteorological Data"},
            "hydrometgrid": {"keyword": "HYDROMETGRID:", "describe": "Hydrometeorological grid data file (*.gdf), see METDATAOPTION",
                             "value": None, "tags": ["meterological"], "section": 3, "subsection": "Meteorological Data"},
            "gaugestations": {"keyword": "GAUGESTATIONS:", "describe": "Rain Gauge station file (*.sdf), see RAINSOURCE",
                              "value": None, "tags": ["meterological"], "section": 3, "subsection": "Meteorological Data"},

            # ==================================================================================================
            # OUTPUT DATA
            # ==================================================================================================

            "outfilename": {"keyword": "OUTFILENAME:", "describe": "Filepath and base name of model outputs",
                            "value": None, "tags": ["output"], "section": 3, "subsection": "Output Data"},
            "nodeoutputlist": {"keyword": "NODEOUTPUTLIST:",
                               "describe": "Filename with Nodes for Dynamic Output (*.nol)", "value": None,
                               "tags": ["output"], "section": 3, "subsection": "Output Data"},
            "hydronodelist": {"keyword": "HYDRONODELIST:",
                              "describe": "Filename with Nodes for HydroModel Output (*.nol)", "value": None,
                              "tags": ["output"], "section": 3, "subsection": "Output Data"},
            "outletnodelist": {"keyword": "OUTLETNODELIST:",
                               "describe": "Filename with Interior Nodes for  Output (*.nol)", "value": None,
                               "tags": ["output"], "section": 3, "subsection": "Output Data"},

            # ==================================================================================================
            # Module Input Files
            # ==================================================================================================

            "respolygonid": {"keyword": "RESPOLYGONID:", "describe": "Path to file of node IDs representing reservoirs, see OPTRESERVOIR",
                             "value": None, "tags": ["hydro"], "section": 3, "subsection": "Module Input Files"},
            "resdata": {"keyword": "RESDATA:", "describe": "Path to file of elevation-discharge-storage information for each type of reservoir, see OPTRESERVOIR",
                        "value": None, "tags": ["hydro"], "section": 3, "subsection": "Module Input Files"},
            "snowfilename": {"keyword": "SNOWFILENAME:", "describe": "Snow parameter reference file (*.spf), see OPTSNOW",
                             "value": None, "tags": ["meterological"], "section": 3, "subsection": "Module Input Files"},

            # ==================================================================================================
            # CHANNEL TRANSMISSION LOSSES
            # ==================================================================================================

            "channelconductivity": {"keyword": "CHANNELCONDUCTIVITY:", "describe": "Conductivity in channel for all methods (mm/hr)",
                                    "value": 0, "tags": ["hydro"], "section": 4, "subsection": "Channel Transmission Losses"},
            "transientconductivity": {"keyword": "TRANSIENTCONDUCTIVITY:", "describe": "Conductivity in channel during transient period in (mm/hr)",
                                      "value": 0, "tags": ["hydro"], "section": 4, "subsection": "Channel Transmission Losses"},
            "transienttime": {"keyword": "TRANSIENTTIME:", "describe": "Time until transient period ends (hours)", "value": 0,
                              "tags": ["hydro"], "section": 4, "subsection": "Channel Transmission Losses"},
            "channelporosity": {"keyword": "CHANNELPOROSITY:", "describe": "Porosity in channel", "value": 0,
                                "tags": ["hydro"], "section": 4, "subsection": "Channel Transmission Losses"},
            "chanporeindex": {"keyword": "CHANPOREINDEX:", "describe": "Channel pore index in channel", "value": 0,
                              "tags": ["hydro"], "section": 4, "subsection": "Channel Transmission Losses"},
            "chanpsib": {"keyword": "CHANPSIB:", "describe": "Matric potential in channel", "value": 0,
                         "tags": ["hydro"], "section": 4, "subsection": "Channel Transmission Losses"},

            # ==================================================================================================
            # RESTART MODE
            # ==================================================================================================
            "restartmode": {"keyword": "RESTARTMODE:", "describe": ("Restart Mode Option\n"
                                                                    "0  No reading or writing of restart\n"
                                                                    "1  Write files (only for initial runs)\n"
                                                                    "2  Read file only (to start at some specified time)\n"
                                                                    "3  Read a restart file and continue to write"),
                            "value": 0, "tags": ["restart"], "section": 5},
            "restartintrvl": {"keyword": "RESTARTINTRVL:", "describe": "Time set for restart output (hours)",
                              "value": None, "tags": ["restart"], "section": 5},
            "restartdir": {"keyword": "RESTARTDIR:", "describe": "Path of directory for restart output",
                           "value": None, "tags": ["restart"], "section": 5},
            "restartfile": {"keyword": "RESTARTFILE:", "describe": "Actual file to restart a run", "value": None,
                            "tags": ["restart"], "section": 5},

            # ==================================================================================================
            # PARALLEL MODE
            # ==================================================================================================
            "parallelmode": {"keyword": "PARALLELMODE:", "describe": ("Parallel or Serial Mode Option\n"
                                                                      "0  Run in serial mode\n"
                                                                      "1  Run in parallel mode"),
                             "value": 0, "tags": ["parallel"], "section": 6},
            "graphoption": {"keyword": "GRAPHOPTION:", "describe": ("Graph File Type Option\n"
                                                                    "0  Default partitioning of the graph\n"
                                                                    "1  Reach-based partitioning\n"
                                                                    "2  Inlet/outlet-based partitioning"),
                            "value": 0, "tags": ["parallel"], "section": 6},
            "graphfile": {"keyword": "GRAPHFILE:", "describe": "Reach connectivity filename (graph file option 1,2)",
                          "value": None, "tags": ["parallel"], "section": 6},
        }

        return input_file
