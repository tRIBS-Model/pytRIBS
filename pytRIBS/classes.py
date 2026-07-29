# system packages

# external packages

# pytRIBS
from pytRIBS.shared.aux import Aux
from pytRIBS.model.model import ModelProcessor
from pytRIBS.shared.infile_mixin import Infile
from pytRIBS.shared.inout import InOut
from pytRIBS.shared.shared_mixin import Shared, Meta
from pytRIBS.results.waterbalance import WaterBalance
from pytRIBS.results.read import Read
from pytRIBS.results.visualize import Viz
from pytRIBS.results.evaluate import Evaluate


# preprocessing componets
from pytRIBS.soil.soil import SoilProcessor
from pytRIBS.met.met import MetProcessor
from pytRIBS.spinup.spinup import SpinupProcessor
from pytRIBS.mesh.mesh import Preprocess, GenerateMesh, MeshProcessor
from pytRIBS.land.land import LandProcessor

import os



class Project:
    """
    pytRIBS Project Class for managing directories and metadata in a specified root directory.

    This class initializes with a base directory, a project name, and an EPSG code. It sets up a
    predefined set of directories for data, results, and various sub-categories. It also provides
    functionality to create these directories if they do not already exist.

    Parameters
    ----------
    base_dir : str
        The base directory path for the project.
    name : str
        The name of the project.
    epsg : int
        The EPSG code representing the coordinate system.

    Attributes
    ----------
    base_dir : str
        The base directory where all project-related directories will be created.
    meta : dict
        A dictionary to store metadata about the project, including 'Name' and 'EPSG'.
    directories : dict
        A dictionary defining the structure of directories to be created within the base directory.
    """
    def __init__(self, base_dir, name, epsg):
        self.base_dir = base_dir
        Meta.__init__(self)
        self.meta['Name'] = name
        self.meta['EPSG'] = epsg
        self.directories = {
            "model": os.path.join("data", "model"),
            "preprocessing": os.path.join("data", "preprocessing"),
            "results": "results",
            "spinup": os.path.join("results", "spinup"),
            "restart": "restart",
            "soil": os.path.join("data", "model", "soil"),
            "land": os.path.join("data", "model", "land"),
            "met_precip": os.path.join("data", "model", "met", "precip"),
            "met_meteor": os.path.join("data", "model", "met", "meteor"),
            "mesh": os.path.join("data", "model", "mesh")
        }
        self._create_directories()

    def _create_directories(self):
        """
        Creates directories defined in the `directories` attribute within the base directory.

        This method uses `os.makedirs` with `exist_ok=True` to ensure that all required directories are
        created if they do not already exist. It traverses through the `directories` dictionary, combining
        the base directory with each relative path to create the full directory paths.
        """
        for key, rel_path in self.directories.items():
            full_path = os.path.join(self.base_dir, rel_path)
            os.makedirs(full_path, exist_ok=True)

class Model(Infile, Shared, Aux, ModelProcessor, Preprocess, InOut, SpinupProcessor):
    """
    pytRIBS Model class.

    This class provides access to the underlying framework of a tRIBS (TIN-based Real-time Integrated Basin
    Simulator) simulation. The Model class can be initialized at the top-level to
    facilitate model setup, simulation, post-processing, and can be used for manipulating and generating multiple
    simulations efficiently.

    Parameters
    ----------
    input_file : str, optional
        Path to a template `.in` file. Default is `None`.
    met : object, optional
        pytRIBS Met Class object Default is `None`.
    land : object, optional
        pytRIBS Land object. Default is `None`.
    soil : object, optional
        pytRIBS Soil object. Default is `None`.
    mesh : object, optional
        pytRIBS Mesh object. Default is `None`.
    meta : dict, optional
        pytRIBS Meta object Default is `None`.

    Attributes
    ----------
    input_options : dict
        A dictionary of the necessary keywords for a tRIBS `.in` file.
    model_input_file : str
        Path to a template `.in` file with the specified paths for model results, inputs, etc.
        """

    def __init__(self, input_file=None, met=None, land=None, soil=None, mesh=None, meta=None):
        # attributes
        self.options = self.create_input_file()  # input options for tRIBS model run
        self.snow_options = self.create_snow_params()  # snow module parameter options

        if input_file is not None:
            self.read_input_file(input_file)

            # Load snow parameters from the *.spf file if one is referenced
            if self.options['snowfilename']['value'] is not None:
                self.read_snow_params()

        Meta.__init__(self)

        if meta is not None:
            self.meta = meta

        # Initialize with provided instances
        self.met = met
        self.land = land
        self.soil = soil
        self.mesh = mesh

        # Merge options from provided instances
        self._update_shared_options(met=met, land=land, soil=soil, mesh=mesh)

    # SIMULATION METHODS
    def __getattr__(self, name):
        # __getattr__ only fires for attributes missing from __dict__. Look up options/snow_options
        # via __dict__ directly so that probing for a missing 'options' (e.g. during copy/pickle of
        # a not-yet-initialized instance) raises AttributeError instead of recursing.
        options = self.__dict__.get('options')
        if options is not None and name in options:
            return options[name]
        snow_options = self.__dict__.get('snow_options')
        if snow_options is not None and name in snow_options:
            return snow_options[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __dir__(self):
        # Include the keys from the options dictionary and the methods of the class
        return list(
            set(super().__dir__() + list(self.options.keys()))) if self.options is not None else super().__dir__()

    def _update_shared_options(self, met=None, land=None, soil=None, mesh=None):
        # List of provided instances
        instances = [met, land, soil, mesh]

        for instance in instances:
            if instance:
                # Update only shared options
                for key in instance.__dict__:
                    if key in self.options:
                        self.options[key] = instance.__dict__[key]

class Results(Infile, Shared, WaterBalance, Read, Viz, Evaluate):
    """
    pytRIBS Results Class.

    This class provides a framework for analyzing and visualizing individual tRIBS simulations. It takes an instance of
    the `Simulation` class and provides time-series and water balance analysis of the model results.

    Parameters
    ----------
    input_file : str, required
        Path to the input file containing the necessary options for initializing the `Results` class attributes.
    meta : dict, optional
        Metadata associated with the `Results` instance.

    Attributes
    ----------
    options : dict
        A dictionary of input options for the tRIBS model run.
    element : dict
        A dictionary for storing elements related to the results.
    mrf : dict
        A dictionary containing `mrf` and `waterbalance`, which are initialized to `None`.
    meta : dict, optional
        Metadata dictionary for additional information. Default is `None`.
    """
    def __init__(self, input_file, meta=None):
        # setup model paths and options for Result Class
        self.options = self.create_input_file()  # input options for tRIBS model run
        self.read_input_file(input_file)

        # attributes for analysis, plotting, and archiving model results
        self.element = {}
        self.mrf = {'mrf': None, 'waterbalance': None}
        Meta.__init__(self)

        if meta is not None:
            self.meta = meta

        self.get_invariant_properties()  # shared


class Soil(SoilProcessor):
    """
    pytRIBS Soil Class.

    This class handles soil-related data and options for the tRIBS model. It manages attributes related to soil
    mapping, soil tables, and groundwater files.

    Parameters
    ----------
    input_file : str, optional
        Path to the input file containing the necessary options for initializing the `Soil` class attributes.
    meta : dict, optional
        Metadata associated with the `Soil` instance.

    Attributes
    ----------
    soilmapname : str
        The name of the soil map file.
    soiltablename : str
        The name of the soil table file.
    scgrid : str
        The path to the SCGRID file.
    optsoiltype : int
        Option for soil type.
    optgroundwater : int
        Option for groundwater.
    optgwfile : int
        Option for groundwater file.
    optbedrock : int
        Option for bedrock.
    bedrockfile : str
        The path to the bedrock file.
    gwaterfile : str
        The path to the groundwater file.
    """

    def __init__(self, input_file=None,meta=None):

        Meta.__init__(self)

        if meta is not None:
            self.meta=meta

        # read_input_file assigns values into self.options in place
        self.options = Infile.create_input_file()
        if input_file is not None:
            Shared.read_input_file(self, input_file)
        options = self.options

        # Initialize attributes
        self.soilmapname = options['soilmapname']
        self.soiltablename = options['soiltablename']
        self.scgrid = options['scgrid']
        self.optsoiltype = options['optsoiltype']
        self.optgroundwater = options['optgroundwater']
        self.optgwfile = options['optgwfile']
        self.optbedrock = options['optbedrock']
        self.bedrockfile = options['bedrockfile']
        self.gwaterfile = options['gwaterfile']


class Land(LandProcessor):
    """
    pytRIBS Land Class.

    This class handles land-related data and options for the tRIBS model. It manages attributes related to land mapping,
    land tables, and land use grids.

    Parameters
    ----------
    input_file : str, optional
        Path to the input file containing the necessary options for initializing the `Land` class attributes.
    meta : dict, optional
        Metadata associated with the `Land` instance.

    Attributes
    ----------
    landmapname : str
        The name of the land map file.
    landtablename : str
        The name of the land table file.
    lugrid : str
        The path to the LUGRID file.
    optlanduse : int
        Option for land use.
    optluintercept : int
        Option for land use interpolation.
    """

    def __init__(self, input_file=None,meta=None):

        Meta.__init__(self)

        if meta is not None:
            self.meta=meta

        # read_input_file assigns values into self.options in place
        self.options = Infile.create_input_file()
        if input_file is not None:
            Shared.read_input_file(self, input_file)
        options = self.options

        # Initialize attributes
        self.landmapname = options['landmapname']
        self.landtablename = options['landtablename']
        self.lugrid = options['lugrid']
        self.optlanduse = options['optlanduse']
        self.optluintercept = options['optluinterp']


class Mesh(MeshProcessor):
    """
    A pytRIBS Mesh Class.

    This class manages the creation and processing of mesh data for tRIBS simulations. It handles preprocessing of
    watershed and stream network data, and integrates with mesh generation routines. For more details see base classes
    and example below.

    Parameters
    ----------
    preprocess_args : tuple, optional
        Arguments for initializing the Preprocess class. Required if `generate_mesh_args` is provided.
    generate_mesh_args : tuple, optional
        Arguments for initializing the GenerateMesh class. Must be provided if `preprocess_args` is given.
    input_file : str, optional
        Path to the input file for initializing attributes.
    meta : dict, optional
        Metadata associated with the mesh.
    mesh_dir : str, optional
        Default output directory for generated mesh files (typically
        ``proj.directories['mesh']``). Used by :meth:`build_mesh` and friends when no
        explicit output location is given. Defaults to the current working directory.

    Attributes
    ----------
    pointfilename : str
        The name of the file containing the mesh points.
    graphfile : str
        The name of the reach connectivity (``.reach``) file used to partition a parallel
        run. Optional: if the file does not exist tRIBS generates the partition in-process
        with METIS and writes it there for reuse, sized to the number of MPI processes.
    optmeshinput : int
        Option flag for mesh input processing.
    graphoption : int
        Reach partitioning method used for parallel runs: 0 = SF (surface flow edges only),
        1 = SSF (flow plus subsurface flux edges), 2 = SSFH (SSF plus headwater balancing).
    demfile : str
        The name of the file containing the Digital Elevation Model (DEM) data.
    preprocess : :class:`~pytRIBS.preprocess.Preprocess`, optional
        An instance of the :class:`~pytRIBS.preprocess.Preprocess` class used for initial data extraction and processing.
    mesh_generator : :class:`~pytRIBS.meshgeneration.GenerateMesh`, optional
        An instance of the :class:`~pytRIBS.meshgeneration.GenerateMesh` class used for mesh generation.


    Example
    -------
    To create and use an instance of the `Mesh` class:
    TODO UPDATE!
    >>> mesh = Mesh(preprocess_args=(arg1, arg2, arg3), generate_mesh_args=(arg4, arg5, arg6, arg7))
    >>> print(mesh.pointfilename)
    'path/to/pointfile'
    >>> print(mesh.demfile)
    'path/to/demfile'
    """

    def __init__(self, preprocess_args=None, generate_mesh_args=None,
                 input_file=None, meta=None, mesh_dir=None):
        Meta.__init__(self)

        if meta is not None:
            self.meta = meta

        # Default directory for generated mesh outputs (e.g. proj.directories['mesh']).
        # Used by build_mesh/generate_pslg_mesh/generate_points_mesh when no explicit
        # output location is given, mirroring Preprocess's dir_proccesed convention.
        self.mesh_dir = mesh_dir

        boundary_gdf = None
        if preprocess_args is not None:  # TODO need to catch if generate_mesh_args is NONE
            self.preprocess = Preprocess(*preprocess_args)
            _, bound_path, stream_path, out_path, _ = generate_mesh_args
            boundary_gdf = self.preprocess.extract_watershed_and_stream_network(out_path, bound_path,
                                                                                stream_path)
            self.meta = self.preprocess.meta

        if generate_mesh_args is not None:
            self.mesh_generator = GenerateMesh(*generate_mesh_args)

        # read_input_file assigns values into self.options in place
        self.options = Infile.create_input_file()
        if input_file is not None:
            Shared.read_input_file(self, input_file)
        options = self.options

        # Initialize attributes
        self.pointfilename = options['pointfilename']
        self.inputdatafile = options['inputdatafile']
        self.graphfile = options['graphfile']
        self.optmeshinput = options['optmeshinput']
        self.graphoption = options['graphoption']
        self.demfile = options['demfile']

        # Solar position options auto-populated from the watershed centroid (see update_solar_position)
        self.utcoffset = options['utcoffset']
        self.centroidlat = options['centroidlat']
        self.centroidlong = options['centroidlong']

        # If a watershed was delineated above, auto-populate the solar position keywords from it
        if boundary_gdf is not None:
            Aux.update_solar_position(self, boundary_gdf.geometry.iloc[0])

class Met(MetProcessor):
    """
    A pytRIBS Met Class.

    This class handles the meteorological data for tRIBS simulations. It initializes various parameters related to
    meteorological stations, rain files, and other related metadata. The class is used to configure and manage the
    meteorological input options required for the simulation.

        Parameters
    ----------
    input_file : str, optional
        Path to the input file containing the necessary options for initializing the `Met` class attributes.
    meta : dict, optional
        Metadata associated with the `Met` instance.

    Attributes
    ----------
    hydrometstations : str
        The path or name of the file containing hydrometeorological station data.
    gaugestations : str
        The path or name of the file containing gauge station data.
    rainfile : str
        The path or name of the file containing rainfall data.
    hydrometgrid : str
        The path or name of the file containing the hydrometeorological grid data.
    rainsource : str
        The source of the rainfall data.
    rainextension : str
        The file extension for the rainfall data files.
    """

    def __init__(self, input_file=None, meta=None):

        Meta.__init__(self)

        if meta is not None:
            self.meta = meta

        # read_input_file assigns values into self.options in place
        self.options = Infile.create_input_file()
        if input_file is not None:
            Shared.read_input_file(self, input_file)
        options = self.options

        self.hydrometstations = options['hydrometstations']
        self.gaugestations = options['gaugestations']
        self.rainfile = options['rainfile']
        self.hydrometgrid = options['hydrometgrid']
        self.metdataoption = options['metdataoption']
        self.rainsource = options['rainsource']
        self.rainextension = options['rainextension']

        # pytRIBS-internal naming prefix for generated met/precip output files. Not a tRIBS input                                
        # keyword (so it is not written to the .in file); used by the met workflow.                                              
        self.hydrometbasename = {'value': None}       

        # Solar position options auto-populated from the watershed centroid (see update_solar_position)
        self.utcoffset = options['utcoffset']
        self.centroidlat = options['centroidlat']
        self.centroidlong = options['centroidlong']
