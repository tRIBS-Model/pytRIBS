import os
import shutil

import pandas as pd
from rasterio.windows import from_bounds
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
from scipy.spatial import distance

from shapely.vectorized import contains
from shapely.ops import nearest_points

from math import ceil
from rasterio.mask import mask
import numpy as np
import geopandas as gpd
from rasterio.features import shapes
from whitebox import WhiteboxTools
from shapely.geometry import Point
import rasterio
import pywt
import pyvista as pv
from shapely.ops import unary_union

from pytRIBS.shared.inout import InOut

from pytRIBS.shared.shared_mixin import Meta, Shared
from pytRIBS.mesh.run_docker import MeshBuilderDocker


class Preprocess:
    """
    A class for preprocessing digital elevation models (DEMs) and extracting watershed and stream network data.

    This class provides methods for preparing DEMs, setting up metadata, and processing elevation data to
    extract watershed and stream network features. It uses the WhiteboxTools library for various geospatial
    operations and supports setting verbose mode for logging.

    Parameters
    ----------
    outlet : tuple of float
        A tuple containing the outlet coordinates (longitude, latitude) used in the processing.
    snap_distance : float
        The distance used for snapping outlet points to the nearest flow path.
    threshold_area : float
        The area threshold used to identify streams in the flow accumulation raster.
    dem_path : str
        The file path to the digital elevation model (DEM) to be processed.
    verbose_mode : bool
        Flag to enable verbose mode for WhiteboxTools, controlling the amount of log output.
    meta : dict
        A dictionary containing metadata for the project, including 'Name', 'EPSG', and 'Scenario'.
    dir_proccesed : str, optional
        Directory path for saving processed files. If not provided, defaults to 'preprocessing'.

    Raises
    ------
    OSError
        If there is an issue creating directories or accessing files.
    ValueError
        If the EPSG code of the DEM does not match the project metadata and cannot be reconciled.

    Attributes
    ----------
    outlet : tuple of float
        The outlet coordinates used in the processing.
    snap_distance : float
        The distance used for snapping outlet points to the nearest flow path.
    threshold_area : float
        The area threshold used to identify streams.
    dem_preprocessing : str
        The absolute path to the digital elevation model (DEM) file.
    output_dir : str
        The directory where processed files are saved.
    meta : dict
        Metadata for the project, including 'Name', 'EPSG', and 'Scenario'.
    wbt : WhiteboxTools
        Instance of WhiteboxTools for performing geospatial operations.

    Examples
    --------
    To initialize and preprocess a DEM:

    >>> meta_data = {'Name': 'Watershed Project', 'EPSG': 4326, 'Scenario': 'Base'}
    >>> dem_preprocessor = DEMPreprocessor(
    ...     outlet=(45.1234, -120.5678),
    ...     snap_distance=100.0,
    ...     threshold_area=50.0,
    ...     dem_path="path/to/dem.tif",
    ...     verbose_mode=True,
    ...     meta=meta_data
    ... )
    >>> dem_preprocessor.process()

    """
    def __init__(self, outlet, snap_distance, threshold_area, dem_path, verbose_mode, meta, dir_proccesed=None):

        self.outlet = outlet
        self.snap_distance = snap_distance
        self.threshold_area = threshold_area

        Meta.__init__(self)

        self.meta['Name'] = meta['Name']
        self.meta['EPSG'] = meta['EPSG']
        self.meta['Scenario'] = meta['Scenario']

        self.wbt = WhiteboxTools()
        self.wbt.set_verbose_mode(verbose_mode)

        name = self.meta['Name']

        if name is None:
            name = 'Basin'
            self.meta['Name'] = name

        if dir_proccesed is None:
            dir_proccesed = 'preprocessing'
            os.makedirs(dir_proccesed, exist_ok=True)

        if dem_path is not None:
            with rasterio.open(dem_path) as src:
                crs = src.crs
                epsg = self.meta['EPSG']
                if epsg != crs.to_epsg():
                    print(f'EPSG code for DEM { crs.to_epsg()}  does not match with project meta data {epsg}.\n'
                          'Converting project EPSG to DEM EPSG')
                    self.meta['EPSG'] = crs.to_epsg()

        self.dem_preprocessing = os.path.abspath(dem_path)
        self.output_dir = dir_proccesed

    def fill_depressions(self, output_path=None):
        """
        Fill depressions (sinks) in the DEM within the watershed and save the filled DEM to the specified location.

        This method uses the WhiteboxTools library to fill depressions (sinks) in the digital elevation model (DEM).
        After the depressions are filled, the DEM is saved to the provided output path or a default path based on
        the project metadata.

        Parameters
        ----------
        output_path : str, optional
            The file path where the filled DEM will be saved. If not provided, the default path will be
            `'{output_dir}/{project_name}_filled.tif'`.

        Returns
        -------
        str
            The absolute path to the saved filled DEM file.

        Examples
        --------
        To fill depressions and save the result to a specified file:

        >>> filled_dem_path = dem_preprocessor.fill_depressions(output_path='path/to/filled_dem.tif')
        >>> print(f"Filled DEM saved at: {filled_dem_path}")

        To use the default output path for saving the filled DEM:

        >>> filled_dem_path = dem_preprocessor.fill_depressions()
        >>> print(f"Filled DEM saved at: {filled_dem_path}")
        """

        wbt = self.wbt
        name = self.meta['Name']

        if output_path is None:
            output_path = f'{self.output_dir}/{name}_filled.tif'

        wbt.fill_depressions(self.dem_preprocessing, os.path.abspath(output_path), fix_flats=True)

        return output_path

    def create_outlet(self, x, y, flow_accumulation_raster, snap_distance, output_path=None):
        """
         Create a shapefile of the pour point (outlet) using specified coordinates, and snap it to the nearest flow path.

         This method takes the pour point coordinates (x, y), creates a shapefile for the pour point, and snaps
         the point to the nearest flow path within the specified snapping distance using the flow accumulation raster.

         Parameters
         ----------
         x : float
             The x-coordinate (longitude or easting) of the pour point.
         y : float
             The y-coordinate (latitude or northing) of the pour point.
         flow_accumulation_raster : str
             The file path to the flow accumulation raster used for snapping the pour point.
         snap_distance : float
             The distance used to snap the pour point to the nearest flow path.
         output_path : str, optional
             The file path where the outlet shapefile will be saved. If not provided, the default path will be
             `'{output_dir}/{project_name}_outlet.shp'`.

         Returns
         -------
         str
             The absolute path to the saved shapefile of the snapped pour point.

         """
        pour_point_geometry = Point(x, y)
        pour_point_gdf = gpd.GeoDataFrame(geometry=[pour_point_geometry])
        pour_point_gdf.set_crs(epsg=self.meta['EPSG'], inplace=True)
        name = self.meta['Name']

        if output_path is None:
            output_path = f'{self.output_dir}/{name}_outlet.shp'

        pour_point_gdf.to_file(output_path)

        wbt = self.wbt
        wbt.snap_pour_points(os.path.abspath(output_path), os.path.abspath(flow_accumulation_raster),
                             os.path.abspath(output_path), snap_distance)

        return output_path

    def generate_flow_direction_raster(self, filled_dem, output_path=None):
        """
        Generate a flow direction raster using the D8 method and save it to the specified location.

        This method uses the WhiteboxTools library to create a flow direction raster based on the D8 flow algorithm.
        The generated raster is saved to the provided output path, or a default path based on the project metadata
        if no path is specified.

        Parameters
        ----------
        filled_dem : str
            The file path to the filled DEM (digital elevation model) that will be used to compute flow directions.
        output_path : str, optional
            The file path where the flow direction raster will be saved. If not provided, the default path will be
            `'{output_dir}/{project_name}_d8.tif'`.

        Returns
        -------
        str
            The absolute path to the saved flow direction raster.
        """
        wbt = self.wbt
        name = self.meta['Name']

        if output_path is None:
            output_path = f'{self.output_dir}/{name}_d8.tif'

        wbt.d8_pointer(os.path.abspath(filled_dem), os.path.abspath(output_path), esri_pntr=True)

        return output_path

    def generate_flow_accumulation_raster(self, flow_direction_raster, output_path=None):
        """
         Generate a flow accumulation raster using the flow direction raster obtained from the D8 method.

         This method creates a flow accumulation raster based on the flow direction raster, which was generated
         using the D8 method. The generated flow accumulation raster is saved to the provided output path, or
         a default path based on the project metadata if no path is specified.

         Parameters
         ----------
         flow_direction_raster : str
             The file path to the flow direction raster (D8 method) used to compute the flow accumulation.
         output_path : str, optional
             The file path where the flow accumulation raster will be saved. If not provided, the default path will be
             `'{output_dir}/{project_name}_flow_acc.tif'`.

         Returns
         -------
         str
             The absolute path to the saved flow accumulation raster.

        """
        wbt = self.wbt
        name = self.meta['Name']

        if output_path is None:
            output_path = f'{self.output_dir}/{name}_flow_acc.tif'

        wbt.d8_flow_accumulation(os.path.abspath(flow_direction_raster), os.path.abspath(output_path),
                                 pntr=True, esri_pntr=True)

        return output_path

    def generate_streams_raster(self, flow_accumulation_raster, threshold_area, output_path=None):
        """
        Generate a stream raster from the flow accumulation raster based on the specified stream threshold.

        This method creates a stream raster by extracting streams from the flow accumulation raster. The streams
        are identified using the provided threshold area. The generated stream raster is saved to the specified
        output path, or a default path based on the project metadata if no path is provided.

        Parameters
        ----------
        flow_accumulation_raster : str
            The file path to the flow accumulation raster used to extract streams.
        threshold_area : float
            The area threshold (in map units) used to define streams in the flow accumulation raster.
        output_path : str, optional
            The file path where the stream raster will be saved. If not provided, the default path will be
            `'{output_dir}/{project_name}_stream.tif'`.

        Returns
        -------
        str
            The absolute path to the saved stream raster.
        """
        wbt = self.wbt
        name = self.meta['Name']

        if output_path is None:
            output_path = f'{self.output_dir}/{name}_stream.tif'

        wbt.extract_streams(
            os.path.abspath(flow_accumulation_raster),
            os.path.abspath(output_path),
            threshold_area
        )

        return output_path

    def generate_watershed_mask(self, flow_direction_raster, pour_point_shp, output_path=None):
        """
         Delineate the watershed in the form of a raster using the pour point shapefile and the flow direction raster.

         This method uses the pour point shapefile and the flow direction raster (D8 flow direction) to delineate
         the watershed area. The result is saved as a raster, which can be stored in a specified location or a
         default location based on the project metadata if no path is provided.

         Parameters
         ----------
         flow_direction_raster : str
             The file path to the flow direction raster (D8 method) used to delineate the watershed.
         pour_point_shp : str
             The file path to the shapefile containing the pour point(s) used to define the watershed outlet.
         output_path : str, optional
             The file path where the watershed mask raster will be saved. If not provided, the default path will be
             `'{output_dir}/{project_name}_watershed_msk.tif'`.

         Returns
         -------
         str
             The absolute path to the saved watershed mask raster.

        """
        wbt = self.wbt
        name = self.meta['Name']

        if output_path is None:
            output_path = f'{self.output_dir}/{name}_watershed_msk.tif'

        wbt.watershed(os.path.abspath(flow_direction_raster), os.path.abspath(pour_point_shp),
                      os.path.abspath(output_path), esri_pntr=True)

        return output_path

    def generate_watershed_boundary(self, watershed_mask, output_path=None):
        """
           Generate the watershed boundary shapefile from the watershed raster.

           This method extracts the watershed boundary from a given watershed mask (raster) and saves the boundary
           as a shapefile. The watershed boundary is created by converting raster values to vector shapes, and the
           result is saved to a specified output path or a default path based on the project metadata if no path is provided.

           Parameters
           ----------
           watershed_mask : str
               The file path to the watershed mask raster, which is used to delineate the watershed boundary.
           output_path : str, optional
               The file path where the watershed boundary shapefile will be saved. If not provided, the default path will be
               `'{output_dir}/{project_name}_watershed_bound.tif'`.

        """
        name = self.meta['Name']

        if output_path is None:
            output_path = f'{self.output_dir}/{name}_watershed_bound.tif'

        with rasterio.open(watershed_mask) as src:
            image = src.read(1)  # Read the first band
            mask = image != src.nodata  # Create a mask for valid data values

            results = (
                {'properties': {'raster_val': v}, 'geometry': s}
                for i, (s, v) in enumerate(
                shapes(image, mask=mask, transform=src.transform))
            )

            # Convert the results to a GeoDataFrame
            geoms = list(results)
            gdf = gpd.GeoDataFrame.from_features(geoms)
            gdf = gdf.dissolve()
        gdf.set_crs(epsg=self.meta["EPSG"],inplace=True)
        gdf.to_file(output_path)

        return gdf, output_path

    def convert_stream_raster_to_vector(self, stream_raster, flow_direction_raster, output_path=None):
        """
        Create a shapefile of the stream network from the stream raster.

        This method converts a stream raster into a vector format (shapefile) using the flow direction raster.
        The resulting stream network is saved to the specified output path, or a default path based on the
        project metadata if no path is provided.

        Parameters
        ----------
        stream_raster : str
            The file path to the stream raster, which contains the stream network data.
        flow_direction_raster : str
            The file path to the flow direction raster (D8 method) used to convert the stream raster to vector format.
        output_path : str, optional
            The file path where the stream network shapefile will be saved. If not provided, the default path will be
            `'{output_dir}/{project_name}_streams.shp'`.

        Returns
        -------
        str
            The absolute path to the saved stream network shapefile.
        """
        wbt = self.wbt
        name = self.meta['Name']

        if output_path is None:
            output_path = f'{self.output_dir}/{name}_streams.shp'

        # Convert Stream Raster to Vector
        wbt.raster_streams_to_vector(
            os.path.abspath(stream_raster),
            os.path.abspath(flow_direction_raster),
            os.path.abspath(output_path),
            esri_pntr=True
        )

        return output_path

    def clip_rasters(self, raster_list, watershed_boundary, method='boundary', output_dir=None):
        """
        Clip a list of rasters using the watershed boundary polygon or extent.

        This method clips a list of rasters (e.g., filled DEM, flow direction raster, stream raster) using the
        provided watershed boundary polygon. The clipping can be performed based on the polygon boundary, the
        bounding box (extent), or both. The clipped rasters are saved to the specified output directory.

        Parameters
        ----------
        raster_list : list of str
            A list of file paths to the rasters that need to be clipped.
        watershed_boundary : str
            The file path to the watershed boundary shapefile, which is used to clip the rasters.
        method : str, optional
            The clipping method to use. Options are:
            - 'boundary': Clip rasters using the polygon boundary.
            - 'extent': Clip rasters using the bounding box of the polygon.
            - 'both': Perform both boundary and extent clipping.
            Default is 'boundary'.
        output_dir : str, optional
            The directory where the clipped rasters will be saved. If not provided, the default directory
            `{output_dir}/` will be used.

        Returns
        -------
        None
            This method saves the clipped rasters to the specified output directory.

        """

        if output_dir is None:
            output_dir = f'{self.output_dir}/'

        name = self.meta['Name']
        ws_bound = gpd.read_file(watershed_boundary)
        geometry = [unary_union(ws_bound.geometry)]
        bounds = ws_bound.total_bounds  # [minx, miny, maxx, maxy]

        for raster_file in raster_list:

            with rasterio.open(raster_file) as src:
                if method == 'boundary' or method == 'both':
                    output_raster = f'{output_dir}/{name}_clipped.tif'

                    clipped_data, clipped_transform = mask(src, geometry, crop=True)

                    clipped_profile = src.profile
                    clipped_profile.update({
                        'height': clipped_data.shape[1],
                        'width': clipped_data.shape[2],
                        'transform': clipped_transform
                    })

                    with rasterio.open(output_raster, 'w', **clipped_profile) as dst:
                        dst.write(clipped_data)

                if method == 'extent' or method == 'both':
                    output_raster = f'{output_dir}/{name}_clipped_ext.tif'

                    window = from_bounds(*bounds, transform=src.transform)

                    clipped_data = src.read(window=window)
                    clipped_transform = src.window_transform(window)
                    clipped_meta = src.meta.copy()

                    # Update the metadata with the new dimensions and transform
                    clipped_meta.update({
                        "driver": "GTiff",
                        "height": clipped_data.shape[1],
                        "width": clipped_data.shape[2],
                        "transform": clipped_transform
                    })

                    # Save the clipped raster to the output path
                    with rasterio.open(output_raster, "w", **clipped_meta) as dst:
                        dst.write(clipped_data)

    def clip_streamline(self, stream_shapefile, watershed_boundary, output_path=None):
        """
        Clip the streamlines shapefile by the watershed boundary polygon.

        This method clips the streamlines from the provided stream shapefile using the watershed boundary polygon.
        The clipped streamlines are saved to the specified output path, or a default path based on the project
        metadata if no path is provided.

        Parameters
        ----------
        stream_shapefile : str
            The file path to the streamlines shapefile to be clipped.
        watershed_boundary : str
            The file path to the watershed boundary shapefile, which will be used to clip the streamlines.
        output_path : str, optional
            The file path where the clipped streamlines shapefile will be saved. If not provided, the default path will be
            `'{output_dir}/{project_name}_streams.shp'`.

        Returns
        -------
        str
            The absolute path to the saved clipped streamlines shapefile.

        """

        name = self.meta['Name']

        if output_path is None:
            output_path = f'{self.output_dir}/{name}_streams.shp'

        lines = gpd.read_file(stream_shapefile)
        lines.set_crs(epsg=self.meta["EPSG"],inplace=True)
        watershed_boundary_t = gpd.read_file(watershed_boundary)
        clipped_lines = gpd.clip(lines, watershed_boundary_t)
        clipped_lines.set_crs(epsg=self.meta["EPSG"],inplace=True)
        clipped_lines.to_file(output_path)

        return output_path

    def extract_watershed_and_stream_network(self, outlet_path, boundary_path, output_streams_path, clean=True):
        """
        Extract the watershed and stream network from elevation data and save the results to specified paths.

        This method processes elevation data to extract both the watershed and stream network. It performs a series of operations, including filling depressions, generating flow direction and accumulation rasters, identifying streams, and creating/clipping the watershed boundary and stream network. The results are saved to the provided file paths. Optionally, temporary files and directories can be cleaned up after processing.

        Parameters
        ----------
        outlet_path : str
            The file path where the outlet raster will be saved.
        boundary_path : str
            The file path where the watershed boundary shapefile will be saved.
        output_streams_path : str
            The file path where the clipped stream network shapefile will be saved.
        clean : bool, optional
            If True, temporary files and directories will be cleaned up after processing. Defaults to True.

        Returns
        -------
        None

        Raises
        ------
        OSError
            If there is an issue creating directories or accessing files.
        ValueError
            If the input parameters are not correctly specified.

        """
        output_dir = self.output_dir

        if clean is True:
            temp = self.output_dir + '/temp'
            os.makedirs(temp, exist_ok=True)
            self.output_dir = temp

        filled = self.fill_depressions()
        d8_raster = self.generate_flow_direction_raster(filled)
        flow_acc = self.generate_flow_accumulation_raster(d8_raster)
        streams = self.generate_streams_raster(flow_acc, self.threshold_area)
        outlet = self.create_outlet(self.outlet[0], self.outlet[1], flow_acc, self.snap_distance,
                                    output_path=f'{outlet_path}')
        ws_mask = self.generate_watershed_mask(d8_raster, outlet)
        _, ws_bound = self.generate_watershed_boundary(ws_mask, output_path=f'{boundary_path}')
        stream_shp = self.convert_stream_raster_to_vector(streams, d8_raster)
        self.clip_rasters([filled], ws_bound, output_dir=output_dir, method='both')
        self.clip_streamline(os.path.abspath(stream_shp), ws_bound, output_path=f'{output_streams_path}')

        if clean is True:
            shutil.rmtree(temp)


class GenerateMesh:
    """
    A class for generating a locally refined triangular irregular network (TIN) mesh from raster data, watershed boundaries, and stream networks.

    This class performs several operations to generate the TIN mesh, including extracting raster and wavelet information,
    loading watershed boundaries, stream networks, and outlet points, and utilizing wavelet decomposition to refine the mesh
    and incorporate significant details.

    Parameters
    ----------
    path_to_raster : str
        Path to the raster file used for extracting elevation data and wavelet analysis.
    path_to_watershed : str
        Path to the shapefile containing watershed boundary data.
    path_to_stream_network : str
        Path to the shapefile containing stream network data.
    path_to_outlet : str
        Path to the shapefile containing outlet point data.
    maxlevel : int, optional
        Maximum level for wavelet decomposition. If None, the maximum level is determined from the data.

    Attributes
    ----------
    normalizing_coeff : float, optional
        Coefficient used for normalizing wavelet coefficients, initially None.
    raster : str
        Path to the raster file.
    maxlevel : int, optional
        Maximum level for wavelet decomposition.
    watershed : GeoDataFrame
        GeoDataFrame containing watershed boundaries.
    stream_network : GeoDataFrame
        GeoDataFrame containing stream network data.
    outlet : GeoDataFrame
        GeoDataFrame containing outlet points.

    """

    def __init__(self, path_to_raster, path_to_watershed, path_to_stream_network, path_to_outlet, maxlevel=None, feature_method='curvature'):
        self.wavelet_packet = None
        self.y_grid = None
        self.height = None
        self.width = None
        self.bounds = None
        self.transform = None
        self.data = None
        self.x_grid = None
        self.normalizing_coeff = None
        self.feature_method = feature_method
        self.raster = path_to_raster
        self.maxlevel = maxlevel
        self.extract_raster_and_wavelet_info()
        self.watershed = gpd.read_file(path_to_watershed)
        self.stream_network = gpd.read_file(path_to_stream_network)
        self.outlet = gpd.read_file(path_to_outlet)

    def extract_raster_and_wavelet_info(self):
        """
        Extracts information from a raster file and performs wavelet decomposition on the raster data.

        This method reads the first band of the raster file, extracts its spatial information
        (coordinates, transformation, bounds, width, and height), and computes the corresponding
        meshgrids of x and y coordinates. It then performs a 2D wavelet decomposition on the raster
        data and stores the resulting wavelet packet. The wavelet decomposition level is set or updated.

        Parameters
        ----------
        None

        Attributes
        ----------
        data : numpy.ndarray
            The first band of the raster file as a 2D array.
        transform : affine.Affine
            The affine transformation matrix for the raster file.
        bounds : rasterio.coords.BoundingBox
            The bounding box of the raster file.
        width : int
            The width of the raster in pixels.
        height : int
            The height of the raster in pixels.
        x_grid : numpy.ndarray
            The x-coordinates of the raster data as a meshgrid.
        y_grid : numpy.ndarray
            The y-coordinates of the raster data as a meshgrid, adjusted based on the transformation.
        wavelet_packet : pywt.WaveletPacket2D
            The wavelet packet object containing the decomposed data.
        maxlevel : int
            The maximum level for wavelet decomposition, updated if necessary.
        normalizing_coeff : float
            The coefficient used to normalize wavelet coefficients, computed by `find_max_maximum_coeffs()`.

        Raises
        ------
        None

        Notes
        -----
        The y-coordinates (`y_grid`) are flipped if the transformation requires it.
        """

        with rasterio.open(self.raster) as src:
            self.data = src.read(1)  # Read the first band
            self.transform = src.transform
            self.bounds = src.bounds
            self.width = src.width
            self.height = src.height

        cols = np.arange(self.width)
        rows = np.arange(self.height)

        # Compute the x and y coordinates
        x_coords = self.transform[0] * cols + self.transform[2]
        y_coords = self.transform[4] * rows + self.transform[5]

        # Create meshgrids
        self.x_grid, self.y_grid = np.meshgrid(x_coords, y_coords)

        # Adjust y_grid if necessary
        if self.transform[4] > 0:
            self.y_grid = np.flipud(self.y_grid)

        self.wavelet_packet = pywt.WaveletPacket2D(data=self.data, wavelet='db1',
                                                   maxlevel=self.maxlevel)
        # update maxlevel incase it's none
        self.maxlevel = self.wavelet_packet.maxlevel
        self.normalizing_coeff = self.find_max_maximum_coeffs()

    def get_extent(self):
        """
        Returns extent (xmin, xmax, ymin, ymax) of dem data used in mesh generation.
        """

        x_min = self.bounds.left
        x_max = self.bounds.right
        y_min = self.bounds.bottom
        y_max = self.bounds.top

        return x_min, x_max, y_min, y_max

    def find_max_average_coeffs(self):
        """
        Calculates the maximum average coefficient from the wavelet packet decomposition.

        This method computes the average of the absolute values of the vertical, horizontal, and diagonal
        wavelet coefficients at each level of the wavelet decomposition. It then finds and returns the maximum
        of these average coefficients across all decomposition levels.

        The method uses the `self.wavelet_packet` attribute, which should be an instance of `pywt.WaveletPacket2D`
        containing the decomposed data, and `self.maxlevel`, which defines the maximum level of decomposition.

        Returns
        -------
        float
            The maximum average coefficient across all levels of the wavelet decomposition.

        Raises
        ------
        AttributeError
            If `self.wavelet_packet` or `self.maxlevel` is not properly set.
        ValueError
            If there are issues with the wavelet coefficients or their dimensions.

        Notes
        -----
        The wavelet packet decomposition must be performed before calling this method. The method averages
        the absolute values of the coefficients for the vertical ('v'), horizontal ('h'), and diagonal ('d')
        details at each level of the decomposition.

        """

        max_avg_coeffs = []
        for level in range(1, self.maxlevel + 1):
            v = self.wavelet_packet['v' * level].data
            h = self.wavelet_packet['h' * level].data
            d = self.wavelet_packet['d' * level].data
            avg_coeffs = np.mean(np.abs([v, h, d]), axis=0)
            max_avg_coeffs.append(np.max(avg_coeffs))
        return max(max_avg_coeffs)

    def find_max_maximum_coeffs(self):
        """
        Calculates the maximum coefficient from the wavelet packet decomposition.

        This method computes the maximum of the absolute values of the vertical, horizontal, and diagonal
        wavelet coefficients at each level of the wavelet decomposition. It then finds and returns the maximum
        of these coefficients across all decomposition levels.

        The method uses the `self.wavelet_packet` attribute, which should be an instance of `pywt.WaveletPacket2D`
        containing the decomposed data, and `self.maxlevel`, which defines the maximum level of decomposition.

        Returns
        -------
        float
            The maximum coefficient across all levels of the wavelet decomposition.

        Raises
        ------
        AttributeError
            If `self.wavelet_packet` or `self.maxlevel` is not properly set.
        ValueError
            If there are issues with the wavelet coefficients or their dimensions.

        Notes
        -----
        The wavelet packet decomposition must be performed before calling this method. The method computes the
        maximum of the absolute values of the coefficients for the vertical ('v'), horizontal ('h'), and diagonal ('d')
        details at each level of the decomposition.

        """
        max_coeffs = []
        for level in range(1, self.maxlevel + 1):
            v = self.wavelet_packet['v' * level].data
            h = self.wavelet_packet['h' * level].data
            d = self.wavelet_packet['d' * level].data
            avg_coeffs = np.max(np.abs([v, h, d]), axis=0)
            max_coeffs.append(np.max(avg_coeffs))
        return max(max_coeffs)

    def extract_points_from_significant_details(self, threshold):
        """
        Extracts significant points from wavelet decomposition details and processes them.

        This method identifies significant details from wavelet decomposition at various levels, filters closely
        spaced points, and generates additional points along stream paths. It combines these points with boundary
        codes and interpolated elevations and returns the final set of points along with a buffered watershed geometry.

        Parameters
        ----------
        threshold : float
            The minimum threshold for identifying significant details in the wavelet decomposition.

        Returns
        -------
        tuple of (numpy.ndarray, geometry)
            - numpy.ndarray : Array of points with their x and y coordinates, elevations, and boundary codes.
            - geometry : Buffered watershed geometry, which can be used for further spatial analysis.

        Raises
        ------
        AttributeError
            If `self.wavelet_packet`, `self.maxlevel`, or other required attributes are not properly set.
        ValueError
            If there are issues with the processing or filtering of points.

        Notes
        -----
        This method first identifies the significant detail points in the wavelet decomposition at various levels,
        then removes any points closer than the raster resolution. It generates points along stream paths and
        combines all points with interpolated elevations and boundary codes. The method ensures that the points are
        unique and returns them along with a buffered watershed geometry.

        Examples
        --------
        >>> points, buffered_watershed = obj.extract_points_from_significant_details(threshold=0.5)
        >>> print(points)
        [[x1, y1, elevation1, boundary_code1], [x2, y2, elevation2, boundary_code2], ...]

        >>> print(buffered_watershed)
        <Polygon geometry>
        """

        centers = set()
        for level in range(1, self.maxlevel + 1):
            centers.update(self.process_level(level, threshold))

        centers = np.array(list(centers))

        # remove any points closer to each other than the resolution of the DEM
        centers = self.remove_close_points(centers,self.transform[0]) # resolution of raster
        centers = np.array(list(centers))


        stream_points = self.generate_points_along_stream(centers)
        stream_code = np.ones(len(stream_points)) * 3

        med_distance, max_distance = self.distance_to_nearest_n(centers, n=6)

        centers, boundary_codes, buffered_watershed = self.filter_coords_within_geometry(centers, med_distance)

        centers = np.vstack((centers, stream_points))  # stream_points
        boundary_codes = np.hstack((boundary_codes, stream_code))  # stream_code

        unique_centers, unique_indices = np.unique(centers, axis=0, return_index=True)
        centers = unique_centers
        boundary_codes = boundary_codes[unique_indices]

        elevations = self.interpolate_elevations(centers)

        return np.column_stack((centers, elevations, boundary_codes)), buffered_watershed

    def filter_coords_within_geometry(self, coords, buffer_distance):
        """
        Filters and categorizes coordinates based on their position relative to a buffered watershed geometry.

        Parameters
        ----------
        coords : numpy.ndarray
            An array of shape (N, 2) representing coordinates to be filtered.
        buffer_distance : float
            The distance to buffer the watershed geometry.

        Returns
        -------
        all_coords : numpy.ndarray
            An array of filtered coordinates that include original, boundary, and inner boundary points.
        all_boundary_codes : numpy.ndarray
            An array of boundary codes corresponding to the filtered coordinates. Codes indicate whether a coordinate
            is part of the original watershed (0), the outer boundary (1), or an updated outlet point (2).
        buffered_watershed : shapely.geometry.Polygon
            The buffered version of the original watershed geometry used for filtering.

        Notes
        -----
        The function processes the original watershed geometry by applying a specified buffer distance and generates boundary points.
        It finds coordinates within the original watershed and boundary points outside of it, adjusts coordinates based on proximity
        to the updated outlet, and categorizes coordinates with boundary codes.

        - Coordinates within the original watershed are assigned a boundary code of 0.
        - Coordinates on the outer boundary are assigned a boundary code of 1.
        - The outlet is adjusted to the nearest boundary point and given a boundary code of 2.
        """

        def find_nearest_boundary_point(point, polygon):
            # If the point is inside the polygon
            if polygon.contains(point):
                # Compute the nearest point on the boundary
                boundary_point = polygon.exterior.interpolate(polygon.exterior.project(point))
                return boundary_point
            else:
                # If the point is outside, find the nearest point on the polygon
                nearest = nearest_points(point, polygon)
                return nearest[1]

        original_watershed = self.watershed.geometry.iloc[0]

        inner_buffered_watershed = original_watershed.buffer(1e-6)

        buffered_watershed = original_watershed.buffer(buffer_distance)

        num_inner_boundary_points = int(ceil(inner_buffered_watershed.length / buffer_distance))

        num_boundary_points = int(ceil(buffered_watershed.length / buffer_distance))

        inner_boundary_coords = np.array(list(
            inner_buffered_watershed.exterior.interpolate(i / num_inner_boundary_points, normalized=True).coords[0] for
            i in
            range(num_inner_boundary_points)))

        inner_boundary_codes = np.zeros(inner_boundary_coords.shape[0], dtype=int)

        boundary_coords = np.array(list(
            buffered_watershed.exterior.interpolate(i / num_boundary_points, normalized=True).coords[0] for i in
            range(num_boundary_points)))

        boundary_codes = np.ones(boundary_coords.shape[0], dtype=int)

        within_original = contains(original_watershed, coords[:, 0], coords[:, 1])

        # because the buffered watershed is altered you can have closed nodes within the watershed
        bcoords_within_orignal = contains(original_watershed, boundary_coords[:, 0], boundary_coords[:, 1])

        boundary_coords = boundary_coords[~bcoords_within_orignal, :]
        boundary_codes = boundary_codes[~bcoords_within_orignal]

        original_outlet = self.outlet.geometry.iloc[0]

        # Find the nearest point on the polygon boundary
        updated_outlet = find_nearest_boundary_point(original_outlet, buffered_watershed)

        x = updated_outlet.x
        y = updated_outlet.y

        out_points = [[x, y]]
        out_code = 2

        buffer_outlet = updated_outlet.buffer(buffer_distance)
        bcoords_within_outlet = contains(buffer_outlet, boundary_coords[:, 0], boundary_coords[:, 1])
        inner_bcoords_within_outlet = contains(buffer_outlet, inner_boundary_coords[:, 0], inner_boundary_coords[:, 1])

        all_coords = np.vstack([coords[within_original, :], boundary_coords[~bcoords_within_outlet],
                                inner_boundary_coords[~inner_bcoords_within_outlet], out_points])
        all_boundary_codes = np.hstack(
            [np.zeros(np.sum(within_original), dtype=int), boundary_codes[~bcoords_within_outlet],
             inner_boundary_codes[~inner_bcoords_within_outlet], out_code])

        return all_coords, all_boundary_codes, buffered_watershed

    def process_level(self, level, threshold):
        """
         Processes a specific level of wavelet decomposition to extract significant detail points using gradient or curvature.

         This method examines the vertical, horizontal, and diagonal coefficients at the specified wavelet level,
         identifies significant points based on a given threshold, and processes the points using either gradient or
         curvature to find the most significant detail within each raster cell. If the cell is too small for gradient
         or curvature computation, the centroid of the cell is used as the significant point.

         Parameters
         ----------
         level : int
             The level of wavelet decomposition to process.
         threshold : float
             The threshold value used to filter significant coefficients.

         Returns
         -------
         iterator of tuples
             An iterator of tuples containing the x and y coordinates of significant points.

         Raises
         ------
         AttributeError
             If `self.wavelet_packet`, `self.data`, `self.x_grid`, or `self.y_grid` is not properly set.

         Notes
         -----
         The method processes each raster cell to find significant points by either gradient, curvature, or centroid,
         depending on the `self.feature_method` attribute. The `self.feature_method` can be set to 'gradient',
         'curvature', or default to 'centroid' if the method is not specified. The function filters coefficients
         based on the normalized wavelet coefficients and a threshold value, ensuring only significant points are extracted.

         """

        v = self.wavelet_packet['v' * level].data
        h = self.wavelet_packet['h' * level].data
        d = self.wavelet_packet['d' * level].data

        norm_coeffs = np.maximum.reduce([np.abs(v), np.abs(h), np.abs(d)]) / self.normalizing_coeff
        sig_mask = norm_coeffs > (2 ** (-level + 1) * threshold)

        sig_indices = np.argwhere(sig_mask)

        significant_points = []

        for idx in sig_indices:
            row, col = idx

            # Compute the spatial extent of the cell
            rows_per_cell = self.height // v.shape[0]
            cols_per_cell = self.width // v.shape[1]

            row_start = row * rows_per_cell
            row_end = (row + 1) * rows_per_cell
            col_start = col * cols_per_cell
            col_end = (col + 1) * cols_per_cell

            # Handle edge cases where dimensions are not evenly divisible
            row_end = min(row_end, self.height)
            col_end = min(col_end, self.width)

            # Extract DEM data and coordinate grids within the cell
            dem_cell = self.data[row_start:row_end, col_start:col_end]
            x_cell = self.x_grid[row_start:row_end, col_start:col_end]
            y_cell = self.y_grid[row_start:row_end, col_start:col_end]

            if dem_cell.size > 0:
                # Check if dem_cell is large enough for gradient computation
                if dem_cell.shape[0] >= 2 and dem_cell.shape[1] >= 2:
                    if self.feature_method == 'gradient':
                        grad_y, grad_x = np.gradient(dem_cell)
                        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
                        max_idx = np.unravel_index(np.argmax(grad_mag), grad_mag.shape)

                    elif self.feature_method == 'curvature':
                        laplacian = ndimage.laplace(dem_cell)
                        max_idx = np.unravel_index(np.argmax(np.abs(laplacian)), laplacian.shape)

                    else:
                        # Default to centroid
                        max_idx = (dem_cell.shape[0] // 2, dem_cell.shape[1] // 2)
                else:
                    # dem_cell is too small; use centroid
                    max_idx = (dem_cell.shape[0] // 2, dem_cell.shape[1] // 2)

                x_coord = x_cell[max_idx]
                y_coord = y_cell[max_idx]
                significant_points.append((x_coord, y_coord))
            else:

                x_coord = x_cell.mean()
                y_coord = y_cell.mean()
                significant_points.append((x_coord, y_coord))

        return significant_points

    @staticmethod
    def remove_close_points(points, threshold):
        """
        Removes points that are closer than a specified threshold.
        """
        # Convert list of points to a numpy array
        points = np.array(points)

        # Compute pairwise distances
        dist_matrix = distance.cdist(points, points)

        # Create a boolean mask for distances below the threshold (excluding diagonal)
        mask = (dist_matrix < threshold) & (dist_matrix > 0)

        # Get indices of points to remove
        to_remove = set()
        for i in range(len(points)):
            if i not in to_remove:
                close_points = np.where(mask[i])[0]
                to_remove.update(close_points)

        # Remove points
        result = [tuple(points[i]) for i in range(len(points)) if i not in to_remove]

        return result

    @staticmethod
    def distance_to_nearest_n(points, n=6):
        """
        Calculates the average distance to the nearest `n` points for each point in a list.

        This function uses a KD-tree to efficiently compute the distances between points and determine the
        distances to the `n` nearest neighbors for each point. It returns the median and maximum distances to
        these neighbors.

        Parameters
        ----------
        points : list of tuples
            A list of (x, y) coordinates representing the points to analyze.
        n : int, optional
            The number of nearest neighbors to consider for each point. Defaults to 6.

        Returns
        -------
        tuple of (float, float)
            - float : The median distance to the nearest `n` points for each point.
            - float : The maximum distance to the nearest `n` points for each point.

        Raises
        ------
        ValueError
            If the input `points` is empty or if `n` is less than 1.
        TypeError
            If `points` is not a list of tuples or if `n` is not an integer.

        Notes
        -----
        This method computes distances using a KD-tree, ensuring efficient calculations even with large datasets.
        It calculates the distance to the nearest `n` points for each point in the input list, then returns the
        median and maximum distances across all points.

        """
        points_array = np.array(points)
        tree = cKDTree(points_array)
        distances, _ = tree.query(points_array, k=n + 1)

        median_distance = np.max(np.median(distances[:, :], axis=0))
        max_distance = np.max(np.max(distances[:, :], axis=0))

        return median_distance, max_distance

    def convert_coords_to_mesh(self, coords):
        """
        Converts a set of coordinates into a 2D mesh using Delaunay triangulation.

        This method takes an array of coordinates and converts them into a 2D mesh. The input coordinates
        include x, y, and z values, along with an optional boundary code. While the boundary codes label
        points in the mesh, they currently do not influence the mesh creation.

        Parameters
        ----------
        coords : numpy.ndarray
            An array where each row represents a point with x, y, z coordinates and a boundary code.

        Returns
        -------
        pyvista.PolyData
            A 2D mesh generated from the input coordinates using Delaunay triangulation.

        Raises
        ------
        IndexError
            If the input `coords` array does not have at least four columns.
        ValueError
            If the input `coords` array is empty or has incorrect dimensions.

        Notes
        -----
        The boundary codes provided in the `coords` array are stored with the mesh but do not currently affect
        the triangulation process. Elevation data (z-values) are included in the mesh as a scalar field, which
        could be used for subsequent analysis.

        """

        points = coords[:, :3]  # First three columns are x, y, z
        boundary_codes = coords[:, 3]  # Fourth column is boundary code

        point_cloud = pv.PolyData(points)

        elevation = points[:, 2]
        # elevation[boundary_codes == 1] = np.nan  # Set elevation to NaN where boundary code is 1
        point_cloud['Elevation'] = elevation
        point_cloud['BoundaryCode'] = boundary_codes

        mesh = point_cloud.delaunay_2d()

        return mesh

    def interpolate_elevations(self, points):
        """
        Interpolates elevation values for a set of geographic coordinates.

        This method uses a regular grid interpolator to estimate elevation values at specified geographic
        coordinates based on a given elevation data grid. It performs linear interpolation and handles
        coordinates that fall outside the bounds of the grid by returning NaN.

        Parameters
        ----------
        points : numpy.ndarray
            An array of points where each row contains x and y coordinates for which elevation values are
            to be interpolated.

        Returns
        -------
        numpy.ndarray
            An array of interpolated elevation values corresponding to the input coordinates.

        Raises
        ------
        ValueError
            If `points` does not have exactly two columns (x and y coordinates).
        AttributeError
            If `self.data` (the elevation data grid) or `self.transform` (the affine transformation matrix)
            is not properly set.

        Notes
        -----
        The interpolation is performed using a regular grid interpolator with linear interpolation. Points
        outside the bounds of the elevation grid will return NaN as their elevation value. The input `points`
        array must contain two columns, representing the x and y coordinates, respectively.
        """

        height, width = self.data.shape

        x = np.arange(width) * self.transform[0] + self.transform[2]
        y = np.arange(height) * self.transform[4] + self.transform[5]

        interpolator = RegularGridInterpolator((y, x), self.data, method='linear', bounds_error=False,
                                               fill_value=None)

        elevations = interpolator((points[:, 1], points[:, 0]))

        return elevations

    def generate_points_along_stream(self, coords):
        """
        Generates points along a stream network ensuring they are sufficiently spaced from each other.

        This method computes points along the stream network by interpolating positions at regular intervals
        based on the resolution of the DEM (Digital Elevation Model). It ensures that these points are spaced
        sufficiently from each other and from existing interior points by checking their distances from both the
        stream network and any interior points.

        Parameters
        ----------
        coords : numpy.ndarray
            An array of coordinates representing the existing points in the interior.

        Returns
        -------
        list of lists
            A list of points along the stream network, ensuring they are not too close to existing interior points.

        Raises
        ------
        ValueError
            If the input `coords` is empty or has incorrect dimensions.

        Notes
        -----
        The stream network is traversed, and points are interpolated along each stream line at regular intervals
        based on the DEM resolution. A KDTree is used to check distances between stream points and existing
        interior points, ensuring the generated stream points are not too close to each other or to the interior points.

        """

        stream = self.stream_network
        dem_res = self.transform[0]

        # Initialize the interior KDTree
        interior_tree = cKDTree(coords)  # Assumes coords is a NumPy array

        final_stream_pts = []

        for idx, row in stream.iterrows():
            # Extract line and compute points
            line = row['geometry']
            length = line.length
            num_points = int(ceil(length / dem_res))
            xy = [line.interpolate(i * dem_res) for i in range(num_points + 1)]
            stream_xy = np.array([[p.x, p.y] for p in xy])

            # Setup KDTree for stream points
            stream_tree = cKDTree(stream_xy)

            # Initialize stream points
            begin, end = stream_xy[0, :], stream_xy[-1, :]
            working_stream_pts = [begin, end]
            temp_stream_pts = [begin]

            # Check and remove close interior points
            dis_check, idx = stream_tree.query(coords, k=1)
            close_interior_pts = np.where(dis_check <= dem_res)[0]

            if len(close_interior_pts) > 0:
                coords_list = coords.tolist()
                for i in sorted(close_interior_pts, reverse=True):
                    temp_stream_pts.append(coords_list.pop(i))
                coords = np.array(coords_list)
                interior_tree = cKDTree(coords)

            # Initial conditions for while loop
            distance = np.linalg.norm(working_stream_pts[0] - working_stream_pts[1])
            dis_interior, _ = interior_tree.query(working_stream_pts[0], k=1)
            dis_stream, idx = stream_tree.query(working_stream_pts[0], k=len(stream_xy))
            idx_last = 0

            # Iterate to find additional stream points
            while distance > dis_interior:
                ids = idx[dis_stream < dis_interior]
                if len(ids) == 0:
                    break
                id_next = ids[-1]

                if id_next < idx_last:
                    id_next = ids[ids > idx_last][-1]

                temp_stream_pts.append(stream_xy[id_next])
                working_stream_pts = [stream_xy[id_next], end]
                idx_last = id_next

                distance = np.linalg.norm(working_stream_pts[0] - working_stream_pts[1])
                dis_interior, _ = interior_tree.query(working_stream_pts[0], k=1)
                dis_stream, idx = stream_tree.query(working_stream_pts[0], k=len(stream_xy))

            temp_stream_pts.append(end)
            final_stream_pts.extend(temp_stream_pts)

        return final_stream_pts

    @staticmethod
    def convert_points_to_gdf(points):
        """Helper function for converting points generated from `extract_points_from_significant_details` to a geopandas
        data frame."""

        df = pd.DataFrame(points, columns=['x', 'y', 'elevation', 'bc'])
        df['geometry'] = df.apply(lambda row: Point(row['x'], row['y']), axis=1)
        gdf = gpd.GeoDataFrame(df, geometry='geometry')
        gdf = gdf.drop(columns=['x', 'y'])
        return gdf

    @staticmethod
    def write_point_file(gdf, output):
        """Helper function for writing out points file."""
        InOut.write_point_file(gdf, output)

    @staticmethod
    def plot_mesh(mesh, scalar=None, **kwargs):
        """Helper function for plotting a PyVista mesh."""
        plotter = Shared.plot_mesh(mesh, scalar, **kwargs)
        return plotter

    @staticmethod
    def generate_meshbuild_input_file(filename, base_name, point_filename=None, mesh_filename=None):
        """
        Generates the input file for the meshbuilder executable.

        Handles two modes: generating a mesh from points (provide 'point_filename')
        or processing a pre-existing mesh (provide 'existing_mesh_filename').

        Parameters
        ----------
        filename : str
            Path where the input file will be written.
        out_filename : str
            The base name for meshbuilder output files (OUTFILENAME).
        params : dict
            Dictionary of meshbuilder parameters. Expected keys:
            'velocity_ratio', 'baseflow', 'velocity_coef', 'flow_exp'.
        point_filename : str, optional
            Filename of the .points file for mesh generation.
        existing_mesh_filename : str, optional
            Filename of an existing mesh to be processed.
        """
        # Input Validation
        if not (point_filename or mesh_filename):
            raise ValueError("Must provide either 'point_filename' or 'existing_mesh_filename'.")
        if point_filename and mesh_filename:
            raise ValueError("Provide either 'point_filename' or 'existing_mesh_filename', not both.")

        # Build the file content
        lines = []
        lines.append(f"VELOCITYRATIO:\n1.2")
        lines.append(f"BASEFLOW:\n0.2")
        lines.append(f"VELOCITYCOEF:\n60")
        lines.append(f"FLOWEXP:\n0.3")
        lines.append(f"OUTFILENAME:\n{base_name}")

        if point_filename:
            lines.append(f"POINTFILENAME:\n{point_filename}")
            lines.append("OPTMESHINPUT:\n8")
        else: # mesh_filename must be set due to validation above
            lines.append(f"INPUTDATAFILE:\n{mesh_filename}")
            lines.append("OPTMESHINPUT:\n1")
        
        # Write the file
        with open(filename, 'w') as file:
            file.write('\n'.join(lines) + '\n')

    @staticmethod
    def partition_mesh(volume, partition_args):
        """
        Partitions a mesh and produces a .reach file for parallel execution with tRIBS.

        This function handles the partitioning of a mesh by interacting with Docker to build and process
        the mesh based on the provided input files and partitioning parameters. It performs the following steps:

        - Changes the working directory to the specified volume containing the mesh files.
        - Initializes and runs a Docker container to perform the mesh partitioning.
        - Executes the mesh partitioning workflow with the provided arguments.
        - Cleans up the Docker container and restores the working directory.

        Parameters
        ----------
        volume : str
            Path to the directory containing the .in and .points files for the mesh.
        partition_args : list
            A list of arguments for partitioning:
                - str : The name of the input file.
                - int : The number of nodes for partitioning.
                - int : The partition method (1-3).
                - str : The basename for output files.

        Returns
        -------
        None

        Notes
        -----
        This function interacts with Docker to automate the mesh partitioning process for parallel execution
        with tRIBS. It switches to the specified working directory, initializes a Docker container, runs the
        mesh partitioning workflow, and then restores the original working directory after cleaning up.
        """
        current = os.getcwd()
        os.chdir(volume)

        meshbuild = MeshBuilderDocker(os.getcwd())
        meshbuild.start_docker_desktop()
        meshbuild.pull_image()
        meshbuild.run_container()
        meshbuild.execute_meshbuild_workflow(*partition_args)
        meshbuild.cleanup_container()
        meshbuild.clean_directory()

        os.chdir(current)
