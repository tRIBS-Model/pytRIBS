import os
import shutil
import collections

import pandas as pd
import triangle as tr
from rasterio.windows import from_bounds
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
from scipy.spatial import distance
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from shapely.vectorized import contains
from shapely.ops import nearest_points, snap, linemerge
from shapely.geometry import LineString

from math import ceil
from rasterio.mask import mask
import numpy as np
import geopandas as gpd
from rasterio.features import shapes
from whitebox import WhiteboxTools
from shapely.geometry import Point
import rasterio
import pywt
from shapely.ops import unary_union

from pytRIBS.shared.inout import InOut

from pytRIBS.shared.shared_mixin import Meta, Shared
from pytRIBS.mesh.run_docker import MeshBuilderDocker


# Geometry helpers for PSLG-constrained mesh generation
#
# Pure functions used by MeshFromPSLG to normalise coordinates, clean and
# resample stream breaklines, and assemble a planar straight-line graph (PSLG)
# of de-duplicated stream nodes and segments prior to triangulation.

def _normalize_coords(xy, target_span=1e4):
    """Shift to a local origin and scale so the largest span is ~target_span.

    Triangle is sensitive to coordinate magnitude; normalising improves the
    numerical robustness of the triangulation. Returns the normalised
    coordinates along with the origin and scale needed to invert the transform.
    """
    xy = np.asarray(xy, float)
    origin = xy.min(axis=0)
    shifted = xy - origin
    span = shifted.max(axis=0)
    scale = max(span.max() / target_span, 1.0)
    return shifted / scale, origin, scale


def _denormalize_coords(xy_norm, origin, scale):
    """Invert :func:`_normalize_coords`, mapping back to real-world coordinates."""
    return np.asarray(xy_norm) * scale + origin


def _cumlens(coords):
    """Return cumulative arc length and per-segment lengths along a polyline."""
    d = np.diff(coords, axis=0)
    seg = np.sqrt((d * d).sum(axis=1))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    return s, seg


def _remove_spikes(coords, angle_tol_deg):
    """Drop interior vertices that are nearly collinear or fully reversed.

    Vertices whose incoming/outgoing segments are almost parallel
    (cos ~ +1) or anti-parallel (cos ~ -1) add no shape information and can
    create degenerate triangulation constraints, so they are removed.
    """
    if len(coords) <= 2:
        return coords
    keep = [coords[0]]
    for i in range(1, len(coords) - 1):
        a, b, c = coords[i - 1], coords[i], coords[i + 1]
        u, v = b - a, c - b
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        if nu == 0 or nv == 0:
            continue
        cosang = float(np.dot(u, v) / (nu * nv))
        if not (cosang > 0.9999 or cosang < -0.9999):
            keep.append(b)
    keep.append(coords[-1])
    return np.asarray(keep)


def _node_streams(lines, tau_snap=1.0):
    """Snap near-coincident vertices and node all intersections once."""
    ml = linemerge(unary_union(snap(unary_union(lines), unary_union(lines), tau_snap)))
    if isinstance(ml, LineString):
        return [ml]
    return list(ml.geoms)


def _resample_with_min_spacing(line, spacing, m_min, angle_tol_deg=2.0, keep_original_vertices=False):
    """Resample a line to a target spacing while enforcing a minimum spacing.

    Spikes are optionally removed first, then samples are placed along the line
    at ``spacing`` intervals (plus the endpoints, and optionally the original
    vertices). Consecutive samples closer than ``m_min`` are dropped, and the
    final endpoint is always preserved.
    """
    coords = np.asarray(line.coords, float)
    if coords.ndim == 2 and coords.shape[1] > 2:
        coords = coords[:, :2]
    coords = _remove_spikes(coords, angle_tol_deg) if angle_tol_deg > 0 else coords
    s, seg = _cumlens(coords)
    L = float(s[-1])
    if L == 0:
        return coords[:1]

    s_grid = np.arange(0.0, L + 1e-9, spacing)
    s_orig = s if keep_original_vertices else np.array([0.0, L])
    all_s = np.unique(np.round(np.concatenate([s_orig, s_grid]), 9))

    kept = []
    seg_idx = 0
    for t in all_s:
        while seg_idx < len(seg) - 1 and s[seg_idx + 1] < t - 1e-12:
            seg_idx += 1
        k = np.where(np.isclose(s, t, rtol=0, atol=1e-9))[0]
        if k.size:
            pt = coords[k[0]]
        else:
            t0, t1 = s[seg_idx], s[seg_idx + 1]
            if t1 <= t0:
                continue
            a = (t - t0) / (t1 - t0)
            pt = coords[seg_idx] * (1 - a) + coords[seg_idx + 1] * a
        if not kept or np.linalg.norm(pt - kept[-1]) >= m_min - 1e-12:
            kept.append(pt)

    if np.linalg.norm(kept[-1] - coords[-1]) > 1e-9:
        kept.append(coords[-1])
    out = [kept[0]]
    for p in kept[1:]:
        if np.linalg.norm(p - out[-1]) >= m_min - 1e-12:
            out.append(p)
    if np.linalg.norm(out[-1] - coords[-1]) > 1e-9:
        out.append(coords[-1])
    return np.asarray(out)


def _build_stream_pslg(resampled_edges, m_min, dedupe_radius):
    """Assemble resampled stream edges into a de-duplicated PSLG.

    Vertices within ``dedupe_radius`` of one another are merged via a union-find
    so that intersecting/branching streams share nodes. Returns the unique node
    coordinates and the list of (i, j) segment index pairs, with degenerate
    (near-zero-length) segments discarded.
    """
    clean_edges = []
    for arr in (resampled_edges or []):
        if arr is None:
            continue
        a = np.asarray(arr, float)
        if a.ndim != 2 or a.shape[0] < 2:
            continue
        if a.shape[1] > 2:
            a = a[:, :2]
        clean_edges.append(a)

    if not clean_edges:
        return np.empty((0, 2), dtype=float), []

    P = np.vstack(clean_edges)
    kdt = cKDTree(P)
    pairs = kdt.query_pairs(r=max(float(dedupe_radius), 1e-9))

    parent = np.arange(len(P))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, j in pairs:
        union(i, j)

    rep_to_new = {}
    new_index = np.empty(len(P), dtype=int)
    uniq = []
    for i in range(len(P)):
        r = find(i)
        if r not in rep_to_new:
            rep_to_new[r] = len(uniq)
            uniq.append(P[r])
        new_index[i] = rep_to_new[r]
    uniq = np.asarray(uniq, float)

    segs = set()
    cursor = 0
    for arr in clean_edges:
        n = len(arr)
        idxs = new_index[cursor:cursor + n]
        cursor += n
        prev = idxs[0]
        for cur in idxs[1:]:
            if cur != prev:
                segs.add((min(prev, cur), max(prev, cur)))
            prev = cur

    tiny = max(0.5, 0.25 * float(dedupe_radius))
    segments = [(i, j) for i, j in segs if np.linalg.norm(uniq[i] - uniq[j]) >= tiny - 1e-12]
    return uniq, segments


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

        wbt.fill_depressions_wang_and_liu(self.dem_preprocessing, os.path.abspath(output_path), fix_flats=True)

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
        ws_gdf, ws_bound = self.generate_watershed_boundary(ws_mask, output_path=f'{boundary_path}')
        stream_shp = self.convert_stream_raster_to_vector(streams, d8_raster)
        self.clip_rasters([filled], ws_bound, output_dir=output_dir, method='both')
        self.clip_streamline(os.path.abspath(stream_shp), ws_bound, output_path=f'{output_streams_path}')

        if clean is True:
            shutil.rmtree(temp)

        return ws_gdf


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
            nodata = src.nodata

        # Build a nodata-free copy for the wavelet/feature analysis. Nodata cells (often
        # a large sentinel such as -999999) otherwise create an enormous terrain->nodata
        # "cliff" that dominates the significance normalization, hiding all real interior
        # relief and forcing every significant point onto the data edge (the watershed
        # boundary). self.data is left raw for display/elevation reference.
        self.data_filled = self.data.astype(float)
        invalid = ~np.isfinite(self.data_filled)
        if nodata is not None:
            invalid |= (self.data_filled == nodata)
        if invalid.all():
            raise ValueError(f"DEM '{self.raster}' contains no valid data.")
        if invalid.any():
            # Replace each nodata cell with its nearest valid value (no artificial cliff)
            idx = ndimage.distance_transform_edt(invalid, return_distances=False,
                                                 return_indices=True)
            self.data_filled = self.data_filled[tuple(idx)]

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

        self.wavelet_packet = pywt.WaveletPacket2D(data=self.data_filled, wavelet='db1',
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

            # Extract DEM data and coordinate grids within the cell (nodata-filled, so
            # the feature location is not pulled to a terrain->nodata edge)
            dem_cell = self.data_filled[row_start:row_end, col_start:col_end]
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

        interpolator = RegularGridInterpolator((y, x), self.data_filled, method='linear', bounds_error=False,
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


class MeshFromPSLG:
    """
    Generate a complete tRIBS TIN by triangulating a planar straight-line graph (PSLG).

    This is the primary mesh-generation strategy in pytRIBS. Rather than handing a
    cloud of points to tRIBS/meshbuilder and letting it triangulate, this class
    builds an explicit PSLG — a buffered watershed boundary ring, the stream
    network as breaklines, and an outlet reach connecting them — and triangulates
    it directly with Shewchuk's Triangle (the :mod:`triangle` package). Quality and
    area constraints add interior Steiner points as needed, but the ``YY`` switch
    forbids Steiner points on any PSLG segment so every boundary and stream node is
    a preserved constraint vertex.

    After triangulation it samples elevations from the DEM, enforces monotonic
    descent along the stream network (so tRIBS ``FlowDirs`` does not mark stream
    nodes as sinks), and assigns tRIBS boundary codes. The result can be written
    directly as the four tRIBS mesh files (``.nodes``/``.edges``/``.tri``/``.z``)
    via :meth:`write`, bypassing the meshbuilder.

    Interior terrain points are supplied by the caller (typically the code-0 points
    selected by :class:`GenerateMesh`), keeping point selection and triangulation
    decoupled and avoiding a ``.points`` file round-trip.

    Parameters
    ----------
    interior_points : numpy.ndarray or pandas.DataFrame
        Interior (code-0) terrain points. An ndarray of shape (N, 2+) whose first
        two columns are x, y, or a DataFrame with 'x' and 'y' columns. Any extra
        columns are ignored.
    watershed : geopandas.GeoDataFrame or shapely geometry
        Watershed polygon used to derive the buffered no-flow boundary.
    stream_network : geopandas.GeoDataFrame
        Stream network line features used as breaklines.
    dem_file : str
        Path to the DEM raster used for elevation sampling and resolution guidance.
    outlet : geopandas.GeoDataFrame, geopandas.GeoSeries, shapely Point, optional
        Spatial hint for locating the stream outlet among the network's degree-1
        endpoints. If None, the endpoint nearest the watershed boundary is used.
    boundary_buffer_dist : float, optional
        Distance (m) the watershed polygon is buffered outward to place the no-flow
        boundary. Default 30.0. Must be >= 30 m.
    boundary_spacing : float, optional
        Target spacing (m) between nodes on the buffered boundary ring. Default 50.0.
    stream_point_spacing : float, optional
        Target spacing (m) between stream nodes. Default 25.0.
    stream_clearance_radius : float, optional
        Interior points within this distance (m) of a stream are removed. Default 15.0.
    mesh_quality_opts : str, optional
        Triangle quality switches inserted between the hardcoded ``p`` prefix and
        ``DjYY`` suffix (e.g. ``'q10a15050'``). Do NOT include ``YY`` here — it is
        hardcoded. Default ``'q10'``. See
        https://www.cs.cmu.edu/~quake/triangle.switch.html.
    separate_parallel_streams : bool, optional
        After triangulating, find TIN edges that directly bridge distinct stream
        branches which only join far downstream, insert interior separator nodes to
        break them, and re-triangulate (repeated up to ``MAX_SEPARATOR_PASSES``). This
        stops tRIBS (``WeightedShortestPath``) from merging parallel branches
        prematurely. Default True.

    Attributes
    ----------
    vertices : numpy.ndarray
        (nnodes, 3) array of mesh node x, y, z coordinates, populated by :meth:`generate`.
    triangles : numpy.ndarray
        (ntri, 3) array of triangle node indices, populated by :meth:`generate`.
    node_codes : numpy.ndarray
        Per-node boundary code (0=interior, 1=boundary, 2=outlet, 3=stream),
        populated by :meth:`generate`.

    Examples
    --------
    >>> pslg = MeshFromPSLG(interior_xy, watershed_gdf, stream_gdf, 'dem.tif',
    ...                     outlet=outlet_gdf, boundary_buffer_dist=30.0,
    ...                     boundary_spacing=50.0, stream_point_spacing=40.0,
    ...                     mesh_quality_opts='q10a15050')
    >>> pslg.generate()
    >>> pslg.write('SMF_mesh', 'data/model/mesh/')
    """

    MIN_BUFFER_DIST = 30.0          # m: smallest sensible boundary buffer
    MIN_STREAM_SPACING = 5.0        # m: floor on resampled stream node spacing
    INTERIOR_REMOVE_RADIUS = 5.0    # m: thin interior points closer than this
    MIN_STREAM_GRADIENT = 0.001     # minimum enforced stream slope (1 mm/m)
    STEINER_TOL = 0.1               # m: tolerance for snapping Steiner codes to PSLG
    PARALLEL_GRAPH_FACTOR = 3.0     # a TIN-adjacent stream-node pair whose along-network
                                    #   distance exceeds this many node spacings is a
                                    #   premature merge, not a confluence edge
    SEPARATOR_MAX_EDGE_FACTOR = 2.0 # only separate cross-edges shorter than this many
                                    #   node spacings (a narrow corridor a missing node
                                    #   would fill); longer edges across open ground are
                                    #   an interior-density problem, not a merge to break
    MAX_SEPARATOR_PASSES = 6        # cap on triangulate->separate->re-triangulate passes

    def __init__(self, interior_points, watershed, stream_network, dem_file,
                 outlet=None, boundary_buffer_dist=30.0, boundary_spacing=50.0,
                 stream_point_spacing=25.0, stream_clearance_radius=2.0,
                 mesh_quality_opts='q10',
                 separate_parallel_streams=True):

        self.interior_points = self._coerce_interior_points(interior_points)
        self.watershed_geom = self._coerce_geometry(watershed)
        self.stream_network = stream_network
        self.dem_file = dem_file
        self.outlet_hint_xy = self._coerce_outlet(outlet)

        self.boundary_buffer_dist = boundary_buffer_dist
        self.boundary_spacing = boundary_spacing
        self.stream_point_spacing = stream_point_spacing
        self.stream_clearance_radius = stream_clearance_radius
        self.mesh_quality_opts = mesh_quality_opts
        self.separate_parallel_streams = separate_parallel_streams

        # Outputs populated by generate()
        self.vertices = None
        self.triangles = None
        self.node_codes = None

    # Input coercion helpers
    @staticmethod
    def _coerce_interior_points(interior_points):
        """Return interior points as a DataFrame with x, y, code(=0) columns."""
        if isinstance(interior_points, pd.DataFrame):
            df = interior_points[['x', 'y']].copy()
        else:
            arr = np.asarray(interior_points, float)
            if arr.ndim != 2 or arr.shape[1] < 2:
                raise ValueError("interior_points must be (N, 2+) array-like or a DataFrame with x, y.")
            df = pd.DataFrame(arr[:, :2], columns=['x', 'y'])
        df['code'] = 0
        return df.reset_index(drop=True)

    @staticmethod
    def _coerce_geometry(watershed):
        """Return a single shapely geometry for the watershed."""
        if isinstance(watershed, gpd.GeoDataFrame):
            return watershed.geometry.unary_union
        if isinstance(watershed, gpd.GeoSeries):
            return watershed.unary_union
        return watershed  # assume already a shapely geometry

    @staticmethod
    def _coerce_outlet(outlet):
        """Return the outlet hint as an (x, y) array, or None."""
        if outlet is None:
            return None
        if isinstance(outlet, gpd.GeoDataFrame):
            geom = outlet.geometry.iloc[0]
        elif isinstance(outlet, gpd.GeoSeries):
            geom = outlet.iloc[0]
        else:
            geom = outlet  # assume shapely Point
        return np.array([geom.x, geom.y], dtype=float)

    # Orchestration
    def generate(self):
        """
        Run the full PSLG meshing workflow.

        Builds the boundary and stream PSLG, prepares interior points, triangulates,
        samples and conditions elevations, and assigns node codes. Populates and
        returns the mesh.

        Returns
        -------
        tuple of (numpy.ndarray, numpy.ndarray, numpy.ndarray)
            ``(vertices, triangles, node_codes)`` — also stored on the instance.
        """
        self._validate_parameters()

        with rasterio.open(self.dem_file) as src:
            self._dem_res = max(src.res[0], src.res[1])

        bnd_pts, bnd_codes = self._build_boundary()
        stream_nodes_xy, stream_segments_local, unified_stream_geom, \
            stream_outlet_xy, outlet_bnd_local_idx = self._build_stream_network(bnd_pts, bnd_codes)
        # Clear interior points against the *resampled* stream segments (the straight
        # chords the triangulation actually constrains), unioned with the original
        # curved lines. An interior point in a chord-vs-bend gap is far from the curved
        # line yet can sit on a resampled chord, forming a near-zero-area sliver whose
        # circumcenter -- a Voronoi vertex in tRIBS -- flies far outside the domain and
        # breaks soil/land-use resampling.
        if stream_segments_local:
            resampled_stream_geom = unary_union(
                [LineString([stream_nodes_xy[i], stream_nodes_xy[j]])
                 for i, j in stream_segments_local])
            clearance_geom = unary_union([unified_stream_geom, resampled_stream_geom])
        else:
            clearance_geom = unified_stream_geom
        all_interior = self._prepare_interior_points(clearance_geom)

        # Inner boundary ring: evenly-spaced code-0 nodes along the divide so the
        # outermost simulated Voronoi cells form a smooth band, independent of the
        # wavelet interior-point density. Stream-cleared like other interior points,
        # and thinned where wavelet points already sit nearby.
        inner_ring = self._build_inner_boundary()
        if len(inner_ring) and self.stream_clearance_radius > 0:
            ring_gs = gpd.GeoSeries.from_xy(inner_ring[:, 0], inner_ring[:, 1])
            inner_ring = inner_ring[
                (ring_gs.distance(clearance_geom) >= self.stream_clearance_radius).values]
        if len(inner_ring) and len(all_interior):
            d, _ = cKDTree(all_interior[['x', 'y']].values).query(inner_ring)
            inner_ring = inner_ring[d > self.INTERIOR_REMOVE_RADIUS]
        if len(inner_ring):
            ring_df = pd.DataFrame({'x': inner_ring[:, 0], 'y': inner_ring[:, 1], 'code': 0})
            all_interior = pd.concat([all_interior, ring_df], ignore_index=True)

        # Constant parts of the PSLG (independent of separator insertion)
        n_bnd = len(bnd_pts)
        boundary_segs = [(i, (i + 1) % n_bnd) for i in range(n_bnd)]
        bnd_df = pd.DataFrame({'x': bnd_pts[:, 0], 'y': bnd_pts[:, 1], 'code': bnd_codes})
        # Outlet reach connects this preserved stream endpoint to the boundary outlet node
        _, stream_outlet_local_idx = cKDTree(stream_nodes_xy).query(stream_outlet_xy)
        stream_outlet_local_idx = int(stream_outlet_local_idx)

        # Steps 4-6: assemble points (boundary, interior, then stream), build the PSLG
        # segments, and triangulate. With separate_parallel_streams on, repeat: after
        # each triangulation find any TIN edge that directly bridges two
        # topologically-distant stream nodes (a premature merge), drop an interior
        # separator node on its midpoint, and re-triangulate. The midpoint lies on the
        # offending edge so the constrained Delaunay cannot re-form it.
        print("\nStep 4-6: Combining points, defining constraints, and triangulating...")
        sep_pass = 0
        while True:
            final_non_stream_pts = pd.concat([bnd_df, all_interior], ignore_index=True)
            n_non_stream = len(final_non_stream_pts)  # boundary rows stay at 0..n_bnd-1
            stream_nodes_df = pd.DataFrame(stream_nodes_xy, columns=['x', 'y'])
            stream_nodes_df['code'] = 3
            final_input_df = pd.concat([final_non_stream_pts, stream_nodes_df], ignore_index=True)
            all_xy = final_input_df[['x', 'y']].values

            stream_segs = [(i + n_non_stream, j + n_non_stream) for i, j in stream_segments_local]
            outlet_reach_seg = (outlet_bnd_local_idx, stream_outlet_local_idx + n_non_stream)
            all_segs = set(tuple(sorted(s)) for s in boundary_segs + stream_segs + [outlet_reach_seg])
            segments = [list(s) for s in all_segs]

            mesh_vertices_xy, triangles = self._triangulate(all_xy, segments)

            if not self.separate_parallel_streams:
                break
            new_seps = self._detect_stream_merge_separators(
                mesh_vertices_xy, triangles, stream_nodes_xy, stream_segments_local,
                all_interior[['x', 'y']].values if len(all_interior) else np.empty((0, 2)))
            if not len(new_seps):
                if sep_pass:
                    print(f"  Premature stream merges resolved after {sep_pass} pass(es).")
                break
            sep_pass += 1
            if sep_pass > self.MAX_SEPARATOR_PASSES:
                print(f"  WARNING: {len(new_seps)} premature stream merge(s) remain after "
                      f"{self.MAX_SEPARATOR_PASSES} passes; leaving them. A finer "
                      f"stream_point_spacing may help if they matter.")
                break
            print(f"  Pass {sep_pass}: inserting {len(new_seps)} separator node(s) and "
                  f"re-triangulating.")
            sep_df = pd.DataFrame({'x': new_seps[:, 0], 'y': new_seps[:, 1], 'code': 0})
            all_interior = pd.concat([all_interior, sep_df], ignore_index=True)

        # Step 7: elevations for all (incl. Steiner) vertices
        print("\nStep 7: Assigning elevations to mesh vertices...")
        mesh_vertices_z = self._sample_elevations(mesh_vertices_xy)

        # Step 7b: enforce monotonic descent along the stream network
        mesh_vertices_z = self._enforce_monotonic_descent(
            mesh_vertices_xy, mesh_vertices_z, stream_nodes_xy,
            stream_segments_local, int(stream_outlet_local_idx)
        )

        vertices_3d = np.column_stack([mesh_vertices_xy, mesh_vertices_z])

        # Step 8: node codes
        node_codes = self._assign_node_codes(
            mesh_vertices_xy, all_xy, final_input_df, n_bnd, stream_segs, outlet_reach_seg
        )

        self.vertices = vertices_3d
        self.triangles = np.array(triangles)
        self.node_codes = node_codes
        return self.vertices, self.triangles, self.node_codes

    # Step methods
    def _validate_parameters(self):
        """Validate buffer/spacing parameters and warn on under-sampled boundaries."""
        if self.boundary_buffer_dist < self.MIN_BUFFER_DIST:
            raise ValueError(
                f"boundary_buffer_dist={self.boundary_buffer_dist}m is below the minimum allowed "
                f"value of {self.MIN_BUFFER_DIST}m. A buffer this small places the no-flow boundary "
                f"too close to the watershed edge and is unlikely to produce a valid tRIBS mesh."
            )

        if self.boundary_spacing > 2 * self.boundary_buffer_dist:
            optimal_spacing = 2 * self.boundary_buffer_dist
            optimal_buffer = self.boundary_spacing / 2
            print(
                f"\nWARNING: boundary_spacing ({self.boundary_spacing}m) is more than 2× "
                f"boundary_buffer_dist ({self.boundary_buffer_dist}m). Concave corners of the "
                f"watershed boundary may be under-sampled, causing boundary segments to cut "
                f"back into the watershed.\n"
                f"  To fix, choose ONE of:\n"
                f"    - Reduce boundary_spacing to <= {optimal_spacing:.0f}m "
                f"(optimal for your current buffer)\n"
                f"    - Increase boundary_buffer_dist to >= {optimal_buffer:.0f}m "
                f"(optimal for your current spacing)"
                f"\n  If the recommendations above are not followed then a detailed check of the output shapefiles is required to verify that boundary nodes (Code 1) are surrounding the entire TIN, no interior nodes (Code 0) are on the boundary.\n"
            )

    def _build_boundary(self):
        """Step 1: buffered boundary ring nodes (code 1) from the watershed polygon."""
        print("\nStep 1: Generating buffered boundary from watershed shapefile...")
        buffered_geom = self.watershed_geom.buffer(self.boundary_buffer_dist)
        ext = buffered_geom.exterior  # already-ordered polygon ring

        total_bnd_len = ext.length
        n_bnd = max(3, int(round(total_bnd_len / self.boundary_spacing)))
        bnd_pts = np.array([
            [ext.interpolate(i / n_bnd, normalized=True).x,
             ext.interpolate(i / n_bnd, normalized=True).y]
            for i in range(n_bnd)
        ])
        bnd_codes = np.ones(n_bnd, dtype=int)
        print(f"  {n_bnd} boundary nodes at ~{self.boundary_spacing}m spacing "
              f"(buffer={self.boundary_buffer_dist}m)")
        return bnd_pts, bnd_codes

    def _build_inner_boundary(self):
        """Inner ring of interior (code-0) nodes following the watershed divide at
        boundary_spacing.

        The simulated domain edge is the outer edge of the code-0/3 Voronoi cells, 
        so without evenly-spaced nodes along the divide the outermost cells are large 
        and ragged. This lays down a uniform ring just inside the boundary so those 
        cells form a smooth band. Mirrors the buffered outer ring at the same spacing.
        """
        ext = self.watershed_geom.exterior
        n = max(3, int(round(ext.length / self.boundary_spacing)))
        pts = np.array([
            [ext.interpolate(i / n, normalized=True).x,
             ext.interpolate(i / n, normalized=True).y]
            for i in range(n)
        ])
        print(f"  {n} inner boundary nodes at ~{self.boundary_spacing}m spacing")
        return pts

    def _build_stream_network(self, bnd_pts, bnd_codes):
        """Step 2: node the stream network, locate the outlet, resample, build the PSLG.

        Mutates ``bnd_codes`` in place to mark the boundary outlet node (code 2).

        Returns
        -------
        tuple
            ``(stream_nodes_xy, stream_segments_local, unified_stream_geom,
            stream_outlet_xy, outlet_bnd_local_idx)``.
        """
        print("\nStep 2: Building stream network...")
        raw_lines = []
        for g in self.stream_network.geometry.dropna():
            if isinstance(g, LineString):
                raw_lines.append(g)
            elif g.geom_type == "MultiLineString":
                raw_lines.extend(list(g.geoms))

        lines = _node_streams(raw_lines, tau_snap=1.0)

        # Collect all degree-1 endpoints (potential outlet or tributary inlets)
        endpoint_count = collections.Counter()
        for ln in lines:
            c = list(ln.coords)
            endpoint_count[tuple(c[0][:2])] += 1
            endpoint_count[tuple(c[-1][:2])] += 1
        degree1_pts = [np.array(pt) for pt, cnt in endpoint_count.items() if cnt == 1]
        if not degree1_pts:
            raise RuntimeError("Stream network has no degree-1 endpoints; cannot locate outlet.")

        # Mean nearest-neighbour distance of interior points for spacing guidance
        interior_xy = self.interior_points[['x', 'y']].values
        if len(interior_xy) > 1:
            nn_dists, _ = cKDTree(interior_xy).query(interior_xy, k=2)
            mean_nn = float(nn_dists[:, 1].mean())
        else:
            mean_nn = None

        # Locate the stream outlet: nearest degree-1 endpoint to the outlet hint,
        # falling back to the endpoint nearest the watershed boundary.
        if self.outlet_hint_xy is not None:
            dists = [np.linalg.norm(pt - self.outlet_hint_xy) for pt in degree1_pts]
        else:
            ws_boundary = self.watershed_geom.boundary
            dists = [ws_boundary.distance(Point(pt)) for pt in degree1_pts]

        stream_outlet_xy = degree1_pts[int(np.argmin(dists))]
        print(f"  Stream outlet: ({stream_outlet_xy[0]:.2f}, {stream_outlet_xy[1]:.2f})")

        # Nearest buffered boundary node becomes the outlet (code 2)
        bnd_kdt = cKDTree(bnd_pts)
        _, outlet_bnd_local_idx = bnd_kdt.query(stream_outlet_xy)
        outlet_bnd_local_idx = int(outlet_bnd_local_idx)
        bnd_codes[outlet_bnd_local_idx] = 2
        outlet_bnd_xy = bnd_pts[outlet_bnd_local_idx]
        print(f"  Outlet boundary node: idx={outlet_bnd_local_idx} "
              f"({outlet_bnd_xy[0]:.2f}, {outlet_bnd_xy[1]:.2f})")

        # Resample stream edges
        dedupe_radius = max(1.5, 0.05 * float(self.stream_point_spacing))
        resampled_edges = []
        for ln in lines:
            pts = _resample_with_min_spacing(
                ln, spacing=self.stream_point_spacing, m_min=self.MIN_STREAM_SPACING,
                angle_tol_deg=2.0, keep_original_vertices=False
            )
            if len(pts) >= 2:
                resampled_edges.append(pts)

        unified_stream_geom = linemerge(unary_union(lines))
        total_len_km = sum(ln.length for ln in lines) / 1000.0

        stream_nodes_xy, stream_segments_local = _build_stream_pslg(
            resampled_edges, m_min=self.MIN_STREAM_SPACING, dedupe_radius=dedupe_radius
        )
        print(f"  {len(stream_nodes_xy)} stream nodes from {total_len_km:.2f} km "
              f"at ~{self.stream_point_spacing}m spacing")

        # Longest stream flow path via two-sweep BFS on the assembled PSLG graph
        if stream_segments_local:
            sadj = collections.defaultdict(list)
            for i, j in stream_segments_local:
                d = float(np.linalg.norm(stream_nodes_xy[i] - stream_nodes_xy[j]))
                sadj[i].append((j, d))
                sadj[j].append((i, d))

            def bfs_far(adj, src):
                dist = {src: 0.0}
                q = collections.deque([src])
                far, dmax = src, 0.0
                while q:
                    v = q.popleft()
                    for u, d in adj[v]:
                        if u not in dist:
                            dist[u] = dist[v] + d
                            if dist[u] > dmax:
                                dmax, far = dist[u], u
                            q.append(u)
                return far, dmax

            u, _ = bfs_far(sadj, next(iter(sadj)))
            _, longest_m = bfs_far(sadj, u)
            longest_km = longest_m / 1000.0
        else:
            longest_km = 0.0

        print(f"\n  Stream spacing guidance (your setting: {self.stream_point_spacing}m):")
        print(f"    DEM resolution       : {self._dem_res:.1f}m  ← minimum meaningful spacing")
        if mean_nn is not None:
            print(f"    Mean interior spacing: {mean_nn:.1f}m  ← suggested maximum")
            print(f"    Suggested range      : {self._dem_res:.1f}m – {mean_nn:.1f}m")
        print(f"    Longest flow path    : {longest_km:.2f}km (from assembled breakline)")
        if self.stream_point_spacing < self._dem_res:
            print(
                f"  WARNING: stream_point_spacing ({self.stream_point_spacing}m) is finer than the "
                f"DEM resolution ({self._dem_res:.1f}m). This adds detail that does not exist in the "
                f"source data. Consider increasing to at least {self._dem_res:.1f}m."
            )
        if mean_nn is not None and self.stream_point_spacing > mean_nn:
            print(
                f"  WARNING: stream_point_spacing ({self.stream_point_spacing}m) is coarser than "
                f"the mean interior point spacing ({mean_nn:.1f}m). The stream network will "
                f"be represented at a lower resolution than the surrounding terrain. "
                f"Consider decreasing to at most {mean_nn:.1f}m."
            )

        return (stream_nodes_xy, stream_segments_local, unified_stream_geom,
                stream_outlet_xy, outlet_bnd_local_idx)

    def _prepare_interior_points(self, unified_stream_geom):
        """Step 3: thin interior points, optionally cull near trees, apply stream clearance."""
        print("\nStep 3: Preparing interior points...")
        interior_pts = self.interior_points[['x', 'y', 'code']].copy().reset_index(drop=True)

        # Remove interior points too close to each other
        if len(interior_pts) > 1:
            int_kdt = cKDTree(interior_pts[['x', 'y']].values)
            close_pairs = int_kdt.query_pairs(r=self.INTERIOR_REMOVE_RADIUS)
            drop_set = {j for _, j in close_pairs}
            if drop_set:
                print(f"  Removing {len(drop_set)} interior points within "
                      f"{self.INTERIOR_REMOVE_RADIUS}m of another.")
                interior_pts = interior_pts.drop(index=list(drop_set)).reset_index(drop=True)

        # Extension hook: subclasses may add or remove interior nodes here before the
        # stream clearance step (see _augment_interior_points).
        all_interior = self._augment_interior_points(interior_pts)
        if all_interior is None or all_interior.empty:
            all_interior = pd.DataFrame(columns=['x', 'y', 'code'])

        # Stream clearance zone on interior points only
        if self.stream_clearance_radius > 0 and not all_interior.empty:
            print(f"  Applying {self.stream_clearance_radius}m stream clearance zone...")
            interior_gs = gpd.GeoSeries.from_xy(all_interior['x'], all_interior['y'])
            dist_to_stream = interior_gs.distance(unified_stream_geom)
            too_close = dist_to_stream < self.stream_clearance_radius
            n_removed = int(too_close.sum())
            if n_removed:
                print(f"  Removing {n_removed} interior points inside clearance zone.")
                all_interior = all_interior[~too_close.values].reset_index(drop=True)

        return all_interior

    def _augment_interior_points(self, interior_pts):
        """Hook for subclasses to add or remove interior nodes.

        Called by :meth:`_prepare_interior_points` after de-duplication and before
        the stream-clearance step. Receives a DataFrame of interior points with
        ``x``, ``y`` and ``code`` columns and must return a DataFrame with the same
        columns. The base implementation is a no-op, returning the points unchanged.
        Override to inject additional interior nodes from another data source.
        """
        return interior_pts

    def _detect_stream_merge_separators(self, mesh_vertices_xy, triangles,
                                        stream_nodes_xy, stream_segments_local,
                                        existing_interior_xy):
        """Find TIN edges that directly bridge topologically-distant stream nodes and
        return midpoint separator coordinates to break them on re-triangulation.

        tRIBS rebuilds the channel network in ``tFlowNet::WeightedShortestPath`` by
        connecting stream nodes through *direct* stream-to-stream TIN edges, so two
        separate branches merge prematurely exactly when such an edge spans them. This
        inspects the actual triangulation (not a distance proxy): for every TIN edge
        whose endpoints are both stream nodes, if the two are far apart along the
        stream network, more than ``PARALLEL_GRAPH_FACTOR`` node spacings, or
        unreachable (different components), then the edge is a premature merge. A
        separator placed on its midpoint lies on the edge, so the constrained Delaunay
        cannot re-form it; being code-0 it is invisible to ``WeightedShortestPath``,
        forcing the branches to join only at their real confluence.

        Returns an (M, 2) array of separator x, y coordinates (possibly empty).
        """
        S = np.asarray(stream_nodes_xy, float)
        n_s = len(S)
        if n_s < 2 or not stream_segments_local:
            return np.empty((0, 2), float)

        # Map each stream node to its mesh vertex (positions preserved by the YY switch)
        _, stream_mesh_idx = cKDTree(mesh_vertices_xy).query(S)
        mesh_to_stream = {int(m): i for i, m in enumerate(stream_mesh_idx)}

        # Unique TIN edges whose endpoints are both stream nodes. Only short edges are
        # considered: a separator fixes a narrow corridor that a missing interior node
        # would otherwise occupy. Long stream-to-stream edges span open ground (too few
        # interior nodes).
        max_edge = self.SEPARATOR_MAX_EDGE_FACTOR * self.stream_point_spacing
        seen, cand = set(), []
        for t in triangles:
            for k in range(3):
                a = mesh_to_stream.get(int(t[k]))
                b = mesh_to_stream.get(int(t[(k + 1) % 3]))
                if a is None or b is None or a == b:
                    continue
                if np.linalg.norm(S[a] - S[b]) > max_edge:
                    continue
                key = (a, b) if a < b else (b, a)
                if key not in seen:
                    seen.add(key); cand.append(key)
        if not cand:
            return np.empty((0, 2), float)

        # Along-network distances for the stream nodes involved
        rows, cols, wts = [], [], []
        for i, j in stream_segments_local:
            d = float(np.linalg.norm(S[i] - S[j]))
            rows += [i, j]; cols += [j, i]; wts += [d, d]
        graph = csr_matrix((wts, (rows, cols)), shape=(n_s, n_s))
        srcs = sorted({a for a, _ in cand})
        src_row = {s: r for r, s in enumerate(srcs)}
        gdist = dijkstra(graph, directed=False, indices=srcs)

        graph_sep = self.PARALLEL_GRAPH_FACTOR * self.stream_point_spacing
        seps = [0.5 * (S[i] + S[j]) for i, j in cand
                if not np.isfinite(gdist[src_row[i], j]) or gdist[src_row[i], j] > graph_sep]
        if not seps:
            return np.empty((0, 2), float)

        seps = np.asarray(seps, float)
        tol = max(1.0, 0.25 * self.stream_point_spacing)
        # De-duplicate near-coincident separators, and drop any already present as nodes
        drop = {b for _, b in cKDTree(seps).query_pairs(r=tol)}
        seps = seps[[k for k in range(len(seps)) if k not in drop]]
        if len(seps) and len(existing_interior_xy):
            d, _ = cKDTree(np.asarray(existing_interior_xy, float)).query(seps)
            seps = seps[d > tol]
        return seps

    def _sample_elevations(self, points_xy):
        """Sample DEM elevations for a list of (x, y) points, handling nodata.

        The buffered no-flow boundary and the outlet sit outside the watershed, and
        clipped DEMs commonly carry nodata there, so some samples come back as the
        nodata value (e.g. -999999), NaN, or out-of-bounds. Each such sample is
        filled with the elevation of the nearest point that sampled valid terrain, so
        no node carries the raw nodata into the mesh.
        """
        print(f"  - Reading DEM: {self.dem_file}")
        pts = np.asarray(points_xy, float)
        with rasterio.open(self.dem_file) as src:
            nodata = src.nodata
            elev = np.array([val[0] for val in src.sample([tuple(p) for p in pts])], dtype=float)

        bad = ~np.isfinite(elev)
        if nodata is not None:
            bad |= np.isclose(elev, nodata)
        n_bad = int(bad.sum())
        if n_bad:
            if bad.all():
                raise ValueError(
                    f"All {len(elev)} sampled elevations are nodata; does '{self.dem_file}' "
                    f"cover the mesh extent including the buffered boundary?")
            good = ~bad
            _, nn = cKDTree(pts[good]).query(pts[bad])
            elev[bad] = elev[good][nn]
            print(f"  - Filled {n_bad} nodata elevation(s) from the nearest valid sample.")
        print(f"  - Sampled {len(elev)} points.")
        return elev

    def _triangulate(self, all_xy, segments):
        """Step 6: triangulate the PSLG with Triangle (no Steiner points on segments)."""
        print("\nStep 6: Triangulating...")
        Vn, origin_xy, scale_xy = _normalize_coords(all_xy)
        # YY: no Steiner points on any PSLG segment (boundary ring or stream segments).
        # Interior Steiner points are still added as needed by the quality options.
        # This ensures every stream-coded node is an original PSLG node with a
        # monotonically-enforced elevation, not a DEM-raw Steiner interpolation.
        mesh = tr.triangulate({'vertices': Vn, 'segments': segments},
                              f"p{self.mesh_quality_opts}DjYY")
        mesh_vertices_xy = _denormalize_coords(np.asarray(mesh['vertices']), origin_xy, scale_xy)
        return mesh_vertices_xy, mesh['triangles']

    def _enforce_monotonic_descent(self, mesh_vertices_xy, mesh_vertices_z, stream_nodes_xy,
                                   stream_segments_local, stream_outlet_local_idx):
        """Step 7b: nudge stream-node elevations so they strictly descend to the outlet.

        FlowDirs() in tRIBS marks any node with steepest-slope <= 0 as a sink. At
        coarse DEM resolution stream nodes can share a raster cell, producing flat
        or inverted segments that trigger lakelist errors. Walk the assembled PSLG
        upstream from the outlet and raise any node that is not strictly higher than
        its downstream neighbour.
        """
        print("\nStep 7b: Enforcing monotonic descent along stream network...")
        if not stream_segments_local:
            print("  No stream segments — skipping.")
            return mesh_vertices_z

        tmp_kdt = cKDTree(mesh_vertices_xy)
        _, stream_mesh_idx = tmp_kdt.query(stream_nodes_xy)
        mono_z = mesh_vertices_z[stream_mesh_idx].copy()

        mono_adj = collections.defaultdict(list)
        for mi, mj in stream_segments_local:
            mono_adj[mi].append(mj)
            mono_adj[mj].append(mi)

        # The outlet terminal often samples a bank elevation rather than the
        # thalweg (especially at coarse DEM resolution). Starting BFS from it would
        # incorrectly raise all upstream nodes. Since FillLakes always drains to
        # kOpenBoundary regardless of elevation, the terminal's own elevation is
        # irrelevant, skip it and root the BFS at its upstream neighbour instead.
        outlet_upstream = mono_adj[stream_outlet_local_idx]
        if outlet_upstream:
            bfs_root = outlet_upstream[0]
            mono_visited = {stream_outlet_local_idx, bfs_root}
            mono_queue = collections.deque([bfs_root])
        else:
            mono_visited = {stream_outlet_local_idx}
            mono_queue = collections.deque([stream_outlet_local_idx])
        n_adjusted = 0

        while mono_queue:
            dn = mono_queue.popleft()          # downstream node
            for up in mono_adj[dn]:
                if up not in mono_visited:
                    mono_visited.add(up)
                    seg_len = float(np.linalg.norm(stream_nodes_xy[up] - stream_nodes_xy[dn]))
                    min_z = mono_z[dn] + max(seg_len * self.MIN_STREAM_GRADIENT, 0.01)
                    if mono_z[up] <= mono_z[dn]:
                        mono_z[up] = min_z
                        n_adjusted += 1
                    mono_queue.append(up)

        mesh_vertices_z[stream_mesh_idx] = mono_z
        if n_adjusted:
            print(f"  Adjusted {n_adjusted} of {len(stream_nodes_xy)} stream nodes "
                  f"(min gradient enforced: {self.MIN_STREAM_GRADIENT * 1000:.1f} mm/m).")
        else:
            print("  Stream elevations already monotonically decreasing — no adjustments needed.")
        return mesh_vertices_z

    def _assign_node_codes(self, mesh_vertices_xy, all_xy, final_input_df,
                           n_bnd, stream_segs, outlet_reach_seg):
        """Step 8: assign tRIBS boundary codes to every mesh node (incl. Steiner)."""
        print("\nStep 8: Assigning node codes...")
        n_mesh_verts = len(mesh_vertices_xy)
        final_node_codes = np.zeros(n_mesh_verts, dtype=int)

        mesh_kdt = cKDTree(mesh_vertices_xy)
        _, mesh_idx_for_orig = mesh_kdt.query(all_xy)

        for orig_idx, mesh_idx in enumerate(mesh_idx_for_orig):
            final_node_codes[mesh_idx] = final_input_df['code'].iloc[orig_idx]

        is_original = np.zeros(n_mesh_verts, dtype=bool)
        is_original[mesh_idx_for_orig] = True

        # Vectorized Steiner vertex code assignment
        new_vert_indices = np.where(~is_original)[0]
        if new_vert_indices.size > 0:
            new_vert_gs = gpd.GeoSeries.from_xy(
                mesh_vertices_xy[new_vert_indices, 0],
                mesh_vertices_xy[new_vert_indices, 1]
            )
            # Unified geometry of every stream PSLG segment + outlet reach
            stream_pslg_geom = unary_union([
                LineString([all_xy[i], all_xy[j]])
                for i, j in stream_segs + [outlet_reach_seg]
            ])
            dist_to_stream = new_vert_gs.distance(stream_pslg_geom)
            # Use the actual PSLG chord segments (not the smooth exterior ring) so
            # that Steiner points placed on the straight chords are within tolerance.
            bnd_pslg_geom = unary_union([
                LineString([all_xy[i], all_xy[(i + 1) % n_bnd]])
                for i in range(n_bnd)
            ])
            dist_to_bnd = new_vert_gs.distance(bnd_pslg_geom)

            on_stream = dist_to_stream.values < self.STEINER_TOL
            on_bnd = (~on_stream) & (dist_to_bnd.values < self.STEINER_TOL)
            final_node_codes[new_vert_indices[on_stream]] = 3
            final_node_codes[new_vert_indices[on_bnd]] = 1

        print(f"  Final codes — 0(Interior):{(final_node_codes==0).sum()}  "
              f"1(Boundary):{(final_node_codes==1).sum()}  "
              f"2(Outlet):{(final_node_codes==2).sum()}  "
              f"3(Stream):{(final_node_codes==3).sum()}")
        return final_node_codes

    # Output
    def write(self, output_prefix, output_path):
        """Write the four tRIBS mesh files. Call :meth:`generate` first."""
        if self.vertices is None:
            raise RuntimeError("Call generate() before write().")
        InOut.write_mesh_files(output_prefix, output_path,
                               self.vertices, self.triangles, self.node_codes)

    def write_diagnostics(self, output_base, crs=None):
        """Write diagnostic shapefiles for the generated mesh. Call :meth:`generate` first."""
        if self.vertices is None:
            raise RuntimeError("Call generate() before write_diagnostics().")
        InOut.write_mesh_diagnostics(output_base, self.vertices, self.triangles,
                                     self.node_codes, crs=crs)


class MeshProcessor:
    """
    Mixin providing high-level mesh-generation workflows for the :class:`Mesh` class.

    These methods orchestrate the lower-level :class:`GenerateMesh` and
    :class:`MeshFromPSLG` classes and operate on a :class:`Mesh` instance — they
    expect ``self.mesh_generator`` (a :class:`GenerateMesh`), ``self.meta``, and the
    ``self.optmeshinput`` / ``self.inputdatafile`` option dictionaries to be set by
    ``Mesh.__init__``. The class mirrors the :class:`~pytRIBS.model.model.ModelProcessor`
    pattern, keeping orchestration logic in this module rather than in ``classes.py``.
    """

    def build_mesh(self, method='pslg', **kwargs):
        """
        Generate a tRIBS mesh using one of two strategies.

        Parameters
        ----------
        method : {'pslg', 'points', 'station'}, optional
            Mesh-generation strategy. Default ``'pslg'``.

            - ``'pslg'`` (primary): build a planar straight-line graph (buffered
              boundary + stream breaklines + outlet reach) and triangulate it
              in-process with Triangle, writing the four tRIBS mesh files
              (``.nodes``/``.edges``/``.tri``/``.z``) that tRIBS reads directly.
              See :meth:`generate_pslg_mesh`.
            - ``'points'`` (legacy): select significant terrain points and write a
              ``.points`` file for tRIBS/meshbuilder to triangulate. See
              :meth:`generate_points_mesh`.
            - ``'station'``: build a minimal single-polygon mesh for a point/station
              model — one central stream node ringed by closed boundary nodes with a
              single downslope outlet. See :meth:`generate_station_mesh`.
        **kwargs
            Forwarded to the selected method.

        Returns
        -------
        object
            The :class:`MeshFromPSLG` instance (``'pslg'``) or the points
            GeoDataFrame (``'points'`` / ``'station'``).
        """
        if method == 'pslg':
            return self.generate_pslg_mesh(**kwargs)
        if method == 'points':
            return self.generate_points_mesh(**kwargs)
        if method == 'station':
            return self.generate_station_mesh(**kwargs)
        raise ValueError(f"Unknown mesh method '{method}'. Use 'pslg', 'points', or 'station'.")

    def generate_pslg_mesh(self, threshold=None, interior_points=None,
                           output_prefix=None, output_path=None, write_files=True,
                           diagnostics=False, crs=None, pslg_class=MeshFromPSLG,
                           **pslg_kwargs):
        """
        Generate a complete TIN via PSLG triangulation and write the tRIBS mesh files.

        Reuses the watershed, stream network, outlet and DEM already loaded by the
        :class:`GenerateMesh` instance (``self.mesh_generator``), so no shapefiles
        are re-read. Interior terrain points are either supplied directly or selected
        from the DEM via the wavelet/curvature routine. On a successful write the
        model is pointed at the new mesh (``OPTMESHINPUT`` = 1, ``INPUTDATAFILE`` =
        base name), which propagates to :class:`~pytRIBS.classes.Model` via shared
        options.

        Parameters
        ----------
        threshold : float, optional
            Significance threshold passed to
            :meth:`GenerateMesh.extract_points_from_significant_details` to select
            interior (code-0) points. Required unless ``interior_points`` is given.
        interior_points : numpy.ndarray or pandas.DataFrame, optional
            Pre-computed interior points (x, y). If given, ``threshold`` is ignored.
        output_prefix : str, optional
            Base name for the mesh files. Defaults to ``'{Name}_mesh'``.
        output_path : str, optional
            Output directory for the mesh files. Defaults to the Mesh instance's
            ``mesh_dir`` (e.g. ``proj.directories['mesh']``), or the current working
            directory if neither is set. Created if it does not exist.
        write_files : bool, optional
            Write the four tRIBS mesh files. Default True.
        diagnostics : bool, optional
            Also write diagnostic shapefiles. Default False.
        crs : optional
            CRS for diagnostic shapefiles. Defaults to the project EPSG from metadata.
        pslg_class : type, optional
            The PSLG mesh class to instantiate. Defaults to :class:`MeshFromPSLG`.
            Extension packages may pass a subclass to customise mesh generation
            (any extra keyword arguments are forwarded to it via ``pslg_kwargs``).
        **pslg_kwargs
            Additional parameters forwarded to ``pslg_class`` (e.g.
            ``boundary_buffer_dist``, ``boundary_spacing``, ``stream_point_spacing``,
            ``mesh_quality_opts``).

        Returns
        -------
        :class:`MeshFromPSLG`
            The generated mesh object (also stored as ``self.pslg_mesh``).
        """
        if getattr(self, 'mesh_generator', None) is None:
            raise RuntimeError(
                "generate_pslg_mesh requires a GenerateMesh instance; initialize Mesh "
                "with generate_mesh_args so watershed/stream/outlet/DEM are loaded."
            )
        gm = self.mesh_generator

        if interior_points is None:
            if threshold is None:
                raise ValueError("Provide either 'threshold' (to select interior points) "
                                 "or an explicit 'interior_points' array.")
            points, _ = gm.extract_points_from_significant_details(threshold)
            interior_points = points[points[:, 3] == 0][:, :2]  # code-0 interior only

        pslg = pslg_class(
            interior_points=interior_points,
            watershed=gm.watershed,
            stream_network=gm.stream_network,
            dem_file=gm.raster,
            outlet=gm.outlet,
            **pslg_kwargs,
        )
        pslg.generate()

        if output_prefix is None:
            output_prefix = f"{self.meta.get('Name') or 'mesh'}_mesh"

        # Resolve the output directory: explicit arg > Mesh.mesh_dir > cwd.
        out_dir = output_path if output_path is not None else (getattr(self, 'mesh_dir', None) or '')
        out_dir = os.path.join(out_dir, '')  # ensure a trailing separator
        base = f"{out_dir}{output_prefix}"

        if write_files:
            if out_dir.strip(os.sep):
                os.makedirs(out_dir, exist_ok=True)
            pslg.write(output_prefix, out_dir)
            # Point the model at the complete 4-file mesh: OPTMESHINPUT 1 (tMesh data)
            # read from the INPUTDATAFILE base name. Picked up by Model via shared options.
            self.optmeshinput['value'] = 1
            self.inputdatafile['value'] = base

        if diagnostics:
            if crs is None and self.meta.get('EPSG') is not None:
                crs = f"EPSG:{self.meta['EPSG']}"
            pslg.write_diagnostics(base, crs=crs)

        self.pslg_mesh = pslg
        return pslg

    def generate_points_mesh(self, threshold, output_file=None):
        """
        Points path: select significant terrain points and write a ``.points`` file.

        tRIBS triangulates the points itself (``OPTMESHINPUT`` = 2, Point
        Triangulator). This is a convenience wrapper around the historical workflow
        (:meth:`GenerateMesh.extract_points_from_significant_details` →
        :meth:`GenerateMesh.convert_points_to_gdf` →
        :meth:`GenerateMesh.write_point_file`); those methods remain available for
        direct use. On success it points the model at the file
        (``POINTFILENAME`` + ``OPTMESHINPUT`` = 2, propagated to
        :class:`~pytRIBS.classes.Model` via shared options) and stores the buffered
        watershed as ``self.buffered_watershed`` for downstream workflows.

        Parameters
        ----------
        threshold : float
            Significance threshold passed to
            :meth:`GenerateMesh.extract_points_from_significant_details`. Lower values
            keep more points (denser mesh); general range 0–1.
        output_file : str, optional
            Path for the ``.points`` file. Defaults to ``'{Name}.points'`` inside the
            Mesh instance's ``mesh_dir`` (e.g. ``proj.directories['mesh']``), or the
            current working directory if ``mesh_dir`` is not set.

        Returns
        -------
        geopandas.GeoDataFrame
            The selected nodes written to the points file. The buffered watershed is
            also available as ``self.buffered_watershed``.
        """
        if getattr(self, 'mesh_generator', None) is None:
            raise RuntimeError(
                "generate_points_mesh requires a GenerateMesh instance; initialize Mesh "
                "with generate_mesh_args."
            )
        gm = self.mesh_generator
        points, buffered_watershed = gm.extract_points_from_significant_details(threshold)
        gdf = gm.convert_points_to_gdf(points)

        if output_file is None:
            out_dir = getattr(self, 'mesh_dir', None) or ''
            output_file = os.path.join(out_dir, f"{self.meta.get('Name') or 'mesh'}.points")
        parent = os.path.dirname(output_file)
        if parent:
            os.makedirs(parent, exist_ok=True)
        gm.write_point_file(gdf, output_file)

        # Point the model at the points file: OPTMESHINPUT 2 (Point Triangulator).
        # Picked up by Model via shared options.
        self.pointfilename['value'] = output_file
        self.optmeshinput['value'] = 2
        # Expose the buffered watershed (used by downstream met/forcing workflows).
        self.buffered_watershed = buffered_watershed
        return gdf

    # Minimum drainage slope for a station mesh; a flat (zero-gradient) flow edge
    # stalls kinematic routing, so the realised slope is floored at this value.
    MIN_STATION_SLOPE_DEG = 0.5

    # Default station Voronoi cell area (m^2) when the user specifies neither radius
    # nor area: ~100 m^2, i.e. a standard 10 m grid cell
    DEFAULT_STATION_AREA = 100.0

    # Default footprint (m) over which slope/aspect are derived from the DEM. Decoupled
    # from the (cosmetic) cell size so a small cell does not yield a noisy, too-local
    # gradient. ~50 m is good enough for a representative hillslope scale.
    DEFAULT_TERRAIN_RADIUS = 50.0

    def generate_station_mesh(self, station_xy, station_elev=None, slope=None, aspect=None,
                              radius=None, area=None, terrain_radius=None, dem_file=None,
                              n_boundary=8, min_slope=MIN_STATION_SLOPE_DEG, output_file=None):
        """
        Build a minimal single-polygon (point/station) mesh and write a ``.points`` file.

        Produces one central stream node (``bc`` = 3) ringed by ``n_boundary`` evenly
        spaced closed boundary nodes (``bc`` = 1), with a single open outlet
        (``bc`` = 2) placed in the downslope (aspect) direction. tRIBS triangulates
        the points itself (``OPTMESHINPUT`` = 2).

        For a single interior node only two elevations affect the physics: the centre
        Z (the station elevation, which also drives the met lapse-rate correction) and
        the outlet Z, whose difference over the centre→outlet distance sets the one
        flow-edge slope feeding the radiation/ET surface slope and the kinematic
        hydraulic gradient. Voronoi cell area comes from the ring geometry and aspect
        from the outlet azimuth — so no terrain surface is needed. The closed ring
        nodes are inert and are placed at the centre elevation.

        Slope and aspect are taken from ``dem_file`` when provided (the typical case —
        derived from a least-squares plane fit over ``terrain_radius``, a hillslope-scale
        footprint decoupled from the cell size so the gradient reflects the hillslope
        rather than the DEM pixel); otherwise pass them directly.

        Parameters
        ----------
        station_xy : tuple of float
            (x, y) coordinate of the station / central node.
        station_elev : float, optional
            Centre-node elevation. Sampled from ``dem_file`` if not given; required if
            no DEM is provided.
        slope : float, optional
            Drainage slope in **degrees**. Derived from ``dem_file`` if not given;
            otherwise defaults to ``min_slope`` (with a warning). Floored at
            ``min_slope`` so routing does not stall.
        aspect : float, optional
            Outlet azimuth in **degrees** clockwise from north. Derived from
            ``dem_file`` if not given; otherwise defaults to 180° (with a warning).
        radius : float, optional
            Ring radius (m), i.e. the centre→outlet distance. If neither ``radius`` nor
            ``area`` is given, defaults to a ~100 m² cell (see ``area``).
        area : float, optional
            Target Voronoi cell area (m²) for the centre node, converted to the ring
            radius that yields it. Takes precedence over ``radius``. When neither is
            specified, defaults to 100 m² (≈ a 10 m grid cell, matching the standard
            single-station template).
        terrain_radius : float, optional
            Footprint (m) over which slope/aspect are derived from ``dem_file``,
            decoupled from the cell size. Defaults to ``max(50 m, ring radius)`` so the
            gradient is read at a hillslope scale even for a small cell. Has no effect
            when slope/aspect are supplied directly.
        dem_file : str, optional
            DEM used to derive any unspecified ``station_elev`` / ``slope`` / ``aspect``.
        n_boundary : int, optional
            Number of ring nodes. Default 8. (Cosmetic for a single-cell model; only
            even angular spacing matters.)
        min_slope : float, optional
            Minimum drainage slope in degrees. Default 0.5.
        output_file : str, optional
            Path for the ``.points`` file. Defaults to ``'{Name}.points'`` inside the
            Mesh instance's ``mesh_dir``, or the current working directory.

        Returns
        -------
        geopandas.GeoDataFrame
            The station nodes written to the points file.
        """
        cx, cy = float(station_xy[0]), float(station_xy[1])

        # Cell size: the centre node's Voronoi cell is a regular n-gon with apothem
        # R/2, so area = n*(R/2)^2*tan(pi/n). Default to a representative ~100 m^2 cell
        # so users need not pick a size; an explicit ``area`` (preferred) or ``radius``
        # overrides it. Resolve to a radius first so DEM slope/aspect can be derived at
        # the cell scale.
        if radius is None and area is None:
            area = self.DEFAULT_STATION_AREA
        if area is not None:
            radius = 2.0 * np.sqrt(area / (n_boundary * np.tan(np.pi / n_boundary)))

        # Slope/aspect footprint is decoupled from the (cosmetic) cell size: default to
        # a hillslope scale, but never smaller than the cell itself.
        if terrain_radius is None:
            terrain_radius = max(self.DEFAULT_TERRAIN_RADIUS, radius)

        # DEM is the primary source for the physical inputs: derive any unspecified
        # elevation / slope / aspect from the DEM over the terrain footprint, so
        # fine-resolution micro-topography doesn't dominate the gradient.
        if dem_file is not None:
            dem_elev, dem_slope, dem_aspect = self._dem_terrain(dem_file, cx, cy, terrain_radius)
            if station_elev is None:
                station_elev = dem_elev
            if slope is None:
                slope = dem_slope
            if aspect is None:
                aspect = dem_aspect

        if station_elev is None:
            raise ValueError("station_elev is required when no dem_file is provided.")
        if aspect is None:
            aspect = 180.0
            print("  WARNING: no aspect provided or derived; defaulting outlet to due south (180°).")
        if slope is None:
            slope = min_slope
            print(f"  WARNING: no slope provided or derived; defaulting to {min_slope}° "
                  f"so kinematic routing does not stall.")

        # Guardrail: a flat flow edge stalls routing — floor the slope.
        if slope < min_slope:
            print(f"  WARNING: slope {slope:.3f}° below minimum {min_slope}°; raising to {min_slope}°.")
            slope = min_slope

        # Ring: outlet placed exactly at the aspect azimuth, the rest evenly spaced.
        # Azimuth is clockwise from north: 0° -> +y (north), 90° -> +x (east).
        az = aspect + np.arange(n_boundary) * (360.0 / n_boundary)
        az_rad = np.radians(az)
        ring_x = cx + radius * np.sin(az_rad)
        ring_y = cy + radius * np.cos(az_rad)

        # Realised slope uses the actual placed radius (centre->outlet distance).
        outlet_z = station_elev - np.tan(np.radians(slope)) * radius

        # Centre stream node, then ring nodes (index 1 = the aspect-aligned outlet).
        xs = [cx] + list(ring_x)
        ys = [cy] + list(ring_y)
        zs = [station_elev] + [station_elev] * n_boundary
        bcs = [3] + [1] * n_boundary
        zs[1] = outlet_z
        bcs[1] = 2

        crs = f"EPSG:{self.meta['EPSG']}" if self.meta.get('EPSG') else None
        gdf = gpd.GeoDataFrame(
            {'elevation': zs, 'bc': bcs},
            geometry=[Point(x, y) for x, y in zip(xs, ys)],
            crs=crs,
        )

        if output_file is None:
            out_dir = getattr(self, 'mesh_dir', None) or ''
            output_file = os.path.join(out_dir, f"{self.meta.get('Name') or 'station'}.points")
        parent = os.path.dirname(output_file)
        if parent:
            os.makedirs(parent, exist_ok=True)
        InOut.write_point_file(gdf, output_file)

        # Point the model at the points file: OPTMESHINPUT 2 (Point Triangulator).
        self.pointfilename['value'] = output_file
        self.optmeshinput['value'] = 2

        print(f"  Station mesh: centre stream node + {n_boundary} ring nodes; outlet at "
              f"{aspect:.0f}° azimuth, slope {slope:.2f}°, radius {radius:.1f} m "
              f"(Δz {station_elev - outlet_z:.2f} m) -> {output_file}")
        return gdf

    @staticmethod
    def _dem_terrain(dem_file, x, y, radius):
        """Sample station elevation and derive slope/aspect over the cell footprint.

        The station elevation is the DEM value at ``(x, y)``. Slope (deg) and downslope
        aspect (deg azimuth clockwise from north) come from a least-squares plane fit
        to every DEM cell within ``radius`` of ``(x, y)`` — i.e. at the scale of the
        station's representative cell, not the DEM pixel. This prevents fine-resolution
        micro-topography from dominating the gradient. Falls back to a flat result
        (slope 0°, aspect 0°) if fewer than three valid cells fall within the radius
        (e.g. a tiny radius relative to the DEM, or the station on the raster edge).
        """
        with rasterio.open(dem_file) as src:
            elev = float(next(src.sample([(x, y)]))[0])
            row, col = src.index(x, y)
            t = src.transform
            res_x = abs(t.a)
            res_y = abs(t.e)
            npix_x = max(1, int(np.ceil(radius / res_x)))
            npix_y = max(1, int(np.ceil(radius / res_y)))
            r0, r1 = max(row - npix_y, 0), min(row + npix_y + 1, src.height)
            c0, c1 = max(col - npix_x, 0), min(col + npix_x + 1, src.width)
            win = src.read(1, window=((r0, r1), (c0, c1))).astype(float)
            nodata = src.nodata
            # Cell-centre real-world coordinates for the window (north-up assumed).
            xs = t.c + (np.arange(c0, c1) + 0.5) * t.a
            ys = t.f + (np.arange(r0, r1) + 0.5) * t.e
            xg, yg = np.meshgrid(xs, ys)

        mask = (xg - x) ** 2 + (yg - y) ** 2 <= radius ** 2
        mask &= np.isfinite(win)
        if nodata is not None:
            mask &= win != nodata

        n_cells = int(mask.sum())
        if n_cells < 3:
            print(f"  WARNING: terrain footprint (radius {radius:.0f} m) covers only "
                  f"{n_cells} DEM cell(s); cannot fit a slope/aspect plane — treating as "
                  f"flat. Increase terrain_radius.")
            return elev, 0.0, 0.0
        if n_cells < 9:
            print(f"  WARNING: terrain footprint (radius {radius:.0f} m) spans only "
                  f"{n_cells} DEM cells; derived slope/aspect may be noisy. Consider a "
                  f"larger terrain_radius.")

        # Fit z = a*(X-x) + b*(Y-y) + c; a = dz/d_east, b = dz/d_north (per-metre).
        # Coordinates are centred on the station for numerical conditioning.
        A = np.column_stack([xg[mask] - x, yg[mask] - y, np.ones(int(mask.sum()))])
        coef, *_ = np.linalg.lstsq(A, win[mask], rcond=None)
        a, b = float(coef[0]), float(coef[1])
        slope_deg = float(np.degrees(np.arctan(np.hypot(a, b))))
        # Downslope direction = -(a, b) in (east, north); azimuth from north clockwise.
        aspect_deg = float(np.degrees(np.arctan2(-a, -b)) % 360.0)
        return elev, slope_deg, aspect_deg
