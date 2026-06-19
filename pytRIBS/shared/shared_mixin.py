# shared_mixin.py
import os
import glob
import sys

import numpy as np

import geopandas as gpd
import pandas as pd
import math
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.geometry import Polygon
from rasterio.transform import from_origin
from rasterio.features import rasterize


class Meta:
    """
    Class for project metadata.
    """
    def __init__(self):
        self.meta = {"Name": None, "Scenario": None, "EPSG": None}


class Shared:
    """
    Shared methods betweens the pytRIBS Classes.
    """

    def read_input_file(self, file_path):
        """
        Reads .in file for tRIBS model simulation and assigns values to options attribute.
        :param file_path: Path to .in file.

        """
        with open(file_path, 'r') as file:
            lines = file.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()  # Remove leading/trailing whitespace
            for key in self.options.keys():
                # Convert key to lowercase for case-insensitive comparison
                key_lower = key.lower()
                # Convert line to lowercase for case-insensitive comparison
                line_lower = line.lower()
                if line_lower.startswith(key_lower):
                    # Extract the portion of the line after the key
                    if i + 1 < len(lines):
                        # Extract the value from the next line
                        value = lines[i + 1].strip()
                        self.options[key]['value'] = value
            i += 1

    @staticmethod
    def convert_to_datetime(starting_date):
        """
        Returns a pandas date-time object.

        :param starting_date: The start date of a given model simulation, note needs to be in tRIBS format.
        :type starting_date: str
        :rtupe: A pandas Timestamp object
        """
        month = int(starting_date[0:2])
        day = int(starting_date[3:5])
        year = int(starting_date[6:10])
        hour = int(starting_date[11:13])
        minute = int(starting_date[14:16])
        date = pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute)
        return date

    def read_voi_file(self, filename=None):
        """
        Returns GeoDataFrame containing voronoi polygons from tRIBS model domain.
        :param filename: Set to read _reach file specified from OUTFILENAME,but can be changed.
        :return: GeoDataFrame

        """

        if filename is None:
            filename = self.options["outfilename"]["value"] + "_voi"

        ids = []
        polygons = []
        points = []
        line_count = 0

        if os.path.exists(filename):
            with open(filename, 'r') as file:
                current_id = None
                current_voi_points = []
                current_node_points = []

                for line in file:

                    line_count += 1

                    if line.strip() != "END":
                        parts = line.strip().split(',')

                        if parts:
                            if len(parts) == 3:
                                id_, x, y = map(float, parts)
                                current_id = id_
                                current_node_points.append((x, y))
                            elif len(parts) == 2:
                                x, y = map(float, parts)
                                current_voi_points.append((x, y))

                    elif line.strip() == "END":

                        if current_id is None:
                            break  ## catch end of file w/ two ends in a row

                        ids.append(current_id)
                        polygons.append(Polygon(current_voi_points))
                        points.append(Point(current_node_points))

                        current_id = None
                        current_voi_points = []
                        current_node_points = []

            if line_count <= 1:
                print(filename + "is empty.")
                return None

            # Package Voronoi
            if not ids or not polygons:
                raise ValueError("No valid data found in " + filename)

            voi_features = {'ID': ids, 'geometry': polygons}
            node_features = {'ID': ids, 'geometry': points}

            if self.meta["EPSG"] is not None:
                voi = gpd.GeoDataFrame(voi_features, crs=self.meta["EPSG"])
                nodes = gpd.GeoDataFrame(node_features, crs=self.meta["EPSG"])
            else:
                voi = gpd.GeoDataFrame(voi_features)
                nodes = gpd.GeoDataFrame(node_features)
                print("Coordinate Reference System (CRS) was not added to the GeoDataFrame")
            return voi, nodes

        else:
            print("Voi file not found.")
            return None

    @staticmethod
    def read_node_list(file_path):
        """
        Returns node list provide by .dat file.

        The node list can be further modified or used for reading in element/pixel files and subsequent processing.

        :param file_path: Relative or absolute file path to .dat file.
        :type file_path: str
        :return: List of nodes specified by .dat file
        :rtype: list

        """
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # Initialize an empty list to store the IDs
            node_ids = []

            # Check if the file is empty or has invalid content
            if not lines:
                return node_ids

            # Parse the first column as the size of the array
            size = int(lines[0].strip())

            # Extract IDs from the remaining lines
            for line in lines[1:]:
                id_value = line.strip()
                node_ids.append(id_value)

            # Ensure the array has the specified size
            if len(node_ids) != size:
                print("Warning: Array size does not match the specified size in the file.")

            return node_ids
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
            return []

    def read_reach_file(self, filename=None):
        """
        Returns GeoDataFrame containing reaches from tRIBS model domain.
        :param filename: Set to read _reach file specified from OUTFILENAME,but can be changed.
        :return: GeoDataFrame
        """

        if filename is None:
            filename = self.options["outfilename"]["value"] + "_reach"

        with open(filename, 'r') as file:
            lines = file.readlines()

        features = []
        current_id = None
        coordinates = []

        for line in lines:
            line = line.strip()
            if line == "END":
                if current_id is not None:
                    line_string = LineString(coordinates)
                    features.append({"ID": current_id, "geometry": line_string})
                    current_id = None
                    coordinates = []
            else:
                if current_id is None:
                    current_id = int(line)
                else:
                    x, y = map(float, line.split(','))
                    coordinates.append((x, y))
        if self.meta["EPSG"] is not None:
            gdf = gpd.GeoDataFrame(features, crs=self.meta["EPSG"])
        else:
            gdf = gpd.GeoDataFrame(features)
            print("Coordinate Reference System (CRS) was not added to the GeoDataFrame")

        return gdf

    def get_spatial_files(self, suffix="_00d", dtime=0, header=True, colnames=None, single=True):
        """
        Reads and returns spatial output files (Dynamic or Integrated) for tRIBS models.

        As of tRIBS v6.0.0, parallel runs write the same consolidated output files as
        serial runs, so a single file per time step is read regardless of the
        'parallelmode' setting.

        :param str suffix: Either _00d for dynamics outputs or _00i for time-integrated ouputs.
        :param int dtime : Option to specify time step at which to start reading files.
        :param bool header: Set to False if headers are not provided with spatial files.
        :param list colnames: If header = False, column names can be provided here.
        :param bool single: If single = True then only the spatial file specified at dtime is read.
        :return: Dictionary of pandas dataframes keyed by time string.
        """

        # 1. Load Configuration
        runtime = int(self.options["runtime"]["value"])
        spopintrvl = int(self.options["spopintrvl"]["value"])
        outfilename = self.options["outfilename"]["value"]

        dyn_data = {}

        # Calculate time steps to retrieve
        times = [dtime + i * spopintrvl for i in range((runtime - dtime) // spopintrvl + 1)]
        if times[-1] != runtime:
            times.append(runtime)

        for _time in times:
            otime = str(_time).zfill(4)
            target_file = f"{outfilename}.{otime}{suffix}"

            if os.path.exists(target_file):
                try:
                    if header:
                        df = pd.read_csv(target_file, header=0)
                    else:
                        df = pd.read_csv(target_file, header=None, names=colnames)

                    # Sort by ID for consistency
                    if header and 'ID' in df.columns:
                        df = df.sort_values(by='ID')

                    dyn_data[otime] = df

                except pd.errors.EmptyDataError:
                    print(f'The file is empty: {target_file}')
            else:
                print(f"Spatial output file not found: {target_file}")
                if single:
                    break

            if single:
                break

        return dyn_data

    def mesh2vtk(self, outfile):
        """
        Converts mesh data files into a VTK file format for visualization.

        This function reads node, triangle, and elevation data from files and writes them to a VTK file.
        The VTK file will be an unstructured grid dataset containing points and cells, with associated scalar data.

        Parameters
        ----------
        outfile : str
            Path to the output VTK file where the mesh data will be written.

        Returns
        -------
        None

        Notes
        -----
        - The function expects the following files in the directory specified by the 'outfilename' option:
            - A node file with a `.nodes` extension containing node coordinates and boundary codes.
            - A triangle file with a `.tri` extension containing triangle vertex indices.
            - A z-file with a `.z` extension containing elevation values.
        - The node file should contain columns for x, y coordinates, and a boundary code.
        - The triangle file should contain columns for vertex indices of triangles.
        - The z-file should contain elevation values for each node.
        - The output VTK file will include point data (coordinates and elevations) and cell data (triangles).
        - Boundary codes are used to set NaN values in the altitude scalars in the VTK file.

        Example
        -------
        >>> self.mesh2vtk('output_mesh.vtk')

        Raises
        ------
        FileNotFoundError
            If the required node, triangle, or z files cannot be found in the specified directory.
        IndexError
            If there is an issue reading data from the node, triangle, or z files, which may indicate file corruption.
        """
        outfilename = self.options["outfilename"]["value"]
        last_slash_index = outfilename.rfind('/')
        directory_path = outfilename[:last_slash_index + 1]

        if os.path.exists(directory_path):
            node_file = glob.glob(directory_path + '*.nodes*')
        else:
            print(f'Cannot find node file at: {directory_path}. Exiting.')
            return

        if os.path.exists(directory_path):
            tri_file = glob.glob(directory_path + '*.tri*')
        else:
            print(f'Cannot find tri file at: {directory_path}. Exiting.')
            return

        if os.path.exists(directory_path):
            z_file = glob.glob(directory_path + '*.z*')
        else:
            print(f'Cannot find z file at: {directory_path}. Exiting.')
            return

        # read in node,tri,z files:
        try:

            with open(node_file[0], 'r') as f:
                lines = f.readlines()  # skip first since it's relic feature

                # Check if there's at least one line
                if lines:
                    num_nodes = int(lines[1])
                    store_nodes = np.zeros((num_nodes, 2))
                    boundary_code = np.zeros((num_nodes, 1))

                    # Iterate from the second line onward
                    for l in range(2, num_nodes + 2):
                        try:
                            line = lines[l].split()
                            store_nodes[l - 2, 0] = float(line[0])
                            store_nodes[l - 2, 1] = float(line[1])
                            boundary_code[l - 2, 0] = float(line[3])
                        except IndexError as e:
                            print(f'Node file may be corrupted, check line {l}')
                            print(f"Error: {e}")
                            sys.exit(1)

            with open(tri_file[0], 'r') as f:
                lines = f.readlines()

                # Check if there's at least one line
                if lines:
                    num_tri = int(lines[1])
                    store_tri = np.zeros((num_tri, 3))

                    # Iterate from the second line onward
                    for l in range(2, num_tri + 2):
                        try:
                            line = lines[l].split()
                            store_tri[l - 2, 0] = float(line[0])
                            store_tri[l - 2, 1] = float(line[1])
                            store_tri[l - 2, 2] = float(line[2])
                        except IndexError as e:
                            print(f'Tri file may be corrupted, check line {l}')
                            print(f"Error: {e}")
                            sys.exit(1)

            with open(z_file[0], 'r') as f:
                lines = f.readlines()

                # Check if there's at least one line
                if lines:
                    num_z = int(lines[1])
                    store_z = np.zeros((num_z, 1))

                    # Iterate from the second line onward
                    for l in range(2, num_z + 2):
                        try:
                            line = lines[l].split()
                            store_z[l - 2, 0] = float(line[0])
                        except IndexError as e:
                            print(f'Z file may be corrupted, check line {l}')
                            print(f"Error: {e}")
                            sys.exit(1)

            with open(outfile, 'w') as f:
                f.write("# vtk DataFile Version 3.0\n")
                f.write("tRIBS\n")
                f.write("ASCII\n")
                f.write("DATASET UNSTRUCTURED_GRID\n")
                f.write('POINTS {0:10d} float\n'.format(num_nodes))
                for I in range(num_nodes):
                    f.write(
                        "{0:15.5f} {1:15.5f} {2:15.5f}\n".format(store_nodes[I, 0], store_nodes[I, 1], store_z[I, 0]))

                f.write("CELLS {0:10d} {1:10d}\n".format(num_tri, 4 * num_tri))
                for I in range(num_tri):
                    f.write('3 {0:10d} {1:10d} {2:10d}\n'.format(int(store_tri[I, 0]), int(store_tri[I, 1]),
                                                                 int(store_tri[I, 2])))

                f.write("CELL_TYPES {0:10d}\n".format(num_tri))
                for I in range(num_tri):
                    f.write("5\n")

                f.write("POINT_DATA {0:10d}\n".format(num_nodes))
                f.write("SCALARS Altitude float 1\n")
                f.write("LOOKUP_TABLE default\n")
                for I in range(num_nodes):
                    if boundary_code[I, 0] == 1:
                        f.write('NaN' + '\n')
                    else:
                        f.write(str(store_z[I, 0]) + "\n")

                f.write('SCALARS BC_code float 1\n')
                f.write('LOOKUP_TABLE BC_LUT\n')

                for I in range(num_nodes):
                    f.write(str(float(boundary_code[I, 0])) + '\n')

                # possible to add additional scalars
                # f.write("SCALARS Shear_stress float 1\n")
                # f.write("LOOKUP_TABLE default\n")
                # for I in range(num_nodes):
                #     f.write(str(TABTAU[I, 0]) + "\n")

        except FileNotFoundError:
            return


    def get_invariant_properties(self):
        """
        Reads and processes invariant spatial properties from tRIBS output.

        As of tRIBS v6.0.0, parallel runs write the same consolidated output files as
        serial runs, so the integrated spatial file (*_00i) and the Voronoi (_voi) file
        are each read as a single consolidated file regardless of `parallelmode`.

        The method does the following:
        - Reads the integrated spatial variables (*_00i) at runtime.
        - Computes area-weighted `weight` values from the `VAr` column.
        - Loads the Voronoi polygons.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Example
        -------
        >>> obj.get_invariant_properties()

        Raises
        ------
        ValueError
            If there are issues reading the spatial or Voronoi data.
        """

        # Read Integrated Spatial Variables (*_00i)
        runtime_val = int(self.options['runtime']['value'])
        
        # Read the file (returns a dict)
        temp_dict = self.get_spatial_files(suffix="_00i", dtime=runtime_val, single=True)
        
        runtime_str = str(runtime_val).zfill(4)

        if temp_dict and runtime_str in temp_dict:
            self.int_spatial_vars = temp_dict[runtime_str]
            
            # tRIBS outputs Voronoi Area (VAr), but we need the normalized weight for stats.
            if 'VAr' in self.int_spatial_vars.columns:
                 self.int_spatial_vars['weight'] = self.int_spatial_vars.VAr.values / self.int_spatial_vars.VAr.sum()
            else:
                 print("Warning: 'VAr' column not found in spatial output. Cannot calculate weights.")
        else:
            print('Unable To Read Integrated Spatial File (*_00i).')
            self.int_spatial_vars = None

        # read in voronoi file once
        voi = self.read_voi_file()
        if voi is not None:
            self.voronoi, _ = voi
        else:
            print('Unable To Load Voi File.')
            self.voronoi = None

    @staticmethod
    def grid_geodataframe(gdf, value_column, cell_size, nodata_value=-9999.0, fill_nodata_with_mean=False):
        """
        Rasterizes a GeoDataFrame using area-weighted averaging.

        This method is calculating the value of each raster cell based on the proportional area of all voronoi 
        polygons that overlap it.

        Parameters
        ----------
        gdf : GeoDataFrame
            The GeoDataFrame that contains the voronoi polygons and outputs 
            to rasterize. Must have a valid CRS.
        value_column : str
            The name of the column in the gdf to use for the raster values.
        cell_size : float
            The desired cell size (resolution) of the output raster.
        nodata_value : float, optional
            The value for pixels that do not fall within any polygon. A value 
            of -9999.0 is usually appropriate for tRIBS.
        fill_nodata_with_mean : bool, optional
            If True, any remaining nodata cells in the final raster will be
            filled with the mean of all valid data cells. Defaults to False.

        Returns
        -------
        dict or None
            A dictionary containing 'data' and 'profile' for write_ascii.
            Returns None if the input GeoDataFrame has no CRS defined.

        Example
        -------
        >>> dynamic_data_dict = results.get_spatial_files(suffix="_00d", dtime=final_runtime, single=True)
        >>> gdf_final_state = results.voronoi.merge(dynamic_data_dict[str(final_runtime).zfill(4)], on='ID')
        >>> final_gw_raster_dict = results.grid_geodataframe( gdf=gdf_final_state, value_column='Nwt', cell_size=30.0)

        Raises
        ------
        Error
            If there is not a valid CRF attached to the GeoDataFrame.
        """

        # 0. Check for a valid CRS
        if gdf.crs is None:
            print("ERROR: Input GeoDataFrame has no CRS defined.")
            print("Please set one in the pytRIBS project class metadata or using `your_gdf.set_crs('EPSG:XXXX')` before proceeding.")
            return None

        # 1. Create a grid of square polygons (pixels) with an automatic buffer
        data_min_x, data_min_y, data_max_x, data_max_y = gdf.total_bounds
        data_width = data_max_x - data_min_x
        data_height = data_max_y - data_min_y
        buffer_from_scale = 0.02 * (data_width + data_height) / 2
        buffer_from_pixel = cell_size
        final_buffer = max(buffer_from_scale, buffer_from_pixel)
        min_x = math.floor((data_min_x - final_buffer) / cell_size) * cell_size
        max_x = math.ceil((data_max_x + final_buffer) / cell_size) * cell_size
        min_y = math.floor((data_min_y - final_buffer) / cell_size) * cell_size
        max_y = math.ceil((data_max_y + final_buffer) / cell_size) * cell_size
        width = int(round((max_x - min_x) / cell_size))
        height = int(round((max_y - min_y) / cell_size))
        
        # Create the grid of square polygons (pixels)
        x_coords = np.arange(min_x, max_x, cell_size)
        y_coords = np.arange(min_y, max_y, cell_size)
        
        polygons = []
        pixel_ids = []
        for i, y in enumerate(y_coords):
            for j, x in enumerate(x_coords):
                polygons.append(Polygon([(x, y), (x + cell_size, y), (x + cell_size, y + cell_size), (x, y + cell_size)]))
                pixel_ids.append(i * width + j)

        grid_gdf = gpd.GeoDataFrame({'pixel_id': pixel_ids}, geometry=polygons, crs=gdf.crs)

        # Intersect the Voronoi polygons with the pixel grid
        intersection_gdf = gpd.overlay(grid_gdf, gdf, how='intersection', keep_geom_type=False)

        # Filter for only Polygons for clean math
        intersection_gdf = intersection_gdf[intersection_gdf.geometry.type == 'Polygon']

        # Calculate the area of each small intersected piece
        intersection_gdf['overlap_area'] = intersection_gdf.geometry.area

        # Use pandas groupby to calculate the area-weighted mean for each pixel
        def weighted_mean(group):
            weights = group['overlap_area']
            values = group[value_column]
            return np.average(values, weights=weights)

        pixel_values = intersection_gdf.groupby('pixel_id').apply(weighted_mean)

        # Create the final numpy array and populate it
        final_data = np.full((height, width), nodata_value, dtype=np.float32)

        for pixel_id, value in pixel_values.items():
            row = height - 1 - (pixel_id // width)
            col = pixel_id % width
            if 0 <= row < height and 0 <= col < width:
                final_data[row, col] = value
        
        # Optionally fill nodata values 
        if fill_nodata_with_mean:
            nodata_mask = (final_data == nodata_value)
            valid_pixels = final_data[~nodata_mask]
            
            if valid_pixels.size > 0:
                mean_value = np.mean(valid_pixels)
                nodata_count = np.sum(nodata_mask)
                print(f"INFO: Filling {nodata_count} nodata cells with the mean value: {mean_value:.4f}")
                final_data[nodata_mask] = mean_value
            else:
                print("WARNING: No valid data found in the raster. Cannot fill nodata values.")

        # Create the profile for the output raster
        transform = from_origin(min_x, max_y, cell_size, cell_size)
        
        profile = {
            'driver': 'AAIGrid', 'count': 1, 'height': height, 'width': width,
            'transform': transform, 'crs': gdf.crs, 'dtype': 'float32',
            'nodata': nodata_value
        }
        
        return {'data': final_data, 'profile': profile}