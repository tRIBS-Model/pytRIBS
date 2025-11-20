import re
import os
import matplotlib.cm

from pyproj import Transformer
import numpy as np
import geopandas as gpd
from owslib.wcs import WebCoverageService
from rosetta import rosetta, SoilData
from scipy.optimize import curve_fit
from pytRIBS.shared.inout import InOut
from pytRIBS.shared.aux import Aux
from timezonefinder import TimezoneFinder
from datetime import datetime
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
import pytz
import requests
from rasterio.merge import merge


class SoilProcessor:
    """
    Methods for pytRIBS Soil Class.
    """
    # Assigning references to the methods
    @staticmethod
    def discrete_colormap(N, base_cmap=None):
        cmap = Aux.discrete_cmap(N, base_cmap)
        return cmap
    @staticmethod
    def _fillnodata(files, overwrite=False, resample_pixel_size=None, resample_method='nearest', **kwargs):
        Aux.fillnodata(files,
                       overwrite=overwrite,
                       resample_pixel_size=resample_pixel_size,
                       resample_method=resample_method,
                       **kwargs)

    @staticmethod
    def _write_ascii(raster_dict, output_file_path, dtype='float32'):
        InOut.write_ascii(raster_dict, output_file_path, dtype)

    @staticmethod
    def _read_ascii(file_path):
        raster = InOut.read_ascii(file_path)
        return raster

    @staticmethod
    def _read_json(raster_dict, output_file_path, dtype='float32'):
        input = InOut.read_json(raster_dict, output_file_path, dtype)
        return input

    def generate_uniform_groundwater(self, watershed_boundary, value, filename=None):
        """
        Generates a uniform groundwater raster file within the specified watershed boundary.

        This method creates a raster file with uniform groundwater values over the extent of the given
        watershed boundary. The raster file can be written to a specified filename or to a default filename
        from an attribute if no filename is provided.

        Parameters
        ----------
        watershed_boundary : GeoDataFrame
            A GeoDataFrame representing the watershed boundary. It should include a 'bounds' property to
            determine the raster extent.
        value : float
            The uniform groundwater value to be written to the raster file.
        filename : str, optional
            The path to the output file. If not provided, the filename will be retrieved from the `gwaterfile`
            attribute of the object.

        Returns
        -------
        None

        Notes
        -----
        - If `filename` is not provided, the method attempts to use the `gwaterfile` attribute from the object.
        - The raster file is written with a single cell covering the entire extent of the watershed boundary.
        - The raster format includes the number of columns, rows, and cell size, as well as the specified groundwater value.

        Example
        -------
        >>> obj.generate_uniform_groundwater(watershed_gdf, 10.0, 'output_file.txt')

        Raises
        ------
        ValueError
            If the `filename` cannot be determined and `gwaterfile` is not set in the object.
        """

        if filename is None:
            gwfile = self.gwaterfile['value']
            if gwfile is None:
                print("A filename must be provided if a value has not been supplied to the attribute gwaterfile.")
                return
            filename = gwfile

        gdf = watershed_boundary

        bounds = gdf.bounds
        xllcorner, yllcorner, xmax, ymax = bounds

        cellsize = max(xmax - xllcorner, ymax - yllcorner)

        with open(filename, 'w') as f:
            f.write("ncols\t1\n")
            f.write("nrows\t1\n")
            f.write(f"xllcorner\t{xllcorner:.8f}\n")
            f.write(f"yllcorner\t{yllcorner:.7f}\n")
            f.write(f"cellsize\t{cellsize}\n")
            f.write("NODATA_value\t-9999\n")
            f.write(f"{value}\n")

    def read_soil_table(self, textures=False, file_path=None):
        """
        Reads a Soil Reclassification Table Structure (*.sdt) file.

        The .sdt file contains parameters such as:
        - ID, Ks, thetaS, thetaR, m, PsiB, f, As, Au, n, ks, Cs, and optionally soil texture.

        The method reads the specified soil table file and returns a list of dictionaries representing
        the soil types and their associated parameters.

        Parameters
        ----------
        textures : bool, optional
            If True, the method will read and include texture classes in the returned data. Default is False.
        file_path : str, optional
            The file path to the soil table (.sdt file). If not provided, it defaults to `self.soiltablename["value"]`.
            If `self.soiltablename["value"]` is also None, the method will print an error message and return None.

        Returns
        -------
        list of dict or None
            A list of dictionaries, where each dictionary represents a soil type and its associated parameters.
            Each dictionary contains the following keys:
            - "ID" : str, soil type ID
            - "Ks" : float, saturated hydraulic conductivity
            - "thetaS" : float, saturated water content
            - "thetaR" : float, residual water content
            - "m" : float, parameter related to soil pore size distribution
            - "PsiB" : float, bubbling pressure
            - "f" : float, hydraulic decay parameter
            - "As" : float, saturated anisotropy ratio
            - "Au" : float, unsaturated anisotropy ratio
            - "n" : float, porosity of the soil
            - "ks" : float, volumetric heat conductivity
            - "Cs" : float, soil heat capacity
            - "Texture" : str, texture class (only if `textures=True` is passed)

            If the file does not conform to the standard .sdt format or the number of soil types doesn't match the specified count,
            an error message will be printed, and the function returns None.

        Examples
        --------
        Reading a soil table without textures:

        >>> soil_list = read_soil_table(textures=False, file_path="path/to/soil_table.sdt")
        >>> print(soil_list[0]["Ks"])
        0.0001

        Reading a soil table with textures:

        >>> soil_list = read_soil_table(textures=True, file_path="path/to/soil_table_with_textures.sdt")
        >>> print(soil_list[0]["Texture"])
        'Sandy Loam'
        """
        if file_path is None:
            file_path = self.soiltablename["value"]

            if file_path is None:
                print(self.soiltablename["key_word"] + "is not specified.")
                return None

        soil_list = []

        with open(file_path, 'r') as file:
            lines = file.readlines()

        metadata = lines.pop(0)
        num_types, num_params = map(int, metadata.strip().split())
        param_standard = 12

        if textures:
            param_standard += 1

        if num_params != param_standard:
            print(f"The number parameters in {file_path} do not conform with standard soil .sdt format.")
            return

        for l in lines:
            soil_info = l.strip().split()

            if len(soil_info) == param_standard:
                if textures:
                    _id, ks, theta_s, theta_r, m, psi_b, f, a_s, a_u, n, _ks, c_s, textures = soil_info
                    station = {
                        "ID": _id,
                        "Ks": ks,
                        "thetaS": theta_s,
                        "thetaR": theta_r,
                        "m": m,
                        "PsiB": psi_b,
                        "f": f,
                        "As": a_s,
                        "Au": a_u,
                        "n": n,
                        "ks": _ks,
                        "Cs": c_s,
                        "Texture": textures
                    }
                else:
                    _id, ks, theta_s, theta_r, m, psi_b, f, a_s, a_u, n, _ks, c_s = soil_info
                    station = {
                        "ID": _id,
                        "Ks": ks,
                        "thetaS": theta_s,
                        "thetaR": theta_r,
                        "m": m,
                        "PsiB": psi_b,
                        "f": f,
                        "As": a_s,
                        "Au": a_u,
                        "n": n,
                        "ks": _ks,
                        "Cs": c_s
                    }

                soil_list.append(station)

        if len(soil_list) != num_types:
            print("Error: Number of soil types does not match the specified count.")
        return soil_list

    @staticmethod
    def write_soil_table(soil_list, file_path, textures=False):
        """
        Writes out Soil Reclassification Table(*.sdt) file with the following format:
        #Types #Params
        ID Ks thetaS thetaR m PsiB f As Au n ks Cs

        :param soil_list: List of dictionaries containing soil information specified by .sdt structure above.
        :param file_path: Path to save *.sdt file.
        :param textures: Optional True/False for writing texture classes to the .sdt file.

        """
        param_standard = 12

        if textures:
            param_standard += 1

        with open(file_path, 'w') as file:
            # Write metadata line
            metadata = f"{len(soil_list)} {param_standard}\n"
            file.write(metadata)

            # Write station information
            for type in soil_list:

                if textures:
                    line = f"{str(type['ID'])}   {str(type['Ks'])}    {str(type['thetaS'])}    {str(type['thetaR'])}    {str(type['m'])}    {str(type['PsiB'])}    " \
                           f"{str(type['f'])}    {str(type['As'])}    {str(type['Au'])}    {str(type['n'])}    {str(type['ks'])}    {str(type['Cs'])} {str(type['Texture'])}\n"
                else:
                    line = f"{str(type['ID'])}   {str(type['Ks'])}    {str(type['thetaS'])}    {str(type['thetaR'])}    {str(type['m'])}    {str(type['PsiB'])}    " \
                           f"{str(type['f'])}    {str(type['As'])}    {str(type['Au'])}    {str(type['n'])}    {str(type['ks'])}    {str(type['Cs'])}\n"

                file.write(line)

    def get_soil_grids(self, bbox, depths, soil_vars, stats, replace=False):
        def retrieve_soil_data(self, bbox, depths, soil_vars, stats):
            """
            Retrieves soil data from the ISRIC WCS service, saves it as GeoTIFF files, and returns a list of paths to the downloaded files.

            Parameters
            ----------
            bbox : list of float
                The bounding box coordinates in the format [x1, y1, x2, y2], where:
                - x1 : float, minimum x-coordinate (longitude or easting)
                - y1 : float, minimum y-coordinate (latitude or northing)
                - x2 : float, maximum x-coordinate (longitude or easting)
                - y2 : float, maximum y-coordinate (latitude or northing)
            depths : list of str
                List of soil depths to retrieve data for. Each depth should be specified as a string in the format 'depth_min-depth_max', e.g., '0-5cm', '5-15cm'.
            soil_vars : list of str
                List of soil variables to retrieve from the ISRIC service. Examples include 'bdod' (bulk density), 'clay', 'sand', 'silt', 'wv1500' (wilting point), etc.
                For a full list of variables, see the ISRIC documentation at https://maps.isric.org/.
            stats : list of str
                List of statistics to compute for each variable and depth. Typically includes 'mean', but other quantiles or statistics may be available.
                For more information on prediction quantiles, see the ISRIC SoilGrids FAQ: https://www.isric.org/explore/soilgrids/faq-soilgrids.

            Returns
            -------
            list of str
                A list of file paths to the downloaded GeoTIFF files.

            Examples
            --------
            To retrieve soil data for specific depths and variables within a bounding box:

            >>> bbox = [387198, 3882394, 412385, 3901885]  # x1, y1, x2, y2 (e.g., UTM coordinates)
            >>> depths = ['0-5cm', '5-15cm', '15-30cm', '30-60cm', '60-100cm']
            >>> soil_vars = ['bdod', 'clay', 'sand', 'silt', 'wv1500', 'wv0033', 'wv0010']
            >>> stats = ['mean']
            >>> file_paths = retrieve_soil_data(bbox, depths, soil_vars, stats)
            >>> print(file_paths)
            ['path/to/downloaded_file_1.tif', 'path/to/downloaded_file_2.tif', ...]
            """

        target_epsg = self.meta['EPSG']
        if target_epsg is None:
            print("No EPSG code found. Please update model attribute .meta['EPSG'].")
            return

        # Sanitize EPSG code
        match = re.search(r'(\d+)', str(target_epsg))
        if match:
            target_epsg_code = int(match.group(1))
        else:
            print(f"Invalid EPSG code: {target_epsg}")
            return

        # Make sure sg250 directory exists in the CURRENT working directory
        data_dir = 'sg250'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print(f"Directory '{data_dir}' created.")

        # Calculate WGS84 Bounding Box for the Request
        # We transform the requested BBOX (Model CRS) -> WGS84 (EPSG:4326)
        transformer = Transformer.from_crs(f"EPSG:{target_epsg_code}", "EPSG:4326", always_xy=True)
        minx, miny, maxx, maxy = bbox
        lon_min, lat_min = transformer.transform(minx, miny)
        lon_max, lat_max = transformer.transform(maxx, maxy)
        
        # WCS 1.0.0 expects [minx, miny, maxx, maxy]
        bbox_wgs84 = [lon_min, lat_min, lon_max, lat_max]

        files = []
        complete = False

        print('Downloading data from SoilGrids (WGS84) and reprojecting...')
        
        for var in soil_vars:
            # Connect to ISRIC WCS
            try:
                wcs = WebCoverageService(f'http://maps.isric.org/mapserv?map=/map/{var}.map', version='1.0.0', timeout=300)
            except Exception as e:
                print(f"Could not connect to service for {var}: {e}")
                continue

            for depth in depths:
                for stat in stats:
                    soil_key = f'{var}_{depth}_{stat}'
                    filename = f'{soil_key}.tif'
                    
                    files.append(filename) 
                    
                    # Define where to save it
                    final_path = os.path.join(data_dir, filename)
                    temp_path = os.path.join(data_dir, f'temp_{soil_key}.tif')

                    # Skip if exists
                    if os.path.isfile(final_path) and not replace:
                        complete = True # Mark as success if file exists
                        continue

                    try:
                        # 1. Download Raw WGS84 Data
                        # resx/resy 0.002083 deg is approx 250m
                        response = wcs.getCoverage(
                            identifier=soil_key, 
                            crs='EPSG:4326', 
                            bbox=bbox_wgs84, 
                            resx=0.002083, 
                            resy=0.002083, 
                            format='GEOTIFF_INT16', 
                            timeout=120
                        )
                        
                        with open(temp_path, 'wb') as f:
                            f.write(response.read())

                        # 2. Reproject Locally using Rasterio
                        with rasterio.open(temp_path) as src:
                            transform, width, height = calculate_default_transform(
                                src.crs, f'EPSG:{target_epsg_code}', src.width, src.height, *src.bounds
                            )
                            kwargs = src.meta.copy()
                            kwargs.update({
                                'crs': f'EPSG:{target_epsg_code}',
                                'transform': transform,
                                'width': width,
                                'height': height
                            })

                            with rasterio.open(final_path, 'w', **kwargs) as dst:
                                reproject(
                                    source=rasterio.band(src, 1),
                                    destination=rasterio.band(dst, 1),
                                    src_transform=src.transform,
                                    src_crs=src.crs,
                                    dst_transform=transform,
                                    dst_crs=f'EPSG:{target_epsg_code}',
                                    resampling=Resampling.nearest
                                )
                        
                        # Clean up temp file
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        
                        complete = True

                    except Exception as e:
                        print(f"Failed to process {soil_key}: {e}")
                        # Clean up garbage
                        if os.path.exists(final_path) and os.path.getsize(final_path) < 2000:
                             os.remove(final_path)

        if complete:
            print('Download of SoilGrids250 data complete')
        else:
            print('No SoilGrids250 data was downloaded check inputs or set replace == False.')

        return files

    def create_soil_map(self, grid_input, output=None):
        """
        Writes out an ASCII file with soil classes assigned by soil texture classification.

        Parameters
        ----------
        grid_input : list of dict or str
            If a dictionary list, each dictionary should contain keys "grid_type" and "path" for each soil property.
            The format of the dictionary list is as follows:

            ::

                [{'type': 'sand', 'path': 'path/to/sand_grid'},
                 {'type': 'clay', 'path': 'path/to/clay_grid'}]

            If a string is provided, it is treated as the path to a JSON configuration file containing grid types and output file paths.

        output : str, optional
            The file path where the ASCII soil map will be saved. If not provided, the default output file will be used (`'soil_class.soi'`).

        Returns
        -------
        None
            This function does not return any value. It writes an ASCII file with soil classifications to the specified `output` path.

        Examples
        --------
        To create a soil map using a dictionary list:

        >>> grid_input = [{'type': 'sand', 'path': 'path/to/sand_grid'},
        ...               {'type': 'clay', 'path': 'path/to/clay_grid'}]
        >>> create_soil_map(grid_input, output="path/to/soil_map.asc")

        To create a soil map using a configuration file:

        >>> grid_input = "path/to/config_file.json"
        >>> create_soil_map(grid_input)
        """

        if isinstance(grid_input, str):
            # Read configuration from the file
            config = self._read_json(grid_input)
            grids = config['grid_types']
            output_file = config['output_files']
        elif isinstance(grid_input, list):
            # Use provided dictionary
            grids = grid_input
            output_file = output or ['soil_class.soi']

        def soiltexturalclass(sand, clay):
            """
            Returns USDA soil textural class given percent sand and clay.
            :param sand:
            :param clay:
            """

            silt = 100 - sand - clay

            if sand + clay > 100 or sand < 0 or clay < 0:
                raise ValueError('Inputs add up to more than 100% or are negative')
            elif silt + 1.5 * clay < 15:
                textural_class = 1  # sand
            elif silt + 1.5 * clay >= 15 and silt + 2 * clay < 30:
                textural_class = 2  # loamy sand
            elif (clay >= 7 and clay < 20 and sand > 52 and silt + 2 * clay >= 30) or (
                    clay < 7 and silt < 50 and silt + 2 * clay >= 30):
                textural_class = 3  # sandy loam
            elif clay >= 7 and clay < 27 and silt >= 28 and silt < 50 and sand <= 52:
                textural_class = 4  # loam
            elif (silt >= 50 and clay >= 12 and clay < 27) or (silt >= 50 and silt < 80 and clay < 12):
                textural_class = 5  # silt loam
            elif silt >= 80 and clay < 12:
                textural_class = 6  # silt
            elif clay >= 20 and clay < 35 and silt < 28 and sand > 45:
                textural_class = 7  # 'sandy clay loam'
            elif clay >= 27 and clay < 40 and sand > 20 and sand <= 45:
                textural_class = 8  # 'clay loam'
            elif clay >= 27 and clay < 40 and sand <= 20:
                textural_class = 9  # 'silty clay loam'
            elif clay >= 35 and sand > 45:
                textural_class = 10  # 'sandy clay'
            elif clay >= 40 and silt >= 40:
                textural_class = 11  # 'silty clay'
            elif clay >= 40 > silt and sand <= 45:
                textural_class = 12
            else:
                textural_class = 'na'

            return textural_class

        # Loop through specified file paths
        texture_data = []

        for cnt, g in enumerate(grids):
            grid_type, path = g['type'], g['path']
            print(f"Ingesting {grid_type} from: {path}")
            geo_tiff = self._read_ascii(path)
            array = geo_tiff['data']
            size = array.shape

            if cnt == 0:
                texture_data = np.zeros((2, size[0], size[1]))

            if grid_type == 'sand':
                array = array / 1000 * 100  # convert SSC from g/kg to % SSC
                texture_data[0, :, :] = array
            elif grid_type == 'clay':
                array = array / 1000 * 100  # convert SSC from g/kg to % SSC
                texture_data[1, :, :] = array

        soil_class = np.zeros((1, size[0], size[1]), dtype=int)

        for i in range(0, size[0]):
            for j in range(0, size[1]):
                # Organize array for input into packag
                data = [texture_data[x, i, j] for x in np.arange(0, 2)]
                sand = data[0]
                clay = data[1]
                soil_class[0, i, j] = int(soiltexturalclass(sand, clay))

        soil_classification = {1: 'sand', 2: 'loamy_sand', 3: 'sandy_loam', 4: 'loam', 5: 'silt_loam', 6: 'silt',
                               7: 'sandy_clay_loam', 8: 'clay_loam', 9: 'silty_clay_loam', 10: 'sandy_clay',
                               11: 'silty_clay', 12: 'clay'}

        classes = np.unique(soil_class[0])

        filtered_classes = {}

        count = 1
        for key in soil_classification.keys():
            if key in classes:
                filtered_classes[count] = soil_classification[key]
                soil_class[soil_class == key] = int(count)
                count += 1

        # Need to re-write soil map so that classes start from 1 and sequentially thereafter
        soi_raster = {'data': soil_class[0], 'profile': geo_tiff['profile']}
        self._write_ascii(soi_raster, output_file, dtype='int16')

        # create soil table with  nodata for rasyes
        parameters = ['ID', 'Ks', 'thetaS', 'thetaR', 'm', 'PsiB', 'f', 'As', 'Au', 'n', 'ks', 'Cs', 'Texture']
        soil_list = []
        count = 1
        nodata = 9999.99
        ndefined = 'undefined'

        for key, item in filtered_classes.items():
            d = {}
            for p in parameters:
                # reset ID
                if p == 'ID':
                    d.update({p: count})
                elif p == 'Texture':
                    d.update({p: item})
                # give textural class that need to be updated via user or calibration
                elif p in (['As', 'Au', 'Cs', 'ks']):
                    d.update({p: ndefined})

                # set grid data to nodata value in table
                else:
                    d.update({p: nodata})
            count += 1

            soil_list.append(d)

        return soil_list

    def process_raw_soil(self, grid_input, output=None, ks_only=False):
        """
        Writes ASCII grids for Ks, theta_s, theta_r, psib, and m from gridded soil data for sand, silt, clay, bulk density, and volumetric water content at 33 and 1500 kPa.

        Parameters
        ----------
        grid_input : list of dict or str
            If a dictionary list, each dictionary should contain the keys "grid_type" and "path" for each soil property.
            The format of the dictionary list follows this structure:

            ::

                [{'type': 'sand_fraction', 'path': 'path/to/grid'},
                 {'type': 'silt_fraction', 'path': 'path/to/grid'},
                 {'type': 'clay_fraction', 'path': 'path/to/grid'},
                 {'type': 'bulk_density', 'path': 'path/to/grid'},
                 {'type': 'vwc_33', 'path': 'path/to/grid'},
                 {'type': 'vwc_1500', 'path': 'path/to/grid'}]

            If a string is provided, it is treated as the path to a JSON configuration file.

        output : list, optional
            List of output file names for different soil properties. The list should have exactly 5 file names corresponding to different soil properties.

        ks_only : bool, optional
            If `True`, only write rasters for Ks. This is useful when using the `compute_decay_ks` function.

        Notes
        -----
        - The `grid_input` key should contain a list of dictionaries, each specifying a grid type and its corresponding file path.
        - The `output` list should contain exactly 5 output file names for different soil properties.
        - The file paths in the `grid_input` list should be valid, and the `output` list should have the correct number of file names.

        Examples
        --------
        To write all soil property grids:

        >>> grid_input = [{'type': 'sand_fraction', 'path': 'path/to/sand_grid'},
        ...               {'type': 'silt_fraction', 'path': 'path/to/silt_grid'},
        ...               {'type': 'clay_fraction', 'path': 'path/to/clay_grid'},
        ...               {'type': 'bulk_density', 'path': 'path/to/bulk_density_grid'},
        ...               {'type': 'vwc_33', 'path': 'path/to/vwc_33_grid'},
        ...               {'type': 'vwc_1500', 'path': 'path/to/vwc_1500_grid'}]
        >>> output = ['ks_output.asc', 'theta_s_output.asc', 'theta_r_output.asc', 'psib_output.asc', 'm_output.asc']
        >>> your_function_name(grid_input, output)

        To write only Ks raster:

        >>> grid_input = "path/to/config.json"
        >>> output = ['ks_output.asc']
        >>> your_function_name(grid_input, output, ks_only=True)
        """

        # Check if grid_input is a string (path to a config file)
        if isinstance(grid_input, str):
            # Read configuration from the file
            config = self._read_json(grid_input)
            grids = config['grid_types']
            output_files = config['output_files']
        elif isinstance(grid_input, list):
            # Use provided dictionary
            grids = grid_input
            output_files = output or ['Ks.asc', 'theta_r.asc', 'theta_s.asc', 'psib.asc', 'm.asc']
        else:
            print(
                'Invalid input format. Provide either a list of dictionaries specifying type and path, or a path to a configuration file.')
            return

        # Check if each file specified in the dictionary or config exists
        for g in grids:
            grid_type, path = g['type'], g['path']
            if not os.path.isfile(path):
                raise FileNotFoundError(f'Cannot find: {path} for grid type: {grid_type}')

        # Ensure output_files is a list of the correct size (5 elements)
        if output_files is not None and (not isinstance(output_files, list) or len(output_files) != 5):
            print('Output must be a list with 5 elements.')
            return

        sg250_data = None
        size = None
        geo_tiff = None

        # Loop through specified file paths
        for cnt, g in enumerate(grids):
            grid_type, path = g['type'], g['path']
            print(f"Ingesting {grid_type} from: {path}")
            geo_tiff = self._read_ascii(path)
            array = geo_tiff['data']
            size = array.shape

            if cnt == 0:
                sg250_data = np.zeros((6, size[0], size[1]))

            # each z layer follows:[sa (%), si (%), cl (%), bd (g/cm3), th33, th1500]
            if grid_type == 'sand':
                array = array / 1000 * 100  # convert SSC from g/kg to % SSC
                sg250_data[0, :, :] = array
            elif grid_type == 'silt':
                array = array / 1000 * 100  # convert SSC from g/kg to % SSC
                sg250_data[1, :, :] = array
            elif grid_type == 'clay':
                array = array / 1000 * 100  # convert SSC from g/kg to % SSC
                sg250_data[2, :, :] = array
            elif grid_type == 'bdod':
                array = array / 100  # convert bulk density from cg/cm3 to g/cm3
                sg250_data[3, :, :] = array
            elif grid_type == 'wv0033':
                array = array / 1000  # convert bulk density from cg/cm3 to g/cm3
                sg250_data[4, :, :] = array
            elif grid_type == 'wv1500':
                array = array / 1000  # convert bulk density from cg/cm3 to g/cm3
                sg250_data[5, :, :] = array

        profile = geo_tiff['profile']

        # Initialize parameter grids, 3 grids - 1 mean values, 2 std deviations, 3 code/flag
        theta_r, theta_s, ks, psib, m = np.zeros((3, *size)), np.zeros((3, *size)), np.zeros((3, *size)), np.zeros(
            (1, *size)), np.zeros((1, *size))

        # Loop through raster's and compute soil properties using rosetta-soil package
        # Codes/Flags
        # 2	sand, silt, clay (SSC)
        # 3	SSC + bulk density (BD)
        # 4	SSC + BD + field capacity water content (TH33)
        # 5	SSC + BD + TH33 + wilting point water content (TH1500)
        # -1 no result returned, inadequate or erroneous data
        # each z layer follows:[sa (%), si (%), cl (%), bd (g/cm3), th33, th1500]
        # i.e SoilData([sa (%), si (%), cl (%), bd (g/cm3), th33, th1500])

        for i in range(0, size[0]):
            for j in range(0, size[1]):
                # Organize array for input into packag
                data = [sg250_data[x, i, j] for x in np.arange(0, 6)]
                soil_data = SoilData.from_array([data])
                mean, stdev, codes = rosetta(3, soil_data)  # apply Rosetta version 3
                theta_r[:, i, j] = [mean[0, 0], stdev[0, 0], codes[0]]
                theta_s[:, i, j] = [mean[0, 1], stdev[0, 1], codes[0]]
                # Convert ks from log10(cm/day) into mm/hr
                ks[:, i, j] = [(10 ** mean[0, 4]) * 10 / 24, (10 ** stdev[0, 4]) * 10 / 24, codes[0]]

                # Alpha parameter from rosetta corresponds approximately to the inverse of the air-entry value, cm−1
                # https://doi.org/10.1029/2019MS001784
                # Convert from log10(cm) into -1/mm
                psib[0, i, j] = -1 / (10 ** mean[0, 2]) * 10

                # Pore-size Distribution can be calculated from n using m = n - 1
                # http://dx.doi.org/10.4236/ojss.2012.23025
                # Convert from log10(n) into n
                m[0, i, j] = (10 ** mean[0, 3]) - 1

        # for now only write out mean values
        soil_prop = [ks[0, :, :], theta_r[0, :, :], theta_s[0, :, :], psib[0, :, :], m[0, :, :]]

        if ks_only:
            soi_raster = {'data': soil_prop[0], 'profile': profile}
            self._write_ascii(soi_raster, output_files[0])
        else:
            for soil_property, name in zip(soil_prop, output_files):
                soi_raster = {'data': soil_property, 'profile': profile}
                self._write_ascii(soi_raster, name)
    
    def process_polaris_parameters(self, grid_input, output_files, ks_only=False):
        """
        Writes ASCII grids for Ks, theta_s, theta_r, psib, and m by converting POLARIS 
        gridded soil data into tRIBS-compatible formats and units.

        Parameters
        ----------
        grid_input : list of dict or str
            If a dictionary list, each dictionary should contain the keys "type" and "path" for each soil property.
            The format of the dictionary list follows this structure:

            ::

                [{'type': 'ksat', 'path': 'path/to/ksat_grid'},
                 {'type': 'theta_s', 'path': 'path/to/theta_s_grid'},
                 {'type': 'theta_r', 'path': 'path/to/theta_r_grid'},
                 {'type': 'lambda', 'path': 'path/to/lambda_grid'},
                 {'type': 'hb', 'path': 'path/to/hb_grid'}]

            If a string is provided, it is treated as the path to a JSON configuration file.

        output_files : list
            List of output file names for different soil properties. 
            If `ks_only=False`, the list must have exactly 5 file names in this order:
            ['Ks', 'theta_r', 'theta_s', 'psib', 'm'].
            If `ks_only=True`, the list should contain only 1 file name for Ks.

        ks_only : bool, optional
            If `True`, only write rasters for Ks. This is useful when processing multiple depths 
            specifically for the `compute_ks_decay` function. Default is `False`.

        Notes
        -----
        This function performs specific physical conversions required to translate POLARIS 
        probabilistic soil data to tRIBS inputs:

        1. **Ksat**: Converted from log10(cm/hr) to arithmetic mm/hr.
        2. **Bubbling Pressure (hb/psib)**: Converted from log10(kPa) to arithmetic mm H2O.

        Examples
        --------
        To write all soil property grids for tRIBS:

        >>> grid_input = [{'type': 'ksat', 'path': 'polaris/ksat_0-5_mean.tif'},
        ...               {'type': 'theta_s', 'path': 'polaris/thetas_0-5_mean.tif'},
        ...               {'type': 'theta_r', 'path': 'polaris/thetar_0-5_mean.tif'},
        ...               {'type': 'lambda', 'path': 'polaris/lamda_0-5_mean.tif'},
        ...               {'type': 'hb', 'path': 'polaris/hb_0-5_mean.tif'}]
        >>> output = ['Ks.asc', 'theta_r.asc', 'theta_s.asc', 'psib.asc', 'm.asc']
        >>> obj.process_polaris_parameters(grid_input, output)

        To write only Ks raster (e.g., for deep layers):

        >>> grid_input = [{'type': 'ksat', 'path': 'polaris/ksat_60-100_mean.tif'}]
        >>> output = ['Ks_60-100cm.asc']
        >>> obj.process_polaris_parameters(grid_input, output, ks_only=True)
        """
        
        # Read grids based on input dict
        grid_data = {}
        profile = None
        
        for g in grid_input:
            # type maps to POLARIS var name
            g_type = g['type']
            path = g['path']
            print(f"Processing POLARIS parameter {g_type} from {path}")
            
            geo_tiff = self._read_ascii(path)
            grid_data[g_type] = geo_tiff['data']
            if profile is None:
                profile = geo_tiff['profile']

        size = grid_data['ksat'].shape
        
        # Initialize outputs
        ks = np.zeros(size)
        theta_r = np.zeros(size)
        theta_s = np.zeros(size)
        psib = np.zeros(size)
        m = np.zeros(size)
        
        # Perform Unit Conversions
        # Ksat: POLARIS is log10(cm/hr). tRIBS needs mm/hr.
        # Formula: 10^(val) * 10
        ks = (10 ** grid_data['ksat']) * 10.0
        
        if not ks_only:
            # Theta S / R: Direct match (m3/m3)
            theta_s = grid_data['theta_s']
            theta_r = grid_data['theta_r']
            
            # Pore Index (lambda/m). 
            # POLARIS provides 'lambda' for the Brooks-Corey model. Which is what tRIBS needs
            m = grid_data['lambda']

            # Bubbling Pressure (Psi_b / hb).
            # POLARIS v1.0 'hb' is log10(pressure in kPa).
            # tRIBS needs -mm head.
            # 1 kPa ~= 101.97 mm H2O.
            # Formula: 10^(val) * 101.97
            psib = -(10 ** grid_data['hb']) * 101.97

        # Write Outputs
        # Order expected: [Ks, theta_r, theta_s, psib, m]
        soil_prop = [ks, theta_r, theta_s, psib, m]

        if ks_only:
            soi_raster = {'data': ks, 'profile': profile}
            self._write_ascii(soi_raster, output_files[0])
        else:
            for soil_property, name in zip(soil_prop, output_files):
                soi_raster = {'data': soil_property, 'profile': profile}
                self._write_ascii(soi_raster, name)

    def get_polaris_grids(self, bbox, depths, variables, stats, replace=False):
        """
        Retrieves data from the POLARIS database (Duke University), saves it as GeoTIFF files, and returns a list of paths to the downloaded files.

        Parameters
        ----------
        bbox : list of float
            The bounding box coordinates in the format [x1, y1, x2, y2], where:
            - x1 : float, minimum x-coordinate (longitude or easting)
            - y1 : float, minimum y-coordinate (latitude or northing)
            - x2 : float, maximum x-coordinate (longitude or easting)
            - y2 : float, maximum y-coordinate (latitude or northing)
        depths : list of str
            List of soil depths to retrieve data for. Each depth should be specified as a string in the format 'depth_min-depth_max', e.g., '0-5cm', '5-15cm'.
        soil_vars : list of str
            List of soil variables to retrieve from the HTTP site. Examples include 'bd' (bulk density), 'clay', 'sand', 'silt', etc.
            For a full list of variables, see the readme documentation at http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0/Readme.
        stats : list of str
            List of statistics to compute for each variable and depth. Typically includes 'mean', but other quantiles or statistics may be available.
            For more information on prediction quantiles, see the ISRIC SoilGrids FAQ: https://www.isric.org/explore/soilgrids/faq-soilgrids.

        Returns
        -------
        list of str
            A list of file paths to the downloaded GeoTIFF files.

        Examples
        --------
        To retrieve soil data for specific depths and variables within a bounding box:

        >>> bbox = [387198, 3882394, 412385, 3901885]  # x1, y1, x2, y2 (e.g., UTM coordinates)
        >>> depths = ['0-5cm', '5-15cm', '15-30cm', '30-60cm', '60-100cm']
        >>> soil_vars = ['bdod', 'clay', 'sand', 'silt']
        >>> stats = ['mean']
        >>> file_paths = retrieve_soil_data(bbox, depths, soil_vars, stats)
        >>> print(file_paths)
        ['path/to/downloaded_file_1.tif', 'path/to/downloaded_file_2.tif', ...]
        """

        target_epsg = self.meta['EPSG']
        if target_epsg is None:
            print("No EPSG code found. Please update model attribute .meta['EPSG'].")
            return

        # Sanitize EPSG code
        match = re.search(r'(\d+)', str(target_epsg))
        if match:
            target_epsg_code = int(match.group(1))
        else:
            return

        data_dir = 'polaris'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # Transform Model BBOX to WGS84 to identify which tiles to download (POLARIS available in 1deg tiles)
        transformer = Transformer.from_crs(f"EPSG:{target_epsg_code}", "EPSG:4326", always_xy=True)
        minx, miny, maxx, maxy = bbox
        lon_min, lat_min = transformer.transform(minx, miny)
        lon_max, lat_max = transformer.transform(maxx, maxy)

        # Identify Integer Tile Ranges
        lats = range(int(np.floor(lat_min)), int(np.floor(lat_max)) + 1)
        lons = range(int(np.floor(lon_min)), int(np.floor(lon_max)) + 1)

        # Base configuration
        base_domain = "hydrology.cee.duke.edu"
        base_path = "/POLARIS/PROPERTIES/v1.0"
        
        session = requests.Session()
        retries = requests.adapters.Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        session.mount('http://', requests.adapters.HTTPAdapter(max_retries=retries))
        session.mount('https://', requests.adapters.HTTPAdapter(max_retries=retries))
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        output_files = []
        print('Downloading and processing data from POLARIS...')

        for var in variables:
            for depth in depths:
                remote_depth = depth.replace('-', '_').replace('cm', '')
                stat_val = stats[0] if stats else 'mean'
                
                final_filename = f'{var}_{depth}_{stat_val}.tif'
                final_path = os.path.join(data_dir, final_filename)
                output_files.append(final_filename)

                if os.path.exists(final_path) and not replace:
                    continue

                tile_files = []
                for lat in lats:
                    for lon in lons:
                        lat_str = f"lat{lat}{lat + 1}"
                        lon_str = f"lon{lon}{lon + 1}"
                        remote_fname = f"{lat_str}_{lon_str}.tif"
                        temp_tile_path = os.path.join(data_dir, f"temp_{var}_{depth}_{lat}_{lon}.tif")
                        
                        if os.path.exists(temp_tile_path) and os.path.getsize(temp_tile_path) > 0:
                             tile_files.append(temp_tile_path)
                             continue

                        url = f"http://{base_domain}{base_path}/{var}/{stat_val}/{remote_depth}/{remote_fname}"
                        
                        try:
                            r = session.get(url, headers=headers, stream=True, timeout=30)
                            if r.status_code == 200:
                                with open(temp_tile_path, 'wb') as f:
                                    for chunk in r.iter_content(chunk_size=8192):
                                        f.write(chunk)
                                tile_files.append(temp_tile_path)
                            elif r.status_code == 404:
                                pass
                        except Exception:
                            pass

                if not tile_files:
                    print(f"Warning: No tiles successfully downloaded for {var} at {depth}.")
                    continue

                # Mosaic, Clip, and Reproject
                try:
                    src_files_to_mosaic = [rasterio.open(fp) for fp in tile_files]
                    mosaic, out_trans = merge(src_files_to_mosaic)
                    
                    out_meta = src_files_to_mosaic[0].meta.copy()
                    out_meta.update({
                        "driver": "GTiff",
                        "height": mosaic.shape[1],
                        "width": mosaic.shape[2],
                        "transform": out_trans,
                        "crs": "EPSG:4326"
                    })

                    mosaic_path = os.path.join(data_dir, f"temp_mosaic_{var}_{depth}.tif")
                    with rasterio.open(mosaic_path, "w", **out_meta) as dest:
                        dest.write(mosaic)

                    for src in src_files_to_mosaic:
                        src.close()

                    # Instead of transforming the entire mosaic, we define the transform
                    # based on the requested bbox.
                    
                    with rasterio.open(mosaic_path) as src:
                        # Define target resolution (POLARIS is ~30m)
                        target_res = 30.0 
                        
                        # BBox is [minx, miny, maxx, maxy]
                        dst_width = int((bbox[2] - bbox[0]) / target_res)
                        dst_height = int((bbox[3] - bbox[1]) / target_res)

                        # Create transform strictly for the BBOX
                        dst_transform = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], dst_width, dst_height)

                        kwargs = src.meta.copy()
                        kwargs.update({
                            'crs': f'EPSG:{target_epsg_code}',
                            'transform': dst_transform,
                            'width': dst_width,
                            'height': dst_height
                        })

                        with rasterio.open(final_path, 'w', **kwargs) as dst:
                            reproject(
                                source=rasterio.band(src, 1),
                                destination=rasterio.band(dst, 1),
                                src_transform=src.transform,
                                src_crs=src.crs,
                                dst_transform=dst_transform,
                                dst_crs=f'EPSG:{target_epsg_code}',
                                resampling=Resampling.nearest
                            )
                    
                    if os.path.exists(mosaic_path):
                        os.remove(mosaic_path)

                except Exception as e:
                    print(f"Error processing {var} {depth}: {e}")

                for tf in tile_files:
                    if os.path.exists(tf):
                        os.remove(tf)

        return output_files
    
    def compute_ks_decay(self, grid_input, output=None):
        """
        Produces a raster for the conductivity decay parameter `f`, following Ivanov et al., 2004.

        Parameters
        ----------
        grid_input : dict or str
            If a dictionary, it should contain keys "depth" and "path" for each soil property.
            Depth should be provided in units of mm. The format of the dictionary list should follow
            this structure (from shallowest to deepest):

            ::

                [{'depth': 25, 'path': 'path/to/25_mm_ks'},
                 {...},
                 {'depth': 800, 'path': 'path/to/800_mm_ks'}]

            If a string is provided, it is treated as the path to a configuration file. The configuration
            file must be written in JSON format.
        output : str
            Location to save the raster with the conductivity decay parameter `f`.

        Returns
        -------
        None
            This function saves the generated raster to the specified `output` location.

        Examples
        --------
        To generate a raster using a dictionary for `grid_input`:

        >>> grid_input = [{'depth': 25, 'path': 'path/to/25_mm_ks'},
        ...               {'depth': 800, 'path': 'path/to/800_mm_ks'}]
        >>> output = "path/to/output_raster.tif"
        >>> compute_ks_decay(grid_input, output)

        To generate a raster using a configuration file:

        >>> grid_input = "path/to/config_file.json"
        >>> output = "path/to/output_raster.tif"
        >>> compute_ks_decay(grid_input, output)
        """

        # Check if grid_input is a string (path to a config file)
        if isinstance(grid_input, str):
            # Read configuration from the file
            config = self._read_json(grid_input)
            grids = config['grid_depth']
            output_file = config['output_file']
        elif isinstance(grid_input, list):
            # Use provided dictionary
            grids = grid_input
            output_file = output or ['f.asc']
        else:
            print('Invalid input format. Provide either a list or a path to a configuration file.')
            return

        # Check if each file specified in the dictionary or config exists
        for g in grids:
            grid_type, path = g['depth'], g['path']
            if not os.path.isfile(path):
                raise FileNotFoundError(f'Cannot find: {path} for grid type: {grid_type}')

        ks_data = None
        size = None
        raster = None
        depth_vec = np.zeros(len(grids))

        # Loop through specified file paths
        for cnt, g in enumerate(grids):
            depth, path = g['depth'], g['path']
            print(f"Ingesting Ks grid at {depth} from: {path}")
            raster = self._read_ascii(path)
            array = raster['data']
            depth_vec[cnt] = depth

            if cnt == 0:
                size = array.shape
                ks_data = np.zeros((len(grids), size[0], size[1]))
                ks_data[cnt, :, :] = array
            else:
                ks_data[cnt, :, :] = array

        # Ensure that ks grids are sorted from surface to the deepest depth
        depth_sorted = np.argsort(depth_vec)
        ks_data = ks_data[depth_sorted]

        depth_vec = depth_vec[depth_sorted]
        depth_vec = depth_vec.astype(float)  # ensure float for fitting

        profile = raster['profile']  # for writing later

        # Initialize parameter grids
        f_grid = np.zeros(np.shape(array))  # parameter grid
        fcov = np.zeros(np.shape(array))  # coef of variance grid

        # Loop through raster's and compute soil properties using rosetta-soil package
        for i in range(0, size[0]):
            for j in range(0, size[1]):
                y = np.array([ks_data[n, i, j] for n in np.arange(0, len(grids))])

                if np.any(y == profile['nodata']):
                    f_grid[i, j] = profile['nodata']
                    fcov[i, j] = profile['nodata']
                else:

                    try:
                        # ensure float
                        y = y.astype(float)
                    except ValueError:
                        raise ValueError("Input data must be convertible to float")

                    k0 = y[0] # The surface observed value

                    # Define exponential decay function, Ivanov et al. (2004) eqn 17
                    # We define the function such that K0 is hardcoded because it's the y intercept
                    # The optimizer can only change 'f'.
                    def decay_fixed_intercept(x, f):
                        # Handle x=0 case to avoid division by zero or instability
                        # Ivanov function limit as x->0 is k0
                        
                        # Calculation: K = K0 * ( (f*z) / (exp(f*z) - 1) )
                        # To prevent div/0 errors when x is 0 or f is very small:
                        term = np.zeros_like(x)
                        
                        # For x > 0 (Masking 0 values for safe calculation)
                        mask = (x > 1e-6) 
                        
                        if np.any(mask):
                            val = f * x[mask]
                            # Standard Ivanov Formula
                            term[mask] = k0 * (val / (np.exp(val) - 1.0))
                        
                        # For x near 0, the limit is k0
                        term[~mask] = k0
                        
                        return term

                    # Bounds for f only
                    minf, maxf = 1E-7, 1.0  # Upper bound 1.0 is usually sufficient for soil

                    try:
                        # We only optimize for 'f', so p0 (initial guess) is length 1
                        param, param_cov = curve_fit(decay_fixed_intercept, depth_vec, y, p0=[0.005], bounds=([minf], [maxf]))
                        # Write Curve fitting results to grid
                        f_grid[i, j] = param[0]
                        fcov[i, j] = param_cov[0, 0]
                        
                    except RuntimeError:
                        # If fit fails, default to a small decay or nodata
                        f_grid[i, j] = 0.0001 
                        fcov[i, j] = -9999

        f_raster = {'data': f_grid, 'profile': profile}
        self._write_ascii(f_raster, output_file)

    def _polygon_centroid_to_geographic(self, polygon, utm_crs=None, geographic_crs="EPSG:4326"):
        lat,lon, gmt = Aux.polygon_centroid_to_geographic(self,polygon,utm_crs=utm_crs,geographic_crs=geographic_crs)
        return lat, lon, gmt

    def run_soil_workflow(self, watershed, output_dir, source='ISRIC'):
        """
        Executes the soil processing workflow for the given watershed.

        This method performs a series of operations to process soil data, including filling missing values,
        processing raw soil grids, computing soil parameters, and generating soil maps. It assumes specific
        file structures and parameters for soil processing and outputs the results to the specified directory.

        Parameters
        ----------
        watershed : GeoDataFrame
            A GeoDataFrame representing the watershed boundary. It must contain a 'bounds' property for
            determining the spatial extent of the data.
        output_dir : str
            The directory where output files will be saved.
        source : str
            Specifies the source of gridded soil data. Currently there are two options: ISRIC or POLARIS,
            defaults to ISRIC.

        Returns
        -------
        None

        Notes
        -----
        - The method changes the current working directory to `output_dir` for processing and then restores
          the original directory.
        - Soil grids are processed for various depths and soil variables.
        - The method creates a soil map, writes a soil table file, and generates a configuration file (`scgrid.gdf`)
          with paths to the processed soil data.
        - Workflow steps:
            1. Retrieves soil grid files based on the bounding box from the `watershed` GeoDataFrame.
            2. Fills missing data in the soil grids.
            3. Processes raw soil data for specified depths and variables.
            4. Computes soil hydraulic conductivity decay parameters.
            5. Creates a soil classification map.
            6. Writes a soil table file with texture information.
            7. Generates a configuration file for soil grid data (`scgrid.gdf`).

        Examples
        --------
        To run the soil processing workflow:

        >>> obj.run_soil_workflow(watershed_gdf, '/path/to/output_dir')

        Raises
        ------
        FileNotFoundError
            If any of the required input files cannot be found.

        """

        bounds = watershed.bounds
        bbox = [bounds[0], bounds[1], bounds[2], bounds[3]]
        
        # All depths needed for Ksat / Decay calculation
        depths_all = ['0-5cm', '5-15cm', '15-30cm', '30-60cm', '60-100cm']
        # Surface depth for static parameters
        depth_surface = ['0-5cm']

        tribsvars = ['Ks', 'theta_r', 'theta_s', 'psib', 'm']
        stat = ['mean']

        init_dir = os.getcwd()
        os.chdir(output_dir)

        # Download Grids (Optimized for Source)
        files_to_process = []
        folder = ''

        if source == 'ISRIC':
            folder = 'sg250'
            # ISRIC needs all variables at all depths for Rosetta
            soil_vars = ['sand', 'silt', 'clay', 'bdod', 'wv0033', 'wv1500']
            files = self.get_soil_grids(bbox, depths_all, soil_vars, stat)
            files_to_process = [f'{folder}/{f}' for f in files]

        elif source == 'POLARIS':
            folder = 'polaris'
            
            # Batch 1: Download KSAT for ALL depths (for decay curve)
            print("Downloading Ksat profiles...")
            files_ksat = self.get_polaris_grids(bbox, depths_all, ['ksat'], stat)
            
            # Batch 2: Download Hydraulic/Texture params for surface only
            print("Downloading surface parameters...")
            other_vars = ['theta_s', 'theta_r', 'lambda', 'hb', 'sand', 'clay']
            files_others = self.get_polaris_grids(bbox, depth_surface, other_vars, stat)
            
            # Combine lists for cleaning
            files_to_process = [f'{folder}/{f}' for f in files_ksat + files_others]

        else:
            raise ValueError("Source must be 'ISRIC' or 'POLARIS'")

        # Fill NoData 
        # POLARIS is 30m, ISRIC is 250m
        pixel_size = 250 if source == 'ISRIC' else 30 
        self._fillnodata(files_to_process, resample_pixel_size=pixel_size)

        # Process Parameters
        for depth in depths_all:
            out = [f'{folder}/{x}_{depth}.asc' for x in tribsvars]

            if source == 'ISRIC':
                # ISRIC logic remains: grab all inputs, run Rosetta
                soil_vars_isric = ['sand', 'silt', 'clay', 'bdod', 'wv0033', 'wv1500']
                grids = []
                for soi_var in soil_vars_isric:
                    grids.append({'type': soi_var, 'path': f'{folder}/{soi_var}_{depth}_mean_filled.tif'})
                
                if '0-5' in depth:
                    self.process_raw_soil(grids, output=out)
                else:
                    self.process_raw_soil(grids, output=out, ks_only=True)
            
            elif source == 'POLARIS':
                # POLARIS logic: Handle Surface vs Deep differently
                
                if '0-5' in depth:
                    # Surface: We have ALL variables downloaded
                    grids = []
                    polaris_hydra_vars = ['ksat', 'theta_s', 'theta_r', 'lambda', 'hb']
                    for p_var in polaris_hydra_vars:
                        grids.append({'type': p_var, 'path': f'{folder}/{p_var}_{depth}_mean_filled.tif'})
                    
                    # Process all parameters
                    self.process_polaris_parameters(grids, output_files=out, ks_only=False)
                    
                else:
                    # Deep layers: We ONLY have Ksat downloaded
                    grids = [{'type': 'ksat', 'path': f'{folder}/ksat_{depth}_mean_filled.tif'}]
                    
                    # Process only Ks (ks_only=True)
                    # Note: We pass out[0] because out is [Ks, Tr, Ts, Pb, m]
                    self.process_polaris_parameters(grids, output_files=[out[0]], ks_only=True)

        # Compute Decay 'f'
        # Tested fitting ks to different depth and found that using only using the 1st 3 depths 
        # resulted in a much better fit for the regression. better captures surface decay which 
        # is what this parameter should be representing.
        ks_depths = [0.0001, 50, 150] 
        grid_depth = []
        
        # Mapping string depths to numeric depths for the decay function
        # Note: Ensure these indices match depths_all: 0-5(0), 5-15(1), 15-30(2), 30-60(3)
        for cnt in range(0, 3):
            grid_depth.append({'depth': ks_depths[cnt], 'path': f'{folder}/Ks_{depths_all[cnt]}.asc'})

        ks_decay_param = 'f'
        self.compute_ks_decay(grid_depth, output=f'{folder}/{ks_decay_param}.asc')

        # Create Soil Map
        grids = [{'type': 'sand', 'path': f'{folder}/sand_0-5cm_mean_filled.tif'},
                 {'type': 'clay', 'path': f'{folder}/clay_0-5cm_mean_filled.tif'}]
        
        classes = self.create_soil_map(grids, output=f'{folder}/soil_classes.soi')
        self.write_soil_table(classes, 'soils.sdt', textures=True)

        # Write soil gridded data file
        relative_path = f'{output_dir}/{folder}/'
        scgrid_vars = ['KS', 'TR', 'TS', 'PB', 'PI', 'FD',
                       'PO']  # theta_S (TS) and porosity (PO) are assumed to be the same
        
        # Reset tribsvars list for config writing
        tribsvars = ['Ks', 'theta_r', 'theta_s', 'psib', 'm']
        tribsvars.append(ks_decay_param)
        tribsvars.append('theta_s')
        ref_depth = '0-5cm'

        num_param = len(scgrid_vars)
        lat, lon, gmt = self._polygon_centroid_to_geographic(watershed)
        ext = 'asc'

        with open('scgrid.gdf', 'w') as file:
            file.write(str(num_param) + '\n')
            file.write(f"{str(lat)}    {str(lon)}     {str(gmt)}\n")

            for scgrid, prefix in zip(scgrid_vars, tribsvars):
                if scgrid == 'FD':
                    file.write(f"{scgrid}    {relative_path}{prefix}    {ext}\n")
                else:
                    file.write(f"{scgrid}    {relative_path}{prefix}_{ref_depth}    {ext}\n")

        os.chdir(init_dir)

        # update Soil Class attributes
        self.soiltablename['value'] = f'{output_dir}/soils.sdt'
        self.scgrid['value'] = f'{output_dir}/scgrid.gdf'
        self.soilmapname['value'] = f'{output_dir}/{folder}/soil_classes.soi'
        self.optsoiltype['value'] = 1
