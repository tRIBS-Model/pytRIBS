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
from rasterio.warp import calculate_default_transform, reproject, Resampling
import pytz


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

                    k0 = y[0]

                    # define exponential decay function, Ivanov et al. (2004) eqn 17
                    decay = lambda x, f, k=k0: k * (f * x / (np.exp(f * x) - 1.0))

                    # perform curve fitting, set limits on f and surface Ksat for stability
                    minf, maxf = 1E-7, k0 - 1E-5
                    minks = 1

                    param, param_cov = curve_fit(decay, depth_vec, y, bounds=([minf, maxf], [minks, k0]))

                    # Write Curve fitting results to grid
                    f_grid[i, j] = param[0]
                    fcov[i, j] = param_cov[0, 0]

        f_raster = {'data': f_grid, 'profile': profile}
        self._write_ascii(f_raster, output_file)

    def _polygon_centroid_to_geographic(self, polygon, utm_crs=None, geographic_crs="EPSG:4326"):
        lat,lon, gmt = Aux.polygon_centroid_to_geographic(self,polygon,utm_crs=utm_crs,geographic_crs=geographic_crs)
        return lat, lon, gmt

    def run_soil_workflow(self, watershed, output_dir):
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
        depths = ['0-5cm', '5-15cm', '15-30cm', '30-60cm', '60-100cm']
        soil_vars = ['sand', 'silt', 'clay', 'bdod', 'wv0033', 'wv1500']  # note order is important for processing data
        tribsvars = ['Ks', 'theta_r', 'theta_s', 'psib', 'm']
        stat = ['mean']

        init_dir = os.getcwd()
        os.chdir(output_dir)

        files = self.get_soil_grids(bbox, depths, soil_vars, stat)
        files = [f'sg250/{f}' for f in files]

        self._fillnodata(files, resample_pixel_size=250)

        for depth in depths:
            grids = []
            for soi_var in soil_vars:
                grids.append({'type': soi_var, 'path': f'sg250/{soi_var}_{depth}_mean_filled.tif'})
                out = [f'sg250/{x}_{depth}.asc' for x in tribsvars]

            if '0-5' in depth:
                self.process_raw_soil(grids, output=out)
            else:
                self.process_raw_soil(grids, output=out, ks_only=True)

        ks_depths = [0.0001, 50, 150, 300]
        grid_depth = []

        for cnt in range(0, 4):
            grid_depth.append({'depth': ks_depths[cnt], 'path': f'sg250/Ks_{depths[cnt]}.asc'})

        ks_decay_param = 'f'

        self.compute_ks_decay(grid_depth, output=f'sg250/{ks_decay_param}.asc')

        grids = [{'type': 'sand', 'path': 'sg250/sand_0-5cm_mean_filled.tif'},
                 {'type': 'clay', 'path': 'sg250/clay_0-5cm_mean_filled.tif'}]
        classes = self.create_soil_map(grids, output='sg250/soil_classes.soi')
        self.write_soil_table(classes, 'soils.sdt', textures=True)

        relative_path = f'{output_dir}/sg250/'

        scgrid_vars = ['KS', 'TR', 'TS', 'PB', 'PI', 'FD',
                       'PO']  # theta_S (TS) and porosity (PO) are assumed to be the same
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
        self.soilmapname['value'] = f'{output_dir}/sg250/soil_classes.soi'
        self.optsoiltype['value'] = 1
