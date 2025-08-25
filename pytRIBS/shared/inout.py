import os
from datetime import datetime
import getpass
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import math
from shapely.geometry import Point
import json


class InOut:
    "Shared Class for managing reading and writing tRIBS files"
    def read_point_files(self):
        """
        Returns Pandas dataframe of nodes or point used in tRIBS mesh.
        """

        file_path = self.options['pointfilename']['value']

        node_points = []
        node_z = []
        node_bc = []

        with open(file_path, 'r') as file:
            lines = file.readlines()
            num_points = lines.pop(0)

            for line in lines:
                parts = line.strip().split()
                if len(parts) != 0:
                    x, y, z, bc = map(float, parts)
                    node_points.append(Point(x, y))
                    node_z.append(z)
                    node_bc.append(bc)

            node_features = {'bc': node_bc, 'geometry': node_points, 'elevation': node_z}
            if self.meta["EPSG"] is not None:
                nodes = gpd.GeoDataFrame(node_features, crs=self.meta["EPSG"])
            else:
                nodes = gpd.GeoDataFrame(node_features)
                print("Coordinate Reference System (CRS) was not added to the GeoDataFrame")
            return nodes
    @staticmethod
    def write_point_file(nodes_gdf, output_file):
        """
        Write a points file from a GeoDataFrame of nodes.

        Parameters:
        - nodes_gdf: GeoDataFrame
            GeoDataFrame containing nodes with 'geometry', 'elevation', and 'bc' columns.
        - output_file: str
            Path to the output points file.

        Returns:
        None
        """
        # Ensure 'geometry' column is of Point type
        if not isinstance(nodes_gdf['geometry'].iloc[0], Point):
            nodes_gdf['geometry'] = nodes_gdf['geometry'].apply(Point)

        # Open the output file for writing
        with open(output_file, 'w') as file:
            # Write the number of points as the first line
            file.write(f"{len(nodes_gdf)}\n")

            # Write each point's x, y, z, and bc values on separate lines
            for _, row in nodes_gdf.iterrows():
                x, y = row['geometry'].x, row['geometry'].y
                z, bc = row['elevation'], int(row['bc'])
                file.write(f"{x} {y} {z} {bc}\n")

    def write_input_file(self, output_file_path, detailed=False):
        """
        Writes .in file for tRIBS model simulation.
        :param self:
        :param output_file_path: Location to write input file to.
        :param detailed: Option to print input file with option descriptions and related info.
        """
        if detailed:
            tags = ['time', 'mesh', 'flow', 'hydro', 'spatial', 'meterological', 'output', 'forecast', 'stochastic',
                    'restart', 'parallel']
            headers = {'time': 'Time Variables', 'mesh': 'Mesh Options', 'flow': 'Routing Variables',
                       'hydro': 'Hydrologic Processes',
                       'spatial': 'Spatial Data Inputs', 'meterological': 'Meterological Options and Data',
                       'output': 'Model Output Paths and Options',
                       'forecast': 'Forecast Mode', 'stochastic': 'Stochastic Mode', 'restart': 'Restart Mode',
                       'parallel': 'Parallel Mode'}

            meta = "This is a template input file for tRIBS 5.2.0. The file is divided in sections mirroring documentation\n" + \
                   "found at: https://tribshms.readthedocs.io/en/latest/man/Model%20Input%20File.html#input-file-options\n" + \
                   "Some values are already provided in the line following the keyword, where keywords are shown in all caps.\n" + \
                   "Where values are not provided are marked by the string \"Update!\". Following the value is a short description of \n" + \
                   "what the keyword does, alongside available options. Note: only values required by given a option must be specified.\n\n"

            current_datetime = datetime.now()
            current_user = getpass.getuser()

            formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

            with open( output_file_path, 'w') as file:

                file.write(f"Created by: {current_user}\n")
                file.write(f"On: {formatted_datetime}\n\n")

                string = 'Input File Template for tRIBS Version 5.2'
                underline = '=' * len(string)
                file.write(f'{underline}\n{string}\n{underline}\n\n')
                file.write(meta)

                for tag in tags:
                    underline = '=' * len(f'Section: {headers[tag]}')
                    file.write(f'{underline}\nSection: {headers[tag]}\n{underline}\n\n')
                    result = [item for item in self.options.values() if
                              any(tag in _tag for _tag in item.get("tags", []))]

                    for dictionary in result:
                        keyword = dictionary['keyword']
                        file.write(f'{keyword}\n')
                        val = dictionary['value']
                        if val is not None:
                            file.write(f"{dictionary['value']}\n\n")
                        else:
                            file.write(f"Update!\n\n")

                        description = dictionary['describe']
                        if description is not None:
                            file.write(f"Description:\n{description}\n")
                        else:
                            file.write(f"None\n")
                        file.write(f" \n")
        else:
            with open(output_file_path, 'w') as output_file:

                current_datetime = datetime.now()
                current_user = getpass.getuser()

                formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

                output_file.write(f"Created by: {current_user}\n")
                output_file.write(f"On: {formatted_datetime}\n\n")

                for key, subdict in self.options.items():
                    if "keyword" in subdict and "value" in subdict:
                        keyword = subdict["keyword"]
                        value = subdict["value"]
                        if value is None:
                            value = ""
                        output_file.write(f"{keyword}\n")
                        output_file.write(f"{value}\n\n")



    def read_precip_sdf(self, file_path=None):
        """
        Returns list of precip stations, where information from each station is stored in a dictionary.
        :param file_path: Reads from options["hydrometstations"]["value"], but can be separately specified.
        :return: List of dictionaries.
        """

        if file_path is None:
            file_path = self.options["gaugestations"]["value"]

            if file_path is None:
                print(self.options["gaugestations"]["key_word"] + "is not specified.")
                return None

        station_list = []

        with open(file_path, 'r') as file:
            lines = file.readlines()

        metadata = lines.pop(0)
        num_stations, num_parameters = map(int, metadata.strip().split())

        for l in lines:
            station_info = l.strip().split()
            if len(station_info) == 7:
                station_id, file_path, lat, long, record_length, num_params, elevation = station_info
                station = {
                    "station_id": station_id,
                    "file_path": file_path,
                    "y": float(lat),
                    "x": float(long),
                    "record_length": int(record_length),
                    "num_parameters": int(num_params),
                    "elevation": float(elevation)
                }
                station_list.append(station)

        if len(station_list) != num_stations:
            print("Error: Number of stations does not match the specified count.")

        return station_list

    @staticmethod
    def read_precip_station(file_path):
        """
        Returns pandas dataframe of precipitation from a station specified by file_path.
        :param file_path: Flat file with columns Y M D H R
        :return: Pandas dataframe
        """
        # TODO add var for specifying Station ID
        df = pd.read_csv(file_path, header=0, sep=r"\s+")
        df.rename(columns={'Y': 'year', 'M': 'month', 'D': 'day', 'H': 'hour'}, inplace=True)
        df['date'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
        df.drop(['year', 'month', 'day', 'hour'], axis=1, inplace=True)

        return df

    @staticmethod
    def write_precip_sdf(station_list, output_file_path):
        """
        Writes a list of precip stations to a flat file.
        :param station_list: List of dictionaries containing station information.
        :param output_file_path: Output flat file path.
        """
        with open(output_file_path, 'w') as file:
            # Write metadata line
            metadata = f"{len(station_list)} {len(station_list[0])}\n"
            file.write(metadata)

            # Write station information
            for station in station_list:
                line = f"{station['station_id']} {station['file_path']} {station['y']} {station['x']} " \
                       f"{station['record_length']} {station['num_parameters']} {station['elevation']}\n"
                file.write(line)

    @staticmethod
    def write_precip_station(df, output_file_path):
        """
        Converts a DataFrame with 'date' and 'R' columns to flat file format with columns Y M D H R.
        :param df: Pandas DataFrame with 'date' and 'R' columns.
        :param output_file_path: Output flat file path.
        """
        # Extract Y, M, D, and H from the 'date' column
        df['Y'] = df['date'].dt.year
        df['M'] = df['date'].dt.month
        df['D'] = df['date'].dt.day
        df['H'] = df['date'].dt.hour

        # Reorder columns
        df = df[['Y', 'M', 'D', 'H', 'R']]

        # Write DataFrame to flat file
        df.to_csv(output_file_path, sep=' ', index=False)

    def read_met_sdf(self, file_path=None):
        """
        Returns list of met stations, where information from each station is stored in a dictionary.
        :param file_path: Reads from options["hydrometstations"]["value"], but can be separately specified.
        :return: List of dictionaries.
        """
        if file_path is None:
            file_path = self.options["hydrometstations"]["value"]

            if file_path is None:
                print(self.options["hydrometstations"]["key_word"] + "is not specified.")
                return None

        station_list = []

        with open(file_path, 'r') as file:
            lines = file.readlines()

        metadata = lines.pop(0)
        num_stations, num_parameters = map(int, metadata.strip().split())

        for l in lines:
            station_info = l.strip().split()

            if len(station_info) == 10:
                station_id, file_path, lat, y, long, x, gmt, record_length, num_params, other = station_info
                station = {
                    "station_id": station_id,
                    "file_path": file_path,
                    "lat_dd": float(lat),
                    "x": float(x),
                    "long_dd": float(long),
                    "y": float(y),
                    "GMT": int(gmt),
                    "record_length": int(record_length),
                    "num_parameters": int(num_params),
                    "other": other
                }
                station_list.append(station)

        if len(station_list) != num_stations:
            print("Error: Number of stations does not match the specified count.")

        return station_list

    @staticmethod
    def read_met_station(file_path):
        """
        Reads a meteorological station data file and processes it into a pandas DataFrame with a datetime index.

        Parameters
        ----------
        file_path : str
            Path to the meteorological station data file. The file should be in a space-separated format with columns for
            year, month, day, and hour.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the meteorological data with a single 'date' column as a datetime index, and the remaining
            columns from the input file.

        Notes
        -----
        - The function expects the input file to have columns 'Y', 'M', 'D', and 'H' for year, month, day, and hour, respectively.
        - The columns for year, month, day, and hour are converted into a single 'date' column of datetime type.
        - The original columns 'Y', 'M', 'D', and 'H' are dropped from the DataFrame after the datetime conversion.
        """
        # TODO add var for specifying Station ID and doc
        df = pd.read_csv(file_path, header=0, sep=r'\s+')
        # convert year, month, day to datetime and drop columns
        df.rename(columns={'Y': 'year', 'M': 'month', 'D': 'day', 'H': 'hour'}, inplace=True)
        df['date'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
        df = df.drop(['year', 'month', 'day', 'hour'], axis=1)
        return df
    @staticmethod
    def write_met_station(df, output_file_path):
        """
        Converts a DataFrame with 'date' and 'PA','TD' or 'RH' or 'VP','XC','US','TA','TS','NR' columns to flat file format.
        See tRIBS documentation for more details on weather station data structure (i.e. *mdf files).
        :param df: Pandas DataFrame with 'date' and 'R' columns.
        :param output_file_path: Output flat file path.
        """
        # Extract Y, M, D, and H from the 'date' column
        df['Y'] = df['date'].dt.year
        df['M'] = df['date'].dt.month
        df['D'] = df['date'].dt.day
        df['H'] = df['date'].dt.hour

        # Format 'D' and 'H' columns with zero-padding
        df['D'] = df['D'].apply(lambda x: str(x).zfill(2))
        df['H'] = df['H'].apply(lambda x: str(x).zfill(2))

        # Check which column ('TD', 'RH', or 'VP') is present in the DataFrame
        present_column = next((col for col in ['TD', 'RH', 'VP'] if col in df.columns), None)

        if present_column is not None:
            # Reorder columns
            df = df[['Y', 'M', 'D', 'H', 'PA', present_column, 'XC', 'US', 'TA', 'IS', 'TS', 'NR']]

            # Write DataFrame to flat file with tab as separator
            df.to_csv(output_file_path, sep='\t', index=False)
        else:
            print("Error: One of 'TD', 'RH', or 'VP' column must be present in the DataFrame.")


    @staticmethod
    def write_met_sdf(output_file_path, station_list):
        """
        Writes a list of meteorological stations to a flat file (i.e. *.sdf file).
        :param station_list: List of dictionaries containing station information.
        :param output_file_path: Output flat file path.
        """
        with open(output_file_path, 'w') as file:
            # Write metadata line
            metadata = f"{len(station_list)} {len(station_list[0])}\n"
            file.write(metadata)

            # Write station information
            for station in station_list:
                line = f"{station['station_id']} {station['file_path']} {station['lat_dd']} {station['y']} {station['long_dd']} {station['x']} " \
                       f"{station['GMT']} {station['record_length']} {station['num_parameters']} {station['other']}\n"
                file.write(line)

    def read_landuse_table(self, file_path=None):
        """
        Returns list of dictionaries for each type of landuse specified in the .ldt file.

        Land Use Reclassification Table Structure (*.ldt, see tRIBS documentation for more details)
        #Types	#Params
        ID	a	b1	 P	S	K	b2	Al	 h	Kt	Rs	V LAI theta*_s theta*_t

        """
        if file_path is None:
            file_path = self.landtablename["value"]

            if file_path is None:
                print(self.landtablename["key_word"] + "is not specified.")
                return

        landuse_list = []

        with open(file_path, 'r') as file:
            lines = file.readlines()

        metadata = lines.pop(0)
        num_types, num_params = map(int, metadata.strip().split())
        param_standard = 15

        if num_params != param_standard:
            print(f"The number parameters in {file_path} do not conform with standard landuse .sdt format.")
            return

        for l in lines:
            land_info = l.strip().split()

            if len(land_info) == param_standard:
                _id, a, b_1, p, s, k, b_2, al, h, kt, rs, v, lai, tstar_s, tstar_t = land_info
                station = {
                    "ID": _id,
                    "a": a,
                    "b1": b_1,
                    "P": p,
                    "S": s,
                    "K": k,
                    "b2": b_2,
                    "Al": al,
                    "h": h,
                    "Kt": kt,
                    "Rs": rs,
                    "V": v,
                    "LAI": lai,
                    "theta*_s": tstar_s,
                    "theta*_t": tstar_t
                }
                landuse_list.append(station)

        if len(landuse_list) != num_types:
            print("Error: Number of land types does not match the specified count.")

        return landuse_list
    @staticmethod
    def write_landuse_table(landuse_list,file_path):
        """
        Writes out Land Use Reclassification Table(*.ldt) file with the following format:
        #Types	#Params
        ID	a	b1	 P	S	K	b2	Al	 h	Kt	Rs	V LAI theta*_s theta*_t

        :param landuse_list: List of dictionaries containing land information specified by .ldt structure above.
        :param file_path: Path to save *.sdt file.
        """
        param_standard = 15

        with open(file_path, 'w') as file:
            # Write metadata line
            metadata = f"{len(landuse_list)} {param_standard}\n"
            file.write(metadata)

            # Write station information
            for type in landuse_list:
                line = f"{str(type['ID'])} {str(type['a'])} {str(type['b1'])} {str(type['P'])} {str(type['S'])} {str(type['K'])} " \
                       f" {str(type['b2'])} {str(type['Al'])} {str(type['h'])} {str(type['Kt'])} {str(type['Rs'])} {str(type['V'])}" \
                       f" {str(type['LAI'])} {str(type['theta*_s'])} {str(type['theta*_t'])}\n"
                file.write(line)

    def read_grid_data_file(self, grid_type):
        """
        Returns dictionary with content of a specified Grid Data File (.gdf)
        :param grid_type: string set to "weather", "soil", of "land", with each corresponding to HYDROMETGRID, SCGRID, LUGRID
        :return: dictionary containg keys and content: "Number of Parameters","Latitude", "Longitude","GMT Time Zone", "Parameters" (a  list of dicts)
        """

        if grid_type == "weather":
            option = self.options["hydrometgrid"]["value"]
        elif grid_type == "soil":
            option = self.options["scgrid"]["value"]
        elif grid_type == "land":
            option = self.options["lugrid"]["value"]

        parameters = []

        with open(option, 'r') as file:
            num_parameters = int(file.readline().strip())
            location_info = file.readline().strip().split()
            latitude, longitude, gmt_timezone = location_info

            variable_count = 0

            for line in file:
                parts = line.strip().split()
                if len(parts) == 3:
                    variable_name, raster_path, raster_extension = parts
                    variable_count += 1

                    # path_components = raster_path.split(os.path.sep)
                    #
                    # # Exclude the last directory as its actually base name
                    # raster_path = os.path.sep.join(path_components[:-1])

                    # if raster_path != "NO_DATA":
                    #     if not os.path.exists(raster_path+'/'+raster_extension):
                    #         print(
                    #             f"Warning: Raster file not found for Variable '{variable_name}': {raster_path}")
                    #         raster_path = None
                    #     elif os.path.getsize(raster_path) == 0:
                    #         print(
                    #             f"Warning: Raster file is empty for Variable '{variable_name}': {raster_path}")
                    #         raster_path = None
                    # elif raster_path == "NO_DATA":
                    #     print(
                    #         f"Warning: No rasters set for variable '{variable_name}'")
                    #     raster_path = None

                    parameters.append({
                        'Variable Name': variable_name,
                        'Raster Path': raster_path,
                        'Raster Extension': raster_extension
                    })
                else:
                    print(f"Skipping invalid line: {line}")

            if variable_count > num_parameters:
                print(
                    "Warning: The number of variables exceeds the number of parameters. This variable has been reset "
                    "in dictionary.")

        return {
            'Number of Parameters': variable_count,
            'Latitude': latitude,
            'Longitude': longitude,
            'GMT Time Zone': gmt_timezone,
            'Parameters': parameters
        }

    @staticmethod
    def write_grid_data_file(grid_file, data):
        """
        Writes the content of a dictionary to a specified Grid Data File (.gdf)
        :param grid_file: path to write out grid file to.
        :param data: dictionary containing keys and content: "Number of Parameters", "Latitude", "Longitude", "GMT Time Zone", "Parameters" (a list of dicts)
        :return: None
        """

        with open(grid_file, 'w') as file:
            # Write number of parameters
            file.write(f"{data['Number of Parameters']}\n")

            # Write location info (Latitude, Longitude, GMT Time Zone)
            file.write(f"{data['Latitude']} {data['Longitude']} {data['GMT Time Zone']}\n")

            # Write parameters
            for param in data['Parameters']:
                variable_name = param['Variable Name']
                raster_path = param['Raster Path']
                raster_extension = param['Raster Extension']

                # # Check if the raster path exists, and if it doesn't, set it to "NO_DATA"
                # if not os.path.exists(os.path.join(raster_path, raster_extension)):
                #     raster_path = "NO_DATA"

                file.write(f"{variable_name} {raster_path} {raster_extension}\n")

    @staticmethod
    def read_ascii(file_path):
        """
        Returns dictionary containing 'data', 'profile', and additional metadata.
        :param file_path: Path to ASCII (or other formats) raster.
        :return: Dict
        """
        raster = {}

        # Open the raster file using rasterio
        with rasterio.open(file_path) as src:
            # Read the raster data as a NumPy array
            raster['data'] = src.read(1)  # Assuming a single band raster, adjust accordingly

            # Access the metadata
            raster['profile'] = src.profile

        return raster

    @staticmethod
    def read_json(file_path):
        with open(file_path, 'r') as f:
            input = json.load(f)
        return input

    @staticmethod
    def write_ascii(raster_dict, output_file_path,dtype='float32', decimals=None):
        """
        Writes raster data and metadata from a dictionary to an ASCII raster file.
        :param raster_dict: Dictionary containing 'data', 'profile', and additional metadata.
        :param output_file_path: Output ASCII raster file path.
        :param dtype: Data type for the output raster (default is 'float32').
        :param decimals: Optional integer specifying number of decimal places for raster values.
        """
        # Extract data and metadata from the dictionary
        data = raster_dict['data']
        profile = raster_dict['profile']

        # Remove unsupported creation options
        unsupported_options = ['blockxsize', 'blockysize', 'tiled', 'interleave']
        for option in unsupported_options:
            profile.pop(option, None)

        profile.update(dtype=dtype)

        if 'nodata' not in profile:
            profile['nodata'] = -9999.0

        if 'driver' not in profile or profile['driver'] != 'AAIGrid':
            # Update the profile for ASCII raster format
            profile.update(
                count=1,
                #compress=None,
                driver='AAIGrid'  # ASCII Grid format
            )


        # Replace nan values with nodata value
        data = np.where(np.isnan(data), profile['nodata'], data)

        # Write the data and metadata to the ASCII raster file
        with rasterio.open(output_file_path, 'w', **profile) as dst:
            dst.write(data, 1)

        # ensure that header has the following format:
        # ncols
        # nrows
        # xllcorner
        # yllcorner
        # cellsize
        # NODATA_value

        with open(output_file_path, 'r') as file:
            lines = file.readlines()

        updated_lines = []
        replaced = False
    
        # Use enumerate to get the line number (i)
        for i, line in enumerate(lines):
            # Check if we are in the header (first 6 lines)
            if i < 6:
                if line.startswith("dx") or line.startswith("dy"):
                    if not replaced:
                        updated_lines.append("cellsize " + str(math.ceil(float(line.split()[1]))) + "\n")
                        replaced = True
                else:
                    updated_lines.append(line)
            # We are in the data part of the file (everything after line 5)
            else:
                # Check if the user wants to format the numbers
                if decimals is not None and isinstance(decimals, int):
                    # Create a format string, e.g., "%.0f" or "%.1f"
                    fmt = f"%.{decimals}f"
                    # Check for empty lines, which can happen at the end of files
                    if line.strip():
                        new_line = " ".join([fmt % float(n) for n in line.strip().split()]) + "\n"
                        updated_lines.append(new_line)
                else:
                    # If decimals is None, just add the original line back
                    updated_lines.append(line)

        # Write the fully updated content back to the file
        with open(output_file_path, 'w') as file:
            file.writelines(updated_lines)

    @staticmethod
    def write_node_file(node_ids, file_path):
        # Open the file for writing
        with open(file_path, 'w') as file:
            # Write the total number of items at the top
            file.write(f"{len(node_ids)}\n")

            # Write each item on a separate line
            for number in node_ids:
                file.write(f"{number}\n")