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

    def write_input_file(self, output_file_path):
        """
        Writes .in file for tRIBS model simulation, organized into sections
        matching the tRIBS input file template format.
        :param output_file_path: Path to write input file to.
        """
        SECTION_TITLES = {
            1: "Section 1: Model Run Parameters",
            2: "Section 2: Model Run Options",
            3: "Section 3: Model Input Files and Pathnames",
            4: "Section 4: Model Modes",
            5: "Section 5: Restart Mode Options",
            6: "Section 6: Parallel Mode Options",
        }

        HEADER = (
            "##############################################################################\n"
            "##\n"
            "##                    tRIBS Distributed Hydrologic Model\n"
            "##\n"
            "##              TIN-based Real-time Integrated Basin Simulator\n"
            "##                       Ralph M. Parsons Laboratory\n"
            "##                  Massachusetts Institute of Technology\n"
            "##\n"
            "##############################################################################\n"
        )

        def _options_block(entries):
            """Build ## comment lines for keywords that have enumerated options."""
            has_options = [e for e in entries
                           if len([l for l in (e.get('describe') or '').split('\n') if l.strip()]) > 1]
            if not has_options:
                return []
            col = max(len(f"##  {e['keyword']}") for e in has_options) + 3
            lines = []
            for entry in has_options:
                describe = entry.get('describe') or ''
                opt_lines = [l for l in describe.split('\n') if l.strip()][1:]
                prefix = f"##  {entry['keyword']}".ljust(col)
                indent = "##" + " " * (col - 2)
                lines.append(f"{prefix}{opt_lines[0]}\n")
                for opt in opt_lines[1:]:
                    lines.append(f"{indent}{opt}\n")
                lines.append("##\n")
            return lines

        def _section_header(title, option_lines=None):
            parts = [
                "\n##=========================================================================\n",
                "##\n##\n",
                f"##\t\t\t{title}\n",
                "##\n##\n",
            ]
            if option_lines:
                parts.extend(option_lines)
            parts.append("##=========================================================================\n\n")
            return ''.join(parts)

        def _write_entry(f, entry):
            keyword = entry['keyword']
            describe = entry.get('describe') or ''
            inline = describe.split('\n')[0] if describe else ''
            value = entry.get('value')
            if value is None:
                value = ''
            f.write(f"{keyword:<26}{inline}\n")
            f.write(f"{value}\n\n")

        # Group entries by section/subsection, preserving insertion order
        section_data = {}
        extras = []
        for entry in self.options.values():
            sec = entry.get('section')
            if sec is None:
                extras.append(entry)
            else:
                subsec = entry.get('subsection') or ''
                if sec not in section_data:
                    section_data[sec] = {}
                if subsec not in section_data[sec]:
                    section_data[sec][subsec] = []
                section_data[sec][subsec].append(entry)

        with open(output_file_path, 'w') as f:
            f.write(HEADER)

            for sec_num in sorted(section_data):
                title = SECTION_TITLES.get(sec_num, f"Section {sec_num}")
                all_entries = [e for sub in section_data[sec_num].values() for e in sub]
                opt_lines = _options_block(all_entries)
                f.write(_section_header(title, opt_lines if opt_lines else None))

                for subsec, entries in section_data[sec_num].items():
                    if subsec:
                        f.write(f"## {subsec}\n## {'-' * len(subsec)}\n\n")
                    for entry in entries:
                        _write_entry(f, entry)
                    if subsec:
                        f.write('\n')

            if extras:
                f.write(_section_header("Additional Options"))
                for entry in extras:
                    _write_entry(f, entry)

            f.write(
                "\n##=========================================================================\n"
                "##\n##\n"
                "##\t\t\t\tEnd\n"
                "##\n##\n"
                "##=========================================================================\n"
            )

    @staticmethod
    def create_snow_params():
        """
        Creates a dictionary of snow module parameters with default values, mirroring
        the structure of create_input_file(). Called at Model initialization and stored
        as self.snow_options. Parameters can be set the same way as main input keywords:

        >>> model.irreducible_sat['value'] = 0.02

        :return: dict of snow parameters keyed by lowercase parameter name.
        """
        return {
            'irreducible_sat':         {'keyword': 'IRREDUCIBLE_SAT:',         'describe': 'Irreducible water saturation (Volumetric fraction of Pore Space)',                                  'value': 0.01,   'section': 'General Snow Parameters'},
            'k_sat_ref':               {'keyword': 'K_SAT_REF:',               'describe': 'Saturated hydraulic conductivity of snowpack (m/s)',                                                'value': 0.0001, 'section': 'General Snow Parameters'},
            'min_snow_temp':           {'keyword': 'MIN_SNOW_TEMP:',           'describe': 'Minimum temperature of snow (Celsius)',                                                              'value': -27,    'section': 'General Snow Parameters'},
            'fresh_snow_density':      {'keyword': 'FRESH_SNOW_DENSITY:',      'describe': 'Fresh snow density baseline (kg/m^3)',                                                               'value': 60,     'section': 'General Snow Parameters'},
            'canopy_wind_attenuation': {'keyword': 'CANOPY_WIND_ATTENUATION:', 'describe': 'Coefficient of exponential wind attenuation by canopy (Dimensionless)',                             'value': 0.4,    'section': 'General Snow Parameters'},
            'roughness_length':        {'keyword': 'ROUGHNESS_LENGTH:',        'describe': 'Aerodynamic roughness length (z0) of the snow surface (m)',                                         'value': 0.04,   'section': 'General Snow Parameters'},
            'albedo_fresh':            {'keyword': 'ALBEDO_FRESH:',            'describe': 'Fresh snow albedo',                                                                                  'value': 0.85,   'section': 'Albedo Parameters'},
            'albedo_decay_dry':        {'keyword': 'ALBEDO_DECAY_DRY:',        'describe': 'Exponential decay rate of albedo for dry snow',                                                     'value': 0.96,   'section': 'Albedo Parameters'},
            'albedo_decay_wet':        {'keyword': 'ALBEDO_DECAY_WET:',        'describe': 'Exponential decay rate of albedo for wet snow',                                                     'value': 0.82,   'section': 'Albedo Parameters'},
            'albedo_min':              {'keyword': 'ALBEDO_MIN:',              'describe': 'Minimum snow albedo which snow cannot decay below',                                                  'value': 0.2,    'section': 'Albedo Parameters'},
            'albedo_reset_threshold':  {'keyword': 'ALBEDO_RESET_THRESHOLD:',  'describe': 'Minimum snowfall depth required to reset the age of snow surface (mm)',                             'value': 0.5,    'section': 'Albedo Parameters'},
            'optprecpartition':        {'keyword': 'OPTPRECPARTITION:',        'describe': 'Option for precipitation partitioning scheme\n0  Wet-bulb temperature threshold\n1  Linear transition between min/max temperature', 'value': 0, 'section': 'Precipitation Phase Partitioning Parameters'},
            'max_wetbulb_temp':        {'keyword': 'MAX_WETBULB_TEMP:',        'describe': 'Upper wet-bulb temperature at which snowfall can occur for OPTPRECPARTITION 0 (Celsius)',           'value': 5,      'section': 'Precipitation Phase Partitioning Parameters'},
            'min_temp_rain':           {'keyword': 'MIN_TEMP_RAIN:',           'describe': 'Minimum air temperature at which liquid precipitation can occur for OPTPRECPARTITION 1 (Celsius)',  'value': 0,      'section': 'Precipitation Phase Partitioning Parameters'},
            'max_temp_snow':           {'keyword': 'MAX_TEMP_SNOW:',           'describe': 'Maximum air temperature at which snowfall can occur for OPTPRECPARTITION 1 (Celsius)',              'value': 4,      'section': 'Precipitation Phase Partitioning Parameters'},
        }

    def write_snow_params(self, output_file_path='data/model/snow_params.spf'):
        """
        Writes a tRIBS snow parameter file (*.spf) from self.snow_options and automatically
        sets the SNOWFILENAME input option to the written file path. This method is only
        needed when the snow module is enabled (OPTSNOW: 1).

        Parameter values are set the same way as main input file keywords, via self.snow_options:

        Example
        -------
        >>> from pytRIBS.classes import Model
        >>> m = Model()

        >>> # Adjust parameters before writing
        >>> m.irreducible_sat['value'] = 0.02
        >>> m.albedo_fresh['value'] = 0.90
        >>> m.write_snow_params('data/model/snow_params.spf')

        :param output_file_path: Path to write the snow parameter file to. Defaults to
            'data/model/snow_params.spf', matching the standard project directory structure
            created by the Project class. If you are not using the default project structure,
            provide the full path explicitly.
        """
        section_data = {}
        for entry in self.snow_options.values():
            sec = entry['section']
            if sec not in section_data:
                section_data[sec] = []
            section_data[sec].append(entry)

        with open(output_file_path, 'w') as f:
            for section, entries in section_data.items():
                f.write(f"## {section}\n## {'-' * len(section)}\n\n")
                for entry in entries:
                    inline = entry['describe'].split('\n')[0]
                    f.write(f"{entry['keyword']:<26}{inline}\n")
                    f.write(f"{entry['value']}\n\n")

        self.options['snowfilename']['value'] = output_file_path

    def read_snow_params(self, file_path=None):
        """
        Reads a tRIBS snow parameter file (*.spf) and assigns values to self.snow_options.
        The *.spf file shares the same keyword/value format as the main input file, so the
        parsing mirrors read_input_file().

        :param file_path: Path to the *.spf file. Defaults to options['snowfilename']['value'],
            which is set when the main input file is read or when write_snow_params() is called.
        """
        if file_path is None:
            file_path = self.options['snowfilename']['value']

            if file_path is None:
                print(self.options['snowfilename']['keyword'] + " is not specified.")
                return None

        with open(file_path, 'r') as file:
            lines = file.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            line_lower = line.lower()
            for key, entry in self.snow_options.items():
                keyword_lower = entry['keyword'].lower()
                if line_lower.startswith(keyword_lower) and i + 1 < len(lines):
                    self.snow_options[key]['value'] = lines[i + 1].strip()
            i += 1

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

        Land Use Reclassification Table Structure (*.ldt). The first line is a
        descriptive, comma-delimited header that tRIBS skips; each following line is a
        comma-delimited row of parameter values:

        ID,P_[],S_mm,K_mm/hr,b2_1/mm,Al_[],h_m,Kt_[],Rs_s/m,V_[],LAI_[],Theta*s_[],Theta*t_[],RZD_m

        RZD_m (root zone depth, m) must be present for every type; a value of 9999.99 tells
        tRIBS to fall back to its internal default.

        """
        if file_path is None:
            file_path = self.landtablename["value"]

            if file_path is None:
                print(self.landtablename["key_word"] + "is not specified.")
                return

        landuse_list = []

        with open(file_path, 'r') as file:
            lines = file.readlines()

        lines.pop(0)  # discard the descriptive header line (skipped by tRIBS)
        param_standard = 14

        for l in lines:
            if not l.strip():
                continue

            land_info = [v.strip() for v in l.strip().split(',')]

            if len(land_info) == param_standard:
                _id, p, s, k, b_2, al, h, kt, rs, v, lai, tstar_s, tstar_t, rzd = land_info
                station = {
                    "ID": _id,
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
                    "theta*_t": tstar_t,
                    "RZD_m": rzd
                }
                landuse_list.append(station)
            else:
                print(f"Skipping row in {file_path}: expected {param_standard} comma-separated "
                      f"values, got {len(land_info)}.")

        return landuse_list
    @staticmethod
    def write_landuse_table(landuse_list, file_path):
        """
        Writes out a Land Use Reclassification Table (*.ldt) in the tRIBS format: a
        single descriptive header line (skipped by tRIBS) followed by comma-delimited rows
        of parameter values:

        ID,P_[],S_mm,K_mm/hr,b2_1/mm,Al_[],h_m,Kt_[],Rs_s/m,V_[],LAI_[],Theta*s_[],Theta*t_[],RZD_m

        Notes:
        - The Gray (1970) interception parameters (a, b1) present in pre-v6.0.0 land use
          tables have been removed. Tables written by this function are not compatible with
          tRIBS versions prior to v6.0.0.
        - RZD_m (root zone depth, m) must be present for every type. If a dictionary omits it,
          9999.99 is written, which tells tRIBS to use its internal default.

        :param landuse_list: List of dictionaries containing land information specified by .ldt structure above.
        :param file_path: Path to save *.ldt file.
        """
        header = ("ID,P_[],S_mm,K_mm/hr,b2_1/mm,Al_[],h_m,Kt_[],Rs_s/m,V_[],LAI_[],"
                  "Theta*s_[],Theta*t_[],RZD_m")

        with open(file_path, 'w') as file:
            file.write(header + "\n")

            for type in landuse_list:
                row = [type['ID'], type['P'], type['S'], type['K'], type['b2'], type['Al'],
                       type['h'], type['Kt'], type['Rs'], type['V'], type['LAI'],
                       type['theta*_s'], type['theta*_t'], type.get('RZD_m', 9999.99)]
                file.write(",".join(str(v) for v in row) + "\n")

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

    @staticmethod
    def write_geotiff(raster_dict, output_file_path, dtype='float32', compress=None):
        """
        Writes raster data and metadata from a dictionary to a GeoTIFF file.

        This is a more efficient and robust alternative to ASCII for visualization
        and analysis purposes.

        :param raster_dict: Dictionary containing 'data' and 'profile' keys.
                            'data' is the 2D numpy array of raster values.
                            'profile' is the rasterio metadata dictionary.
        :param output_file_path: Path for the output GeoTIFF file (e.g., 'output.tif').
        :param dtype: Data type for the output raster (default is 'float32').
        :param compress: Optional compression method. Common choices are 'lzw',
                        'deflate', or 'packbits'. Using compression is highly
                        recommended to reduce file size.
        """
        # 1. Make a copy of the profile to avoid modifying the original dict
        profile = raster_dict['profile'].copy()
        data = raster_dict['data']

        # 2. Update the profile for the GeoTIFF driver
        profile.update(
            driver='GTiff',  # Set the driver to GeoTIFF
            dtype=dtype      # Set the desired data type
        )
        
        # 3. Add compression to the profile if specified by the user
        # This is a major advantage of GeoTIFFs
        if compress:
            profile['compress'] = compress
            # For LZW or DEFLATE, we can also add a predictor for better compression
            if compress.lower() in ['lzw', 'deflate']:
                profile['predictor'] = 2  # Horizontal differencing

        # 4. Ensure a nodata value is set in the profile
        # If the source data didn't have one, assign a standard one.
        if 'nodata' not in profile:
            profile['nodata'] = -9999.0
        
        # 5. Replace any numpy NaN values in the data array with the nodata value
        # This is crucial as NaN is not a valid value in a saved raster band.
        data = np.where(np.isnan(data), profile['nodata'], data)

        # 6. Ensure the output directory exists before writing
        # This makes the function more robust.
        output_dir = os.path.dirname(output_file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # 7. Write the data to the GeoTIFF file
        # This is much simpler than the ASCII version because rasterio handles
        # all the internal formatting. No manual file editing is needed.
        with rasterio.open(output_file_path, 'w', **profile) as dst:
            dst.write(data.astype(dtype), 1)

        print(f"Successfully wrote GeoTIFF to: {output_file_path}")