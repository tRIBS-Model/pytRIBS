import glob
import os
import re

import numpy as np
import pandas as pd


class Read():
    """ Framework class for Results class"""

    def get_qout_results(self):
        """
        Reads the outlet discharge and water level data from a specified `.qout` file.

        This method reads a `.qout` file containing outlet discharge and water level data, parses it into a DataFrame, and converts
        the time information from hours since the start date to actual timestamps.

        The `.qout` file is expected to be named by appending `'_Outlet.qout'` to the value of the `"outhydrofilename"` option
        from the `self.options` dictionary.

        The method performs the following steps:
        1. Reads the `.qout` file into a DataFrame with columns `['Time_hr', 'Qstrm_m3s', 'Hlev_m']`.
        2. Converts the `Time_hr` column from hours since the start date into actual timestamps.
        3. Returns the DataFrame with an additional `Time` column representing the timestamps.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing columns `['Time_hr', 'Qstrm_m3s', 'Hlev_m', 'Time']`, where:
            - `Time_hr` is the time in hours since the start date.
            - `Qstrm_m3s` is the discharge in cubic meters per second.
            - `Hlev_m` is the water level in meters.
            - `Time` is the converted timestamp corresponding to each time step.
        """
        # currently only read for outlet, neet to add for hydronodelist
        qout_file = self.options["outhydrofilename"]["value"]+'_Outlet.qout'
        qout_df = pd.read_csv(qout_file, header=None, names=['Time_hr', 'Qstrm_m3s', 'Hlev_m'], skiprows=1, sep='\t')

        starting_date = self.options["startdate"]["value"]
        date = self.convert_to_datetime(starting_date)
        dt = pd.to_timedelta(qout_df['Time_hr'], unit='h')
        qout_df['Time'] = [date + step for step in dt]
        return qout_df
    def get_mrf_results(self, mrf_file=None):
        """
        Reads and processes the `.mrf` file containing model results.

        If `mrf_file` is not provided, constructs the filename using the value of the `"outhydrofilename"` option
        from `self.options`, combined with the runtime value, and appends `"_00.mrf"` to it.

        This method performs the following steps:
        1. Reads the column names and units from the first two rows of the `.mrf` file.
        2. Loads the data into a DataFrame, skipping the first two rows which contain metadata.
        3. Assigns the read column names to the DataFrame and adds the units as metadata.
        4. Converts the `Time` column from hours since the start date to actual timestamps.
        5. Updates the `self.mrf` attribute with the results, excluding extra time steps that may be included in the file.

        Parameters
        ----------
        mrf_file : str, optional
            The path to the `.mrf` file. If not provided, the filename is constructed based on the `"outhydrofilename"` and `"runtime"`
            options from `self.options`.

        Returns
        -------
        None
            This method updates the `self.mrf` attribute with the processed results DataFrame.
        """
        if mrf_file is None:
            runtime = self.options["runtime"]["value"]

            while len(runtime) < 4:
                runtime = '0' + runtime

            mrf_file = self.options["outhydrofilename"]["value"] + runtime + "_00.mrf"

        # Read the first two rows to get column names and units
        with open(mrf_file, 'r') as file:
            column_names = file.readline().strip().split('\t')  # Assuming tab-separated data
            units = file.readline().strip().split('\t')  # Assuming tab-separated data

        # Read the data into a DataFrame, skipping the first two rows
        results_data_frame = pd.read_csv(mrf_file, skiprows=1, sep='\t')

        # Assign column names to the DataFrame
        results_data_frame.columns = column_names

        # Add units as metadata
        results_data_frame.attrs["units"] = units

        # # update time from hourly time step to date
        starting_date = self.options["startdate"]["value"]
        date = self.convert_to_datetime(starting_date)
        dt = pd.to_timedelta(results_data_frame['Time'], unit='h')
        results_data_frame['Time'] = [date + step for step in dt]

        self.mrf['mrf'] = results_data_frame.iloc[0:int(self.options['runtime'][
                                                            'value']) - 1]  # This is becasue currently v5.2 tribs
        # has issues with mrf writing extra time steps

    def get_element_results(self):
        """
        Reads and processes element result files, and assigns them to `self.element_results`.

        This method performs the following steps:
        1. Constructs the directory path from the `"outfilename"` option in `self.options`.
        2. Searches for files in the directory that match a certain pattern (`*.pixel`).
        3. For each matching file, extracts the node ID from the filename.
        4. Reads the content of each element file and creates a DataFrame of results.
        5. Stores the results in a dictionary, with keys representing node IDs and the value being another dictionary
           containing the `pixel` DataFrame and a placeholder for `waterbalance`.

        The resulting dictionary is assigned to `self.element_results`. The key `"invar"` is used for the `.invpixel` file,
        while node-specific files use their respective node IDs as keys.

        Returns
        -------
        None
            This method updates the `self.element_results` attribute with the processed data.
        """
        # List to store first integers
        node_id = []

        # Path to the directory containing the files
        directory_path = self.options["outfilename"]["value"]
        # Find the last occurrence of '/'
        last_slash_index = directory_path.rfind('/')

        # Truncate the string at the last occurrence of '/'
        directory_path = directory_path[:last_slash_index + 1]

        # Search for files matching the pattern
        if os.path.exists(directory_path):
            file_list = glob.glob(directory_path + '*.*')
        else:
            print('Cannot find results directory. Returning nothing.')
            return

        if len(file_list) == 0:
            print("Pixel files not found. Returning nothing.")
            return

        # Iterate through the files
        for file_name in file_list:
            file = file_name.split('/')[-1]
            match = re.search(r'(\d+)\.pixel', file)
            if match:
                print(f"Reading in: {file}")
                first_integer = int(match.group(1))
                node_id.append(first_integer)
                self.element.update(
                    {first_integer: {'pixel': self.read_element_files(file_name), 'waterbalance': None}})

    def read_element_files(self, element_results_file):
        """
        Reads a `.pixel` file from tRIBS model results and converts hourly time steps to datetime.

        This method performs the following steps:
        1. Reads the content of the specified `.pixel` file into a pandas DataFrame.
        2. Converts the `Time_hr` column from hourly timesteps into datetime objects based on the starting date.
        3. Adds a `Time` column to the DataFrame that contains the converted datetime values.

        Parameters
        ----------
        element_results_file : str
            Path to the `.pixel` file containing the tRIBS model results.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the results with an updated `Time` column reflecting datetime values.
        """

        results_data_frame = pd.read_csv(element_results_file, sep=r"\s+", header=0)

        # update time from hourly time step to date
        starting_date = self.options["startdate"]["value"]
        date = self.convert_to_datetime(starting_date)
        dt = pd.to_timedelta(results_data_frame['Time_hr'], unit='h')
        results_data_frame['Time'] = [date + step for step in dt]

        return results_data_frame

    def get_element_wb_dataframe(self, element_id):
        """
        Generates a DataFrame with water balance results for a specified element.

        This method retrieves water balance data for the given `element_id` and calculates various metrics based on
        the pixel data. It uses attributes like porosity and element area from the spatial variables to compute
        values such as saturated water, canopy snow water equivalent, and surface runoff.

        Parameters
        ----------
        element_id : int
            Identifier for the element whose water balance data is to be retrieved.

        Returns
        -------
        pd.DataFrame
            DataFrame containing water balance metrics with the following columns:
            - `Time`: Time series of the data
            - `Unsat_mm`: Unsaturated moisture in mm
            - `Sat_mm`: Saturated moisture in mm, adjusted by porosity
            - `CanopySWE_mm`: Canopy snow water equivalent in mm
            - `SWE_mm`: Snow water equivalent in mm
            - `Canop_mm`: Canopy storage in mm
            - `P_mm_h`: Precipitation in mm/h
            - `ET_mm_h`: Evapotranspiration in mm/h, adjusted for sublimation and evaporation
            - `Qsurf_mm_h`: Surface runoff in mm/h
            - `Qunsat_mm_h`: Unsaturated runoff in mm/h
            - `Qsat_mm_h`: Saturated runoff in mm/h, adjusted by element area
        """

        pixel = self.element[element_id]
        if isinstance(pixel, dict):
            pixel = pixel['pixel']  # waterbalance calc already called

        porosity = self.int_spatial_vars.loc[self.int_spatial_vars.ID == element_id, 'ThetaS'].values[0]
        element_area = self.int_spatial_vars.loc[self.int_spatial_vars.ID == element_id, 'VAr'].values[0]

        df = pd.DataFrame({
            'Time': pixel['Time'],
            'Unsat_mm': pixel['Mu_mm'].values,
            'Sat_mm': pixel['Nwt_mm'].values * porosity,
            'CanopySWE_mm': 10 * pixel['IntSWEq_cm'].values,
            'SWE_mm': 10 * pixel['SnWE_cm'].values,
            'Canop_mm': pixel['CanStorage_mm'],
            'P_mm_h': pixel['Rain_mm_h'],
            'ET_mm_h': pixel['EvpTtrs_mm_h'] - (
                    pixel['SnSub_cm'] * 10 + pixel['SnEvap_cm'] * 10 + pixel['IntSub_cm'] * 10),
            'Qsurf_mm_h': pixel['Srf_Hour_mm'],
            'Qunsat_mm_h': pixel['QpOut_mm_h'] - pixel['QpIn_mm_h'],
            'Qsat_mm_h': pixel['GWflx_m3_h'] / element_area * 1000
        })

        return df

    def get_mrf_wb_dataframe(self):
        """
        Generates a DataFrame with water balance results based on the MRF data.

        This method computes water balance metrics from the MRF data using attributes such as drainage area, porosity,
        and various MRF parameters. The resulting DataFrame includes calculated values for unsaturated moisture,
        saturated moisture, snow water equivalent, and other hydrological metrics.

        Returns
        -------
        pd.DataFrame
            DataFrame containing water balance metrics with the following columns:
            - `Time`: Time series of the data
            - `Unsat_mm`: Unsaturated moisture in mm, calculated using the product of moisture storage, drainage weight, and porosity
            - `Sat_mm`: Saturated moisture in mm, calculated using the product of drainage weight and porosity
            - `CanopySWE_mm`: Canopy snow water equivalent in mm
            - `SWE_mm`: Snow water equivalent in mm
            - `Canop_mm`: Canopy storage in mm (currently set to 0 as it is not averaged)
            - `P_mm_h`: Precipitation in mm/h
            - `ET_mm_h`: Evapotranspiration in mm/h, adjusted for sublimation and evaporation
            - `Qsurf_mm_h`: Surface runoff in mm/h, adjusted for drainage area
            - `Qunsat_mm_h`: Unsaturated runoff in mm/h (currently set to 0; subject to validation)
            - `Qsat_mm_h`: Saturated runoff in mm/h (currently set to 0; subject to validation)
        """
        drainage_area = self.int_spatial_vars['VAr'].sum()  ## in m, TODO investigate why sum 'VAr' != max CAr ?
        weights = self.int_spatial_vars['VAr'].values / drainage_area
        porosity = np.sum(self.int_spatial_vars['ThetaS'].values * weights)
        bedrock_depth = np.sum(self.int_spatial_vars['Bedrock_Depth_mm'].values * weights)

        mrf = self.mrf['mrf']

        df = pd.DataFrame({
            'Time': mrf['Time'],
            'Unsat_mm': mrf['MSMU'].values * mrf['MDGW'].values * porosity,
            'Sat_mm': (bedrock_depth-mrf['MDGW'])* porosity,
            'CanopySWE_mm': 10 * mrf['AvInSn'].values,
            'SWE_mm': 10 * mrf['AvSWE'].values,
            'Canop_mm': 0,  # not average canopy  storage
            'P_mm_h': mrf['MAP'],
            'ET_mm_h': mrf['MET'] - 10 * (mrf['AvSnSub'] + mrf['AvSnEvap'] + mrf['AvInSu']),
            'Qsurf_mm_h': mrf['Srf'] * 3600 * 1000 / drainage_area,
            'Qunsat_mm_h': mrf['Qunsat'],  # assumed zero, but not sure if correct?
            'Qsat_mm_h': 0  # assumed zero, but not sure if correct?
        })

        return df
