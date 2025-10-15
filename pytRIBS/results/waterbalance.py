import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class WaterBalance:
    def get_mrf_water_balance(self, method):
        """
        Calculate the water balance from watershed averaged results (*.mrf).

        This method computes the water balance for the *.mrf file based on the specified method and stores the
        result in the `obj.mrf` dictionary.

        Parameters
        ----------
        method : str or list of str
            A string specifying the calculation method ('full' for the entire time period, 'water_year' for water year calculations),
            or a list of custom date ranges in 'YYYY-MM-DD' format (e.g., ['2024-01-01', '2024-01-31']).

        Returns
        -------
        None

        Notes
        -----
        - The method uses the `_run_mrf_water_balance` internal function to compute the water balance based on the
          provided method.
        - The resulting water balance is stored in the `self.mrf` dictionary with the key `"waterbalance"`.

        Examples
        --------
        1. Calculate full MRF water balance:
           >>> get_mrf_water_balance('full')

        2. Calculate custom MRF water balance for specific dates:
           >>> get_mrf_water_balance(['2024-01-01', '2024-01-31'])
        """

        waterbalance = self._run_mrf_water_balance(method)
        self.mrf['waterbalance'] = waterbalance
    
    
    def get_element_water_balance(self, method):
        """
        Calculate water balance for elements in the model.

        This method iterates through element results and calculates the water balance for a specific node or key.
        The user can specify the method for calculating the time frames over which the water balance is computed.

        Parameters
        ----------
        method : str or list of str
            A string specifying the calculation method ('full' for the entire time period or 'water_year' for water year calculations),
            or a list of custom date ranges in 'YYYY-MM-DD' format (e.g., ['2024-01-01', '2024-01-31']).

        Returns
        -------
        None

        Notes
        -----
        - The method reads node results from the node output list and calculates the water balance for each element.
        - The `method` parameter determines how the time period is defined for water balance calculations:
          - 'full': The full simulation period.
          - 'water_year': Calculations based on the water year.
          - A custom date range can be provided as a list of strings in 'YYYY-MM-DD' format.
        - The water balance results are stored in the `self.element` dictionary with the key `"waterbalance"` for each node.

        Examples
        --------
        1. Calculate full water balance:
           >>> get_element_water_balance('full')

        2. Calculate custom water balance for specific dates:
           >>> get_element_water_balance(['2024-01-01', '2024-01-31'])
        """

        node_file = self.options["nodeoutputlist"]["value"]
        nodes = self.read_node_list(node_file)
    
        for n in nodes:
            n = int(n)
            waterbalance = self._run_element_water_balance(n, method)
            self.element[n]["waterbalance"] = waterbalance
    
    
    # WATER BALANCE FUNCTIONS
    def _run_element_water_balance(self, element_id, method):
        """
        Calculates and returns the water balance for a specified element.

        This method computes the water balance for an element based on the provided method and returns it as a pandas DataFrame.
        The water balance components are calculated over different time periods or methods depending on the input.

        Parameters
        ----------
        element_id : int
            The ID of the element for which to compute the water balance.

        method : str, list of datetime, or tuple
            Specifies the method for computing the water balance:
            - `'full'`: Calculates the water balance for the entire available data range.
            - A list of datetime objects: Specifies a custom range of dates to compute the water balance.
            - A tuple containing start and end dates: Used to determine the time periods for water balance calculation.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the water balance components with columns:
            - `Time`: Time of the data points.
            - `dS`: Change in storage.
            - `nQ`: Net fluxes including runoff.

        Notes
        -----
        - If the `method` is `'full'` or a list, the resulting DataFrame has a single row with the aggregated water balance for the specified range.
        - If `method` is a tuple, the DataFrame has multiple rows, each corresponding to a time period within the specified range.
        - The `dS` column represents the sum of changes in unsaturated, saturated, canopy SWE, SWE, and canopy storage.
        - The `nQ` column represents net fluxes, with surface runoff converted to negative values.
        """
        data = self.element[element_id]
        if isinstance(data, dict):
            data = data['pixel']  # waterbalance calc already called
    
        if method == 'full':
            min_date = min(data.Time)
            max_date = max(data.Time)
            waterbalance = self._element_wb_components(element_id, min_date, max_date)
            timeframe = data.Time.mean()
    
        elif isinstance(method, list):
            min_date = min(method)
            max_date = max(method)
            waterbalance = self._element_wb_components(element_id, min_date, max_date)
            timeframe = data.Time.mean()
    
        else:
            begin, end, timeframe = self._water_balance_dates(data.Time, method)
    
            for n in range(0, len(timeframe)):
                if n == 0:
                    waterbalance = self._element_wb_components(element_id, begin[n], end[n])
                else:
                    temp = self._element_wb_components(element_id, begin[n], end[n])
    
                    for key, val in temp.items():
    
                        if key in waterbalance:
                            waterbalance[key] = np.append(waterbalance[key], val)
    
        waterbalance.update({"Time": timeframe})
    
        if method == 'full' or isinstance(method, list):
            waterbalance = pd.DataFrame(waterbalance, index=[0])
        else:
            waterbalance = pd.DataFrame.from_dict(waterbalance)
    
        waterbalance.set_index('Time', inplace=True)
    
        waterbalance['dS'] = waterbalance['dUnsat'].values + waterbalance['dSat'].values + waterbalance['dCanopySWE'].values \
                             + waterbalance['dSWE'].values + waterbalance['dCanopy'].values
    
        # net fluxes from surface and saturated and unsaturated zone,
        waterbalance['nQ'] = waterbalance['nQsurf'].values + waterbalance['nQunsat'].values + waterbalance['nQsat'].values
    
        return waterbalance
    
    

    def _element_wb_components(self, element_id, begin, end):
        """
        Computes water balance calculations for an individual computational element over a specified time frame.

        This method calculates the change in storage and net cumulative fluxes for various water balance components
        based on the data from the specified element's .pixel file within the given time period.

        Parameters
        ----------
        element_id : int
            The ID of the element for which to compute the water balance components.

        begin : datetime
            The start date and time for the calculation period.

        end : datetime
            The end date and time for the calculation period.

        Returns
        -------
        dict
            A dictionary containing water balance components with the following keys:
            - `dUnsat` : Change in unsaturated storage (mm).
            - `dSat` : Change in saturated storage (mm).
            - `dCanopySWE` : Change in canopy snow water equivalent (mm).
            - `dSWE` : Change in snow water equivalent (mm).
            - `dCanopy` : Change in canopy storage (mm).
            - `nP` : Net precipitation over the period (mm).
            - `nET` : Net evapotranspiration over the period (mm).
            - `nQsurf` : Net surface runoff over the period (mm).
            - `nQunsat` : Net unsaturated zone runoff over the period (mm).
            - `nQsat` : Net saturated zone runoff over the period (mm).

        Notes
        -----
        - The values for `d*` keys represent the change in the corresponding water component from the start to the end
          of the specified period.
        - The values for `n*` keys represent the total cumulative flux of the corresponding water component over the
          entire period.
        - Assumes that `df['Time']` is already in a suitable format for comparison with `begin` and `end`.

        """
    
        # data frame with continuous water balance components
        df = self.get_element_wb_dataframe(element_id)
        duration_id = (df['Time'].values >= begin) & (df['Time'].values <= end)
    
        df = df[duration_id]
    
        # return dictionary with values
        waterbalance = {}
    
        # calculate individual water balance components
        waterbalance.update({'dUnsat': df['Unsat_mm'].iloc[-1] - df['Unsat_mm'].iloc[0]})
        waterbalance.update({'dSat': df['Sat_mm'].iloc[0] - df['Sat_mm'].iloc[-1]})
        waterbalance.update({'dCanopySWE': df['CanopySWE_mm'].iloc[-1] - df['CanopySWE_mm'].iloc[0]})
        waterbalance.update({'dSWE': df['SWE_mm'].iloc[-1] - df['SWE_mm'].iloc[0]})
        waterbalance.update({'dCanopy': df['Canop_mm'].iloc[-1] - df['Canop_mm'].iloc[0]})
        waterbalance.update({'nP': df['P_mm_h'].sum()})
        waterbalance.update({'nET': df['ET_mm_h'].sum()})
        waterbalance.update({'nQsurf': df['Qsurf_mm_h'].sum()})
        waterbalance.update({'nQunsat': df['Qunsat_mm_h'].sum()})
        waterbalance.update({'nQsat': df['Qsat_mm_h'].sum()})
    
        return waterbalance
    
    
    def _run_mrf_water_balance(self, method):
        """
        Computes water balance calculations for the MRF model over a specified time frame.

        This method calculates the water balance components for the MRF model based on the provided time frame
        and method. It supports different methods for specifying the time period and computes the net and change
        in storage values for various water components.

        Parameters
        ----------
        method : str, list, or other
            Defines the method for calculating the water balance:
            - `'full'`: Uses the entire time period from the data.
            - `list`: A list of datetime objects defining the time period.
            - `other`: A method for specifying the time periods, which is processed to determine the begin and end dates.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the water balance components with the following columns:
            - `dS` : Change in storage (sum of changes in unsaturated, saturated, canopy SWE, SWE, and canopy storage).
            - `nQ` : Net fluxes (negative surface runoff, plus unsaturated and saturated zone runoff).
            - Other columns include net and change in storage components like `dUnsat`, `dSat`, `dCanopySWE`,
              `dSWE`, `dCanopy`, `nP`, `nET`, `nQsurf`, `nQunsat`, and `nQsat`.

        Notes
        -----
        - Assumes `self.mrf['mrf']` contains a DataFrame with a `'Time'` column and other necessary water balance data.
        - The `waterbalance` dictionary is updated with net and change in storage components, as well as net fluxes.
        - The `method` parameter determines how the time frame is defined and processed. If `method` is `'full'`
          or a list, the water balance is calculated for the entire period or specified periods, respectively.
        - For other methods, the `_water_balance_dates` function is used to determine the start and end dates.

        """
        global waterbalance
        data = self.mrf['mrf']
    
        if method == 'full':
            min_date = min(data.Time)
            max_date = max(data.Time)
            waterbalance = self._mrf_wb_components(min_date, max_date)
            timeframe = data.Time.mean()
    
        elif isinstance(method, list):
            min_date = min(method)
            max_date = max(method)
            waterbalance = self._mrf_wb_components(min_date, max_date)
            timeframe = data.Time.mean()
    
        else:
            begin, end, timeframe = self._water_balance_dates(data.Time, method)
    
            for n in range(0, len(timeframe)):
                if n == 0:
                    waterbalance = self._mrf_wb_components(begin[n], end[n])
                else:
                    temp = self._mrf_wb_components(begin[n], end[n])
    
                    for key, val in temp.items():
    
                        if key in waterbalance:
                            waterbalance[key] = np.append(waterbalance[key], val)
    
        waterbalance.update({"Time": timeframe})
        # change in storage
        waterbalance.update({"dS": waterbalance['dUnsat'] + waterbalance['dSat'] + waterbalance['dCanopySWE'] +
                                   waterbalance['dSWE'] + waterbalance['dCanopy']})
        # net fluxes from surface and saturated and unsaturated zone
        waterbalance.update({'nQ': waterbalance['nQsurf'] + waterbalance['nQunsat'] + waterbalance['nQsat']})

        if method == 'full' or isinstance(method, list):
            waterbalance = pd.DataFrame(waterbalance, index=[0])
        else:
            waterbalance = pd.DataFrame.from_dict(waterbalance)
    
        waterbalance.set_index('Time', inplace=True)
    
        return waterbalance
    
    

    def _mrf_wb_components(self, begin, end):
        """
        Computes water balance calculations for the MRF model over a specified time frame.

        This method calculates various water balance components for the MRF model based on the given time period.
        The calculations are performed on data retrieved from the MRF model and are summarized in a dictionary
        with net and change in storage values for various water components.

        Parameters
        ----------
        begin : datetime
            The start date of the time period for which the water balance is calculated.

        end : datetime
            The end date of the time period for which the water balance is calculated.

        Returns
        -------
        dict
            A dictionary containing the water balance components with the following keys:
            - `dSat` : Change in saturated storage (difference between the initial and final values).
            - `dUnsat` : Change in unsaturated storage (difference between the final and initial values).
            - `dCanopySWE` : Change in canopy snow water equivalent (difference between the final and initial values).
            - `dSWE` : Change in snow water equivalent (difference between the final and initial values).
            - `dCanopy` : Change in canopy storage (currently set to 0; future updates may be required).
            - `nP` : Net precipitation (sum of precipitation values over the time period).
            - `nET` : Net evapotranspiration (sum of evapotranspiration values over the time period).
            - `nQsurf` : Net surface runoff (sum of surface runoff values over the time period).
            - `nQunsat` : Net unsaturated zone runoff (assumed to be 0 due to model assumptions).
            - `nQsat` : Net saturated zone runoff (assumed to be 0 due to model assumptions).

        Notes
        -----
        - Assumes `self.get_mrf_wb_dataframe()` returns a DataFrame with water balance components and a `'Time'` column.
        - Assumptions in the model include closed boundaries at divide and outlet, leading to zero values for `nQunsat` and `nQsat`.
        - Canopy storage is currently not updated in the MRF model; this is marked as TODO for future updates.

        """
    
        # logical index for calculating water balance
        df = self.get_mrf_wb_dataframe()
        duration_id = (df['Time'].values >= begin) & (df['Time'].values <= end)
        df = df[duration_id]
    
        # return dictionary with values
        waterbalance = {}
    
        waterbalance.update({'dSat': df['Sat_mm'].iloc[-1]-df['Sat_mm'].iloc[0]})
        waterbalance.update({'dUnsat': df['Unsat_mm'].iloc[-1] - df['Unsat_mm'].iloc[0]})
        waterbalance.update({'dCanopySWE': df['CanopySWE_mm'].iloc[-1] - df['CanopySWE_mm'].iloc[0]})  # convert from cm to mm
        waterbalance.update({'dSWE': df['SWE_mm'].iloc[-1] - df['SWE_mm'].iloc[0]})
        waterbalance.update({'dCanopy': 0})  # TODO update mrf w/ mean intercepted canopy storage
        waterbalance.update({'nP': df['P_mm_h'].sum()})
        waterbalance.update({'nET': df['ET_mm_h'].sum()})
        waterbalance.update({'nQsurf': df['Qsurf_mm_h'].sum()})
        waterbalance.update({'nQunsat': df['Qunsat_mm_h'].sum()})  # Assumption in model is closed boundaries at divide and outlet
        waterbalance.update({'nQsat': 0})
    
        return waterbalance
    
    
    @staticmethod
    def _water_balance_dates(time_vector, method):
        """
        Returns three vectors of time objects for specified time intrevals: beginning dates, ending dates,
        and years.
        :param time_vector: Vector containing sequential dates from model simulations.
        :param str method: String specifying time interval to segment time vector. "water_year", segments time frame by water year and
        discards results that do not start first or end on last water year in data. "year", just segments based on
        year, "month" segments base on month, and "cold_warm",segments on cold (Oct-April) and warm season (May-Sep).
        """
    
        min_date = min(time_vector)
        max_date = max(time_vector)
        begin_dates = None
        end_dates = None
    
        if method == "water_year":
            years = np.arange(min_date.year, max_date.year)
            begin_dates = [pd.Timestamp(year=x, month=10, day=1, hour=0, minute=0, second=0) for x in years]
            years += 1
            end_dates = [pd.Timestamp(year=x, month=9, day=30, hour=23, minute=0, second=0) for x in years]
    
            # make sure water years are in data set
            while begin_dates[0] < min_date:
                begin_dates.pop(0)
                end_dates.pop(0)
    
            while end_dates[len(end_dates) - 1] > max_date:
                begin_dates.pop(len(end_dates) - 1)
                end_dates.pop(len(end_dates) - 1)
    
        if method == "year":
            years = np.arange(min_date.year, max_date.year)
            begin_dates = [pd.Timestamp(year=x, month=1, day=1, hour=0, minute=0, second=0) for x in years]
            end_dates = [pd.Timestamp(year=x, month=12, day=31, hour=23, minute=0, second=0) for x in years]
    
            # adjust start date according to min_date
            begin_dates[0] = min_date
    
            # add ending date according to end_date
            end_dates.append(max_date)
    
            # add last year to years
            years = np.append(years, max_date.year)
    
        if method == "cold_warm":
            years = np.arange(min_date.year, max_date.year + 1)
            begin_dates = [[pd.Timestamp(year=x, month=5, day=1, hour=0, minute=0, second=0),
                            pd.Timestamp(year=x, month=10, day=1, hour=0, minute=0, second=0)] for x in years]
            begin_dates = [date for sublist in begin_dates for date in sublist]
            end_dates = [[pd.Timestamp(year=x, month=9, day=30, hour=23, minute=0, second=0),
                          pd.Timestamp(year=x + 1, month=4, day=30, hour=23, minute=0, second=0)] for x in years]
            end_dates = [date for sublist in end_dates for date in sublist]
    
            # make sure season are in data set
            while begin_dates[0] < min_date:
                begin_dates.pop(0)
                end_dates.pop(0)
    
            while end_dates[len(end_dates) - 1] > max_date:
                begin_dates.pop(len(end_dates) - 1)
                end_dates.pop(len(end_dates) - 1)
    
        # Update date time to reflect middle of period over which the waterbalance is calculated
        years = [x + (y - x) / 2 for x, y in zip(begin_dates, end_dates)]
        return begin_dates, end_dates, years
    
    
