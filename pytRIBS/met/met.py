import numpy as np
import pandas as pd
from shapely.geometry import box
import pynldas2 as nldas
import pyproj
import os
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
import xarray as xr
import requests
from io import BytesIO
from pytRIBS.shared.inout import InOut
from pytRIBS.shared.aux import Aux
import io
import requests
from netrc import netrc
from requests.auth import HTTPBasicAuth
import earthaccess


class MetProcessor(Aux, InOut):
    """
    Framework for Met Class. See classes.py
    """
    def __init__(self):
        # This calls Meta.__init__() and initializes self.meta
        super().__init__() 
        
        # Initialize the dictionaries used in run_met_workflow 
        # to prevent errors when run in a standalone script
        if not hasattr(self, 'hydrometstations'):
            self.hydrometstations = {'value': None}
        if not hasattr(self, 'hydrometbasename'):
            self.hydrometbasename = {'value': None}
        if not hasattr(self, 'gaugestations'):
            self.gaugestations = {'value': None}

    def polygon_centroid_to_geographic(self, polygon, utm_crs=None, geographic_crs="EPSG:4326"):
        "Helper function from `Aux` Class"
        lat,lon, gmt = Aux.polygon_centroid_to_geographic(self,polygon,utm_crs=utm_crs,geographic_crs=geographic_crs)
        return lat, lon, gmt
    
    def get_nldas_point(self, centroids, begin, end, epsg=None):
        """
        Fetch NLDAS-2 forcing data from NASA Giovanni for specific coordinates.
        
        This method handles authentication via Earthdata (creating a .netrc file if needed),
        retrieves a session token, and downloads timeseries data directly from the Giovanni API.

        Prerequisites:
        1. An Earthdata Login account.
        2. The 'NASA GESDISC DATA ARCHIVE' application must be authorized in your Earthdata profile.

        Parameters
        ----------
        centroids : list of tuples or list of lists
            Coordinates [(x, y), ...] in the projection specified by 'epsg'.
        begin : str
            Start date 'YYYY-MM-DD'.
        end : str
            End date 'YYYY-MM-DD'.
        epsg : int, optional
            The EPSG code of the input centroids. Defaults to self.meta['EPSG'] if None.

        Returns
        -------
        pandas.DataFrame
            Combined dataframe of all variables.
        """

        # Informational Header
        print("\nInitializing NLDAS Download (NASA Giovanni)")
        print("NOTE: This workflow requires an Earthdata Login.")
        print("CRITICAL: You must authorize the 'NASA GESDISC DATA ARCHIVE' app in your profile.")
        print("If you see 401 errors, check your authorized apps here: https://urs.earthdata.nasa.gov/users/new\n")

        # Authentication
        # We need a username/password to get a Giovanni token.
        # We use earthaccess to ensure the user has a .netrc file set up.
        try:
            # Check if credentials exist locally
            _ = netrc().hosts['urs.earthdata.nasa.gov']
        except (FileNotFoundError, KeyError):
            print("(!) Earthdata credentials not found in .netrc.")
            print("    Initiating interactive login to save credentials...")
            earthaccess.login(strategy="interactive", persist=True)

        # Retrieve credentials from the file (guaranteed to exist now)
        try:
            login_info = netrc().hosts['urs.earthdata.nasa.gov']
            username, password = login_info[0], login_info[2]
        except Exception:
            raise PermissionError("Could not retrieve Earthdata credentials. Please ensure you have an Earthdata account.")

        # Get Giovanni Session Token
        print("Retrieving Giovanni Session Token...")
        signin_url = "https://api.giovanni.earthdata.nasa.gov/signin"
        
        try:
            token_resp = requests.get(signin_url, auth=HTTPBasicAuth(username, password))
            token_resp.raise_for_status()
            token = token_resp.text.replace('"', '').strip()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                # This is the specific error for missing App Authorization
                print("\n\033[91mAuthentication Failed (401 Unauthorized)\033[0m")
                print("Possible causes:")
                print("1. Invalid Username/Password")
                print("2. Missing App Authorization")
                print("   Action: Go to https://urs.earthdata.nasa.gov/users/new")
                print("   Application > Authorized Apps > Approve More Applications > Authorize 'NASA GESDISC DATA ARCHIVE')")
                print("   Once authorized try again.\n")
                raise PermissionError("Earthdata Authorization Failed. See instructions above.") from e
            else:
                raise PermissionError(f"Failed to retrieve Giovanni token: {e}")

        # Configuration
        api_url = "https://api.giovanni.earthdata.nasa.gov/timeseries"
        headers = {
            "authorizationtoken": token,
            "User-Agent": "pytRIBS/1.0"
        }
        
        # Giovanni Variable ID Mapping
        var_map = {
            'NLDAS_FORA0125_H_2_0_Rainf':   'prcp',
            'NLDAS_FORA0125_H_2_0_Tair':   'temp',
            'NLDAS_FORA0125_H_2_0_Qair':   'humidity',
            'NLDAS_FORA0125_H_2_0_PSurf':  'psurf',
            'NLDAS_FORA0125_H_2_0_Wind_E': 'wind_u',
            'NLDAS_FORA0125_H_2_0_Wind_N': 'wind_v',
            'NLDAS_FORA0125_H_2_0_SWdown': 'rsds'
        }

        # Coordinate Transformation setup
        if epsg is None:
            epsg = self.meta.get('EPSG', 4326) 
            
        transformer = None
        if epsg != 4326:
            transformer = pyproj.Transformer.from_crs(epsg, 4326, always_xy=True)

        # Data Retrieval Loop
        results_list = []
        # NLDAS should always be in UTC then we apply the timeshift elsewhere
        time_str = f"{begin}T00:00:00/{end}T23:00:00"

        for pt in centroids:
            x, y = pt
            if transformer:
                lon, lat = transformer.transform(x, y)
            else:
                lon, lat = x, y
            
            location_str = f"[{lat:.4f},{lon:.4f}]"
            print(f"Downloading NLDAS for {location_str}...")

            point_dfs = []
            
            for giovanni_id, tribs_name in var_map.items():
                params = {
                    "data": giovanni_id,
                    "time": time_str,
                    "location": location_str,
                    "version": "2.0"
                }

                try:
                    r = requests.get(api_url, headers=headers, params=params)
                    r.raise_for_status()
                    
                    csv_io = io.StringIO(r.text)
                    header_line = 0
                    for i, line in enumerate(csv_io):
                        if line.startswith("Timestamp") or line.startswith("Time"):
                            header_line = i
                            break
                    csv_io.seek(0)
                    
                    df = pd.read_csv(csv_io, skiprows=header_line)
                    
                    if len(df.columns) >= 2:
                        val_col = df.columns[1] 
                        df = df.rename(columns={df.columns[0]: 'time', val_col: tribs_name})
                        df['time'] = pd.to_datetime(df['time'])
                        df.set_index('time', inplace=True)
                        point_dfs.append(df[[tribs_name]])
                        
                except Exception as e:
                    print(f"  - Error fetching {tribs_name}: {e}")

            if point_dfs:
                station_df = pd.concat(point_dfs, axis=1)
                results_list.append(station_df)
            else:
                print(f"Warning: No valid data found for {location_str}")

        if not results_list:
            raise RuntimeError("NLDAS download failed: No data returned.")

        if len(results_list) == 1:
            return results_list[0]
        else:
            return pd.concat(results_list)

    @staticmethod
    def get_nldas_geom(geom, begin, end, epsg, write_path=None, **hyriver_env_vars):
        """
        Fetch NLDAS-2 data for a given geometry and time period, with optional caching and environment variable configuration.

        This method retrieves NLDAS-2 data for a specified geometry and time range, using the `pynldas2` library.
        It supports environment variables for controlling caching and verbosity, and optionally saves the resulting
        xarray dataset to a NetCDF file.

        Parameters
        ----------
        geom : str
            The geometry (as a Polygon or MultiPolygon) for which the data is being requested.
        begin : str
            The start date for the data request in 'YYYY-MM-DD' format.
        end : str
            The end date for the data request in 'YYYY-MM-DD' format.
        epsg : int
            The EPSG code for the coordinate reference system of the geometry.
        write_path : str, optional
            The file path where the resulting xarray dataset should be saved as a NetCDF file. If not provided, the
            dataset is not saved to a file.
        **hyriver_env_vars : dict, optional
            Additional keyword arguments representing environment variables to control request/response caching and verbosity.
            Supported variables include:
            - HYRIVER_CACHE_NAME: Path to the caching SQLite database for asynchronous HTTP requests.
            - HYRIVER_CACHE_NAME_HTTP: Path to the caching SQLite database for HTTP requests.
            - HYRIVER_CACHE_EXPIRE: Expiration time for cached requests in seconds.
            - HYRIVER_CACHE_DISABLE: Disable reading/writing from/to the cache.
            - HYRIVER_SSL_CERT: Path to an SSL certificate file.

        Returns
        -------
        xarray.Dataset
            The dataset containing the NLDAS-2 data for the specified geometry and time period.

        Raises
        ------
        Exception
            If an error occurs during the data retrieval or saving process.

        Notes
        -----
        - This method uses the `NLDAS-2.get_bygeom` function from the `pynldas2` library to fetch NLDAS-2 data for the specified geometry.
        - The geometry is automatically converted to a `MultiPolygon` if it is provided as a `Polygon`.
        - The HyRiver library should be cited as follows:
          Chegini T, Li H-Y, Leung LR. 2021. HyRiver: Hydroclimate Data Retriever. Journal of Open Source Software 6: 3175.
          DOI: 10.21105/joss.03175.
        - If the `write_path` is specified, the resulting dataset is saved as a NetCDF file at the given location.
        """

        # Assuming gdf is your GeoDataFrame with a 'geometry' column

        # Check if geometry column contains only Polygons
        if geom.geom_type == 'Polygon':
            # Convert Polygon to MultiPolygon
            geom = MultiPolygon([geom])

        # Set environment variables from hyriver_env_vars
        for key, item in hyriver_env_vars.items():
            os.environ[key] = item

        # Fetch data using the NLDAS-2 library
        ds_xarray = nldas.get_bygeom(geom, begin, end, epsg, source='netcdf')

        # Write to NetCDF file if write_path is provided
        if write_path is not None:
            ds_xarray.to_netcdf(write_path)

        return ds_xarray

    @staticmethod
    def get_nldas_elevation(watershed, epsg):
        """
        Download the NLDAS-2 elevation grid as a NetCDF file and return it as an xarray Dataset.

        This method downloads the NLDAS-2 elevation data from a specified URL, clips it to the extent of the provided
        watershed, reprojects it to the specified EPSG code, and returns the processed data as an xarray Dataset.

        Parameters
        ----------
        watershed : geopandas.GeoDataFrame
            The watershed for which the elevation data should be clipped.
        epsg : int
            The EPSG code for the desired projection of the output data.

        Returns
        -------
        xarray.Dataset or None
            The processed elevation data clipped to the watershed extent and reprojected.
            Returns `None` if there is an error during the download or processing.

        Raises
        ------
        requests.exceptions.RequestException
            If there is an error downloading the NLDAS-2 elevation file.
        Exception
            If there is any other error during the processing of the elevation data.

        Notes
        -----
        - The NLDAS-2 elevation data is downloaded from NASA's LDAS repository as a NetCDF file.
        - The downloaded dataset is processed to drop unnecessary variables, and the CRS is assigned using the `epsg` parameter.
        - The EPSG code 32662 (Equidistant Cylindrical projection) is used by default if no EPSG code is specified.
        - Caching the dataset or passing it as a variable rather than downloading it every time is a potential improvement.
        """

        # TODO: Need to make it so that this can be cached or passed in as a variable rather than downloaded.

        url = "https://ldas.gsfc.nasa.gov/sites/default/files/ldas/NLDAS-2/NLDAS_elevation.nc4"

        try:
            # Send a GET request to the URL
            response = requests.get(url)
            response.raise_for_status()  # Raise an error for unsuccessful status codes

            # Open the downloaded content as an xarray DataSet
            with xr.open_dataset(BytesIO(response.content)) as ds:
                dataset = ds.load()  # Load the dataset into memory

            # Drop unnecessary variables
            if 'time_bnds' in dataset.variables:
                dataset = dataset.drop_vars('time_bnds')

            # Write the CRS to the dataset
            dataset = dataset.rio.write_crs(32662)  # EPSG code 32662 for Equidistant Cylindrical projection, default

            return dataset

        except requests.exceptions.RequestException as e:
            print(f"Error downloading NLDAS-2 elevation file: {e}")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    @staticmethod
    def create_nldas_grid_mask(ds, epsg=None):
        """
        Create polygons representing each pixel in a grid based on GeoTransform parameters.

        This method generates a grid of polygons representing each pixel in the input xarray dataset, using the
        dataset's spatial reference information. The resulting polygons are returned as a GeoDataFrame.

        Parameters
        ----------
        ds : xarray.Dataset
            The input dataset containing spatial reference information, including the GeoTransform parameters.
        epsg : int, optional
            The EPSG code for the coordinate reference system. If provided, the resulting GeoDataFrame is set
            with this CRS.

        Returns
        -------
        geopandas.GeoDataFrame
            A GeoDataFrame containing polygons representing each pixel in the grid, with an optional CRS set
            if the `epsg` parameter is provided.

        Notes
        -----
        - The method extracts the GeoTransform parameters from the dataset to compute the coordinates of each pixel.
        - The polygons are created using the `shapely.geometry.box` function to define the bounding box of each pixel.
        - If the `epsg` parameter is provided, the resulting GeoDataFrame will be reprojected to the specified CRS.
        - The order of the top and bottom pixel coordinates is corrected if necessary, based on the GeoTransform.
        """

        # Extract geotransform from the dataset's spatial reference
        geotransform_str = ds.spatial_ref.GeoTransform
        geotransform = tuple(map(float, geotransform_str.split()))

        # Get number of rows and columns from the dataset
        cols, rows = len(ds.x.values), len(ds.y.values)

        # Initialize a list to hold polygon geometries
        polygons = []

        # Calculate polygons for each pixel
        for row in range(rows):
            for col in range(cols):
                # Calculate pixel coordinates
                left = geotransform[0] + col * geotransform[1]
                right = geotransform[0] + (col + 1) * geotransform[1]
                top = geotransform[3] + row * geotransform[5]
                bottom = geotransform[3] + (row + 1) * geotransform[5]

                # Correct the order of top and bottom if necessary
                if geotransform[5] > 0:
                    top, bottom = bottom, top

                # Create polygon geometry
                polygon = box(left, bottom, right, top)
                polygons.append(polygon)

        # Create a GeoDataFrame from polygons
        gdf = gpd.GeoDataFrame(geometry=polygons)

        if epsg is not None:
            gdf.set_crs(epsg, inplace=True)

        return gdf

    @staticmethod
    def clip_nldas_grid_mask_to_watershed(mask, watershed, epsg):
        """
        Clip a target GeoDataFrame (watershed) by each polygon in the pixel GeoDataFrame (mask), and reproject to the appropriate UTM zone.

        This method clips the input `watershed` GeoDataFrame by the pixel polygons in the `mask` GeoDataFrame. It calculates the
        appropriate UTM zone for the `watershed` and reprojects the clipped geometries to UTM or Web Mercator based on location.
        The method then calculates the centroids and geographic coordinates (longitude, latitude) of the clipped geometries.

        Parameters
        ----------
        mask : geopandas.GeoDataFrame
            GeoDataFrame containing the pixel polygons from the NLDAS-2 grid.
        watershed : geopandas.GeoDataFrame
            GeoDataFrame representing the watershed to be clipped by the pixel polygons.
        epsg : int
            The EPSG code for the coordinate reference system used for geographic coordinates (longitude, latitude).

        Returns
        -------
        tuple
            - clipped_watershed : geopandas.GeoDataFrame
                The watershed GeoDataFrame clipped by the pixel polygons, reprojected to UTM coordinates, and containing
                additional columns for centroids (x, y), geographic coordinates (longitude, latitude), and area.
            - utm_crs : str
                The EPSG code of the UTM zone or Web Mercator projection used for the clipped geometries.

        Notes
        -----
        - The method first checks the UTM zone of the watershed and determines if it spans multiple UTM zones or hemispheres.
        - If the watershed spans multiple UTM zones or hemispheres, the geometries are projected to Web Mercator (EPSG:3857).
        - The resulting clipped GeoDataFrame includes the centroid coordinates (x, y) in UTM and the geographic coordinates
          (longitude, latitude) after transforming from UTM to the given EPSG code.
        - The method also calculates the area of each clipped geometry, which can be used for thresholding in subsequent analysis.
        """

        # need to convert to utm
        utm_min = [int((x + 180) // 6) + 1 for x in watershed.bounds.minx.values]
        utm_max = [int((x + 180) // 6) + 1 for x in watershed.bounds.maxx.values]

        # for hemisphere check
        min_y = watershed.bounds.miny.min()
        max_y = watershed.bounds.maxy.max()

        # checks if watershed is in one utm zone and assigns EPSG code accordingly. If it spans utm zones or hemispheres
        # than the web meractor projeciton is used. Note defaults on NAD83 since NLDAS-2 is for north america, similarly the
        # above conditionals are likely not needed, but here for future modificaitons/options.

        if np.unique(utm_min) == np.unique(utm_max):
            utm_zone = np.unique(utm_min)[0]
            if min_y >= 0:
                utm_crs = f"EPSG:269{utm_zone}"
            elif max_y <= 0:
                utm_crs = f"EPSG:269{utm_zone}"
            else:
                utm_crs = f"EPSG:3857"  # web mercator
        else:
            utm_crs = f"EPSG:3857"  # web mercator

        clipped_gdfs = []
        for i, pixel in enumerate(mask.geometry):
            clipped = gpd.clip(watershed, pixel)
            clipped_gdfs.append(clipped)

        clipped_watershed = pd.concat(clipped_gdfs, ignore_index=True)
        clipped_watershed.set_crs(watershed.crs, inplace=True)
        # set lat and long columns for centroid
        clipped_watershed.to_crs(utm_crs, inplace=True)
        clipped_watershed['x'] = clipped_watershed.centroid.x.values
        clipped_watershed['y'] = clipped_watershed.centroid.y.values

        # Define the coordinate transformation
        proj = pyproj.CRS(epsg)  # geographic coordinate system
        utm = pyproj.CRS(utm_crs)  # UTM coordinate system
        transformer = pyproj.Transformer.from_crs(utm, watershed.crs, always_xy=True)
        clipped_watershed['long'], clipped_watershed['lat'] = transformer.transform(clipped_watershed['x'],
                                                                                    clipped_watershed['y'])

        # set area for thresholding in extract_nldas_timeseries
        clipped_watershed['area'] = clipped_watershed.geometry.area

        return clipped_watershed, utm_crs

    @staticmethod
    def extract_nldas_timeseries(gridded_watershed, nldas_met_xarray, nldas_elev_xarray, threshold_area=0):
        """
        Extract NLDAS-2 timeseries data and station coordinates from a gridded watershed.

        This function converts NLDAS-2 xarray datasets (meteorological and elevation) to pandas DataFrames for
        locations within a gridded watershed. The extracted timeseries data is filtered based on the specified
        threshold area, and the station coordinates (longitude, latitude, UTM x, UTM y, elevation) are returned
        along with the timeseries data.

        Parameters
        ----------
        gridded_watershed : geopandas.GeoDataFrame
            A GeoDataFrame representing the watershed with polygons for each sub-watershed.
        nldas_met_xarray : xarray.Dataset
            The xarray dataset containing NLDAS-2 meteorological timeseries data.
        nldas_elev_xarray : xarray.Dataset
            The xarray dataset containing NLDAS-2 elevation data.
        threshold_area : float, optional
            The minimum area for a sub-watershed to be considered. Default is 0, meaning all sub-watersheds
            are included.

        Returns
        -------
        tuple of (list of pandas.DataFrame, list of list of float)
            - The first element is a list of pandas DataFrames, each containing NLDAS-2 timeseries data for a sub-watershed.
            - The second element is a list of station coordinates, where each station is represented as
              [longitude, UTM x, latitude, UTM y, elevation].

        Notes
        -----
        - This function uses the 'nearest' method to select the closest data point in the NLDAS-2 xarray dataset
          for each sub-watershed.
        - The elevation data is extracted from `nldas_elev_xarray` and combined with the timeseries data.
        - The station coordinates include both geographic (longitude, latitude) and UTM coordinates (x, y).
        """

        nldas_time_series = []
        station_coordinates = []  # x,y,z

        for count in range(0, len(gridded_watershed)):

            sub_watershed = gridded_watershed.iloc[count]
            area = sub_watershed.area

            if area > threshold_area:
                # get coords
                long = sub_watershed.long
                lat = sub_watershed.lat
                x = sub_watershed.x
                y = sub_watershed.y

                # extract time series and convert to data frame
                met_station = nldas_met_xarray.sel(x=long, y=lat, method='nearest')
                elev_station = nldas_elev_xarray.sel(lon=long, lat=lat, method='nearest')
                met_df = met_station.to_dataframe()
                elev_df = elev_station.to_dataframe()

                # append results to list
                z = elev_df.NLDAS_elev.iloc[0]
                nldas_time_series.append(met_df)
                station_coordinates.append([long, x, lat, y, z])

        return nldas_time_series, station_coordinates

    def convert_and_write_nldas_timeseries(self, list_dfs, station_coords, gmt,
                                           prefix=None, met_path=None, precip_path=None,
                                           orig_begin=None, orig_end=None):
        """
        Convert NLDAS-2 timeseries data to UTM coordinates and prepare for tRIBS input.

        This function processes NLDAS-2 timeseries data from multiple stations, converts the coordinates to UTM,
        and prepares the data for tRIBS model input. The processed data is saved to meteorological and precipitation
        files in the specified directories.

        Parameters
        ----------
        list_dfs : list of pandas.DataFrame
            A list of DataFrames, each containing NLDAS-2 timeseries data with columns such as 'date', 'psurf',
            'wind_u', 'wind_v', 'temp', 'humidity', 'rsds', and 'prcp'.
        station_coords : list of tuples
            A list of tuples, each containing the (longitude, latitude, elevation) for each station.
        prefix : str
            Prefix for the output filenames.
        met_path : str
            Directory path where meteorological files will be saved.
        precip_path : str
            Directory path where precipitation files will be saved.
        gmt : int
            GMT offset for the data.
        utm_epsg : str
            EPSG code for the UTM coordinate system.

        Returns
        -------
        None
            This function does not return anything. The transformed timeseries data and station details are written
            to the specified output files.

        Notes
        -----
        The function assumes that the input NLDAS-2 data is structured in a specific way and that the stations'
        geographic coordinates (longitude, latitude) need to be converted to UTM coordinates using the provided EPSG code.
        """

        if prefix is None and self.hydrometbasename['value'] is not None:
            prefix = self.hydrometbasename['value']
        else:
            prefix = 'MetResults'

        if met_path is None and self.hydrometstations['value'] is not None:
            met_path = self.hydrometstations['value']
        else:
            prefix = ''

        if precip_path is None and self.gaugestations['value'] is not None:
            precip_path = self.gaugestations['value']
        else:
            prefix = ''

        try:
            met_dir = os.path.dirname(met_path)
        except:
            met_dir = ''

        try:
            precip_dir = os.path.dirname(precip_path)
        except:
            precip_dir = ''

        met_sdf_list = []
        precip_sdf_list = []

        # Hard coded params for writing
        count = 1
        num_params_precip = 5
        num_params_met = 12

        # Physical constants
        L = 2.453 * 10 ** 6  # Latent heat of vaporization (J/kg)
        Rv = 461  # Gas constant for moist air (J/kg-K)

        # Wind Profile Constants (Assuming Standard Met Station i.e. Short Grass)
        # Not something we are letting users control for now. High chance for error. 
        # This adjustment is less of a user parameter and more so unit conversion.
        # NLDAS provides wind at 10m. tRIBS expects wind at 2m over a generic surface (usually grass),
        # which it then adjusts internally based on Land Use.
        # Reference: FAO-56 / Maidment (1993)
        z_meas = 10.0   # Height of NLDAS measurement
        z_dest = 2.0    # Height required by tRIBS
        
        # Roughness length (z0) for short grass is approx 0.10m.
        # Displacement height (d) is negligible for short grass or approx 2/3 * z0. 
        # Using a standard simplified log profile with z0=0.10 and d=0.07:
        z0_ref = 0.015
        d = 0.07

        # Scaling factor = ln(z_dest / z0) / ln(z_meas / z0)
        wind_scale_factor = np.log((z_dest - d) / z0_ref) / np.log((z_meas - d) / z0_ref)

        for df in list_dfs:
            # Apply GMT offset to the index
            df.index = df.index + pd.to_timedelta(gmt, unit='h')
            if orig_begin and orig_end:
                df = df.loc[orig_begin:orig_end].copy()

            # Initialize dictionaries for station details
            met_sdf = {'station_id': None, 'file_path': None, 'lat_dd': None, 'y': None, 'long_dd': None, 'x': None,
                       'GMT': None, 'record_length': None, 'num_parameters': None, 'other': None}
            precip_sdf = {'station_id': None, 'file_path': None, 'y': None, 'x': None, 'record_length': None,
                          'num_parameters': None, 'elevation': None}

            # Update to tRIBS variables
            df['XC'] = 9999.99
            df['TS'] = 9999.99
            df['NR'] = 9999.99
            df['psurf'] *= 0.01  # Convert pressure from Pa to hPa

            # Calculate Wind Speed
            # Magnitude of vector
            df['US'] = (df['wind_u'] ** 2 + df['wind_v'] ** 2) ** 0.5
            # Scale 10m -> 2m
            df['US'] = df['US'] * wind_scale_factor

            df['TA'] = df['temp'] - 273.15  # Temperature in Celsius
            df['e_sat'] = 6.11 * np.exp(
                (L / Rv) * ((1 / 273.15) - (1 / df['temp'])))  # Saturation vapor pressure in hPa

            # Calculate saturation vapor pressure (e_sat) in hPa (mb)
            df['VP'] = (df['humidity'] * df['psurf']) / 0.622
            df['RH'] = 100 * (df['VP'] / df['e_sat'])

            df.rename(columns={'rsds': 'IS', 'prcp': 'R', 'psurf': 'PA'}, inplace=True)
            
            # Define rounding rules for each parameter
            rounding_rules = {
                'PA': 2,    # Pressure (hPa)
                'RH': 2,    # Relative Humidity (%)
                'US': 3,    # Wind Speed (m/s)
                'TA': 2,    # Air Temperature (C)
                'IS': 2,    # Incoming Shortwave Radiation (W/m^2)
                'R': 4,     # Precipitation Rate (kg/m^2/s)
                'VP': 3     # Vapor Pressure (hPa)
            }
            df = df.round(rounding_rules)

            df['date'] = df.index.values

            # Write out files with pytrib utility class InOut
            precip_file = f'precip_{prefix}_{count}.mdf'
            met_file = f'met_{prefix}_{count}.mdf'

            precip_file_path = os.path.join(precip_dir, precip_file)
            met_file_path = os.path.join(met_dir, met_file)

            self.write_precip_station(df[['R', 'date']].copy(), precip_file_path)
            self.write_met_station(df[['PA', 'RH', 'XC', 'TS', 'NR', 'TA', 'US', 'VP', 'IS', 'date']].copy(),
                                    met_file_path)

            # Update sdf dictionaries
            met_sdf['station_id'] = count
            precip_sdf['station_id'] = count
            met_sdf['file_path'] = met_file_path
            precip_sdf['file_path'] = precip_file_path

            # Geographic coordinates
            lat = station_coords[count - 1][2]
            y = station_coords[count - 1][3]
            long = station_coords[count - 1][0]
            x = station_coords[count - 1][1]

            met_sdf['lat_dd'] = lat
            met_sdf['long_dd'] = long

            met_sdf['x'] = x
            met_sdf['y'] = y
            precip_sdf['x'] = x
            precip_sdf['y'] = y

            met_sdf['GMT'] = gmt
            precip_sdf['elevation'] = station_coords[count - 1][4]
            met_sdf['other'] = station_coords[count - 1][4]

            met_sdf['num_parameters'] = num_params_met
            precip_sdf['num_parameters'] = num_params_precip

            length = len(df['date'])

            met_sdf['record_length'] = length
            precip_sdf['record_length'] = length

            met_sdf_list.append(met_sdf)
            precip_sdf_list.append(precip_sdf)

            count += 1

        self.write_met_sdf(met_path, met_sdf_list)
        self.write_precip_sdf(precip_sdf_list, precip_path)

    def run_met_workflow(self, watershed, begin, end, elev):
        """
        Execute the meteorological data workflow for a given watershed.

        This method performs the following steps:
        - Calculates the geographic centroid of the provided watershed.
        - Retrieves meteorological data for the centroid from the NLDAS-2 dataset.
        - Converts and writes the NLDAS-2 time series data to the specified format.

        Parameters
        ----------
        watershed : shapely.geometry.Polygon
            A Shapely polygon representing the watershed area. The geographic centroid of this polygon is used for
            data retrieval.
        begin : str
            The start date for the meteorological data retrieval, in 'YYYY-MM-DD' format.
        end : str
            The end date for the meteorological data retrieval, in 'YYYY-MM-DD' format.
        elev : float
            The elevation of the watershed centroid, used in the data processing.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the retrieved and processed NLDAS-2 meteorological data for the specified centroid.

        Notes
        -----
        - The geographic centroid of the watershed is calculated and used to retrieve the meteorological data from the
          NLDAS-2 dataset.
        - The NLDAS-2 time series data is retrieved for the specified time range, processed, and written to the appropriate
          format using the `convert_and_write_nldas_timeseries` method.
        - A cache is used for the NLDAS-2 data retrieval to improve efficiency.
        """

        lat, lon, gmt = self.polygon_centroid_to_geographic(watershed)
        x, y = watershed.centroid.x, watershed.centroid.y
        centroids = [(x,y)]

        # Adjust dates to aquire an extra day for GMT offset
        download_begin = (pd.to_datetime(begin) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        download_end = (pd.to_datetime(end) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

        nldas_df = self.get_nldas_point(centroids, download_begin, download_end, epsg=self.meta['EPSG'])
        coords = [lon, x, lat, y, elev]
        self.convert_and_write_nldas_timeseries([nldas_df.copy()],[coords], gmt,
                                                orig_begin=begin, orig_end=end)

        return nldas_df