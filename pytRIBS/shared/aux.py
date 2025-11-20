import subprocess
import os
import shutil
from datetime import datetime

import pandas as pd
import pytz
import rasterio
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from rasterio import fill
import  matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pyproj import Transformer
from rasterio.enums import Resampling
from timezonefinder import TimezoneFinder


class Aux:

    @staticmethod
    def rename_file_with_date(file_path, date_str):
        """
        Renames a file by appending the provided date and '00' for hours before the file extension.

        Args:
            file_path (str): The full path of the file to be renamed.
            date_str (str): The date string in the format 'YYYY-MM-DD'.

        Returns:
            str: The new file name after renaming.
        """
        # Extract directory, file name, and extension
        directory, file_name = os.path.split(file_path)
        name, ext = os.path.splitext(file_name)

        # Convert the date string to the required format
        try:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            formatted_date = date_obj.strftime('%m%d%Y') + '00'
        except ValueError:
            raise ValueError("Date string must be in 'YYYY-MM-DD' format.")

        # Create the new file name
        new_file_name = f"{name}{formatted_date}{ext}"
        new_file_path = os.path.join(directory, new_file_name)

        # Rename the file
        os.rename(file_path, new_file_path)

        return new_file_name

    def polygon_centroid_to_geographic(self, polygon, utm_crs=None, geographic_crs="EPSG:4326"):
        """
        Converts the centroid of a polygon from UTM coordinates to geographic coordinates (latitude and longitude),
        and calculates the GMT offset of the local time zone at the centroid location.

        Parameters
        ----------
        polygon : shapely.geometry.Polygon
            A Shapely Polygon object for which the centroid's geographic coordinates are to be calculated.

        utm_crs : str, optional
            The EPSG code or CRS string of the UTM coordinate system. If not provided, it defaults to the CRS
            specified in the `self.meta['EPSG']` attribute.

        geographic_crs : str, optional
            The CRS string for the geographic coordinate system. Defaults to `"EPSG:4326"` for WGS84.

        Returns
        -------
        tuple
            A tuple containing:
            - `lat` : float
                Latitude of the centroid in decimal degrees.
            - `lon` : float
                Longitude of the centroid in decimal degrees.
            - `gmt_offset` : int
                GMT offset in hours based on the local time zone at the centroid location.

        Raises
        ------
        ValueError
            If no UTM CRS is found and `self.meta['EPSG']` is `None`, a ValueError is raised.

        Notes
        -----
        - The function uses the `Transformer` class from the `pyproj` library to convert UTM coordinates to geographic coordinates.
        - The `TimezoneFinder` library is used to determine the local time zone based on latitude and longitude.
        - The GMT offset is calculated using the local time zone's UTC offset.

        Examples
        --------
        >>> from shapely.geometry import Polygon
        >>> polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        >>> lat, lon, gmt_offset = self._polygon_centroid_to_geographic(polygon, utm_crs="EPSG:32633")
        >>> print(lat, lon, gmt_offset)
        (52.5167, 13.3833, 1)
        """
        if utm_crs is None:
            utm_crs = self.meta['EPSG']

            if utm_crs is None:
                print(
                    'Could not find a crs for this watershed.\nUpdate the crs in the associated meta attribute or '
                    'provide it as an argument.')
                return

        centroid = polygon.centroid

        transformer = Transformer.from_crs(utm_crs, geographic_crs, always_xy=True)

        lon, lat = transformer.transform(centroid.x, centroid.y)

        tf = TimezoneFinder()
        timezone_str = tf.timezone_at(lng=lon, lat=lat)

        timezone = pytz.timezone(timezone_str)
        local_time = datetime.now(timezone)

        gmt_offset = int(local_time.utcoffset().total_seconds() / 3600)

        return lat, lon, gmt_offset

    def utm_to_latlong(self, easting, northing, epsg=None):
        """
        Convert UTM coordinates to latitude and longitude using an EPSG code with pyproj.

        Parameters:
        easting (float): UTM easting coordinate.
        northing (float): UTM northing coordinate.
        epsg_code (int): EPSG code representing the UTM projection.

        Returns:
        tuple: A tuple containing latitude and longitude.
        """
        if epsg is None:
            epsg = self.meta['EPSG']

        transformer = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(easting, northing)

        return lat, lon

    @staticmethod
    def discrete_cmap(N, base_cmap='viridis'):
        """Create an N-bin discrete colormap from the specified input map."""
        base = plt.get_cmap(base_cmap)
        color_list = base(np.linspace(0, 1, N))
        cmap = ListedColormap(color_list, name=base.name + str(N))
        return cmap

    @staticmethod
    def fillnodata(files, overwrite=False, resample_pixel_size=None, resample_method='nearest', **kwargs):
        """
        Fills nodata gaps in raster files based on a maximum search distance and optionally resamples the raster.

        Parameters:
        files (list): List of paths to raster files.
        overwrite (bool): If True, the original files will be overwritten with filled data. If False, new files with "_filled" suffix will be created.
        resample_pixel_size (float, optional): Target pixel size for resampling. If None, no resampling is performed.
        resample_method (str, optional): Method for resampling. Choices are 'nearest', 'bilinear', 'cubic', etc. Defaults to 'nearest'.
        **kwargs: Additional keyword arguments to be passed to rasterio.fill.fillnodata.

        Note:
        This function essentially wraps rasterio.fill.fillnodata and includes optional resampling.
        """
        resampling_methods = {
            'nearest': Resampling.nearest,
            'bilinear': Resampling.bilinear,
            'cubic': Resampling.cubic,
            'cubic_spline': Resampling.cubic_spline,
            'lanczos': Resampling.lanczos,
            'average': Resampling.average,
            'mode': Resampling.mode
        }

        if resample_method not in resampling_methods:
            raise ValueError(
                f"Invalid resample_method: {resample_method}. Choose from {list(resampling_methods.keys())}")

        for file_path in files:
            with rasterio.open(file_path) as src:
                data = src.read(1)
                msk = src.read_masks(1)
                filled_data = fill.fillnodata(data, mask=msk, **kwargs)

                if resample_pixel_size:
                    # Compute new dimensions and transform for resampling
                    old_pixel_size_x, old_pixel_size_y = abs(src.transform[0]), abs(src.transform[4])
                    new_width = int(src.width * (old_pixel_size_x / resample_pixel_size))
                    new_height = int(src.height * (old_pixel_size_y / resample_pixel_size))

                    # Update profile for the new dimensions and pixel size
                    new_transform = src.transform * src.transform.scale(
                        (src.width / new_width),
                        (src.height / new_height)
                    )

                    resampled_data = np.empty((new_height, new_width), dtype=filled_data.dtype)
                    resampling = resampling_methods[resample_method]
                    resampled_data = rasterio.warp.reproject(
                        source=filled_data,
                        destination=resampled_data,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=new_transform,
                        dst_crs=src.crs,
                        resampling=resampling
                    )[0]

                    # Update profile for the new resolution
                    profile = src.profile.copy()
                    profile.update(
                        height=new_height,
                        width=new_width,
                        transform=new_transform
                    )

                    data_to_write = resampled_data
                else:
                    profile = src.profile.copy()
                    data_to_write = filled_data

                # Write the data to the output file
                if overwrite:
                    with rasterio.open(file_path, 'w', **profile) as dst:
                        dst.write(data_to_write, 1)
                else:
                    base_name, ext = os.path.splitext(file_path)
                    filled_file_path = f"{base_name}_filled{ext}"
                    with rasterio.open(filled_file_path, 'w', **profile) as dst:
                        dst.write(data_to_write, 1)

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
    
    def print_tags(self, tag_name):
        """
        Prints .in options for a specified tag.
        :param tag_name: Currently: "io", input/output, "physical", physical model params, "time", time parameters,
        "opts", parameters for model options, "restart", restart capabilities, "parallel", parallel options.

        Example:
            >>> m.print_tags("io")
        """

        data = self.options  # Assuming m.options is a dictionary with sub-dictionaries

        # Filter sub-dictionaries where "io" is in the "tags" list
        result = [item for item in data.values() if any(tag_name in tag for tag in item.get("tags", []))]

        # Display the filtered sub-dictionaries
        for dictionary in result:
            for item in dictionary:
                if item != "tags":
                    print(item + ": " + str(dictionary[item]))
                elif item == "tags":
                    print("\n")

    # MODEL FUNCTIONS
    @staticmethod
    def run(executable, input_file, mpi_command=None, tribs_flags=None, log_path=None,
            store_input=None, timeit=True, verbose=True):
        """
        Run a tRIBS model simulation with optional arguments.

        Run_simulation assumes that if relative paths are used then the binary and input file are collocated in the
        same directory. That means for any keywords that depend on a relative path, must be specified from the directory
        the tRIBS binary is executed. You can pass the location of the input file and executable as paths, in which case
        the function copies the binary and input file to same directory and then deletes both after the model run is complete.
        Optional arguments can be passed to store

        Args:
            binary_path (str): The path to the binary model executable.
            control_file_path (str): The path to the input control file for the binary.
            optional_args (str): Optional arguments to pass to the binary.

        Returns:
            int: The return code of the binary model simulation.
        """
        if mpi_command is not None:
            command = mpi_command.split()
            command.extend([executable, input_file])
        else:
            command = [executable, input_file]

        if tribs_flags is not None:
            command.extend(tribs_flags.split())

        if log_path is not None:
            command.append(log_path)

        if timeit:
            command.insert(0, "time")

        print(command)

        if verbose:
            subprocess.run(command)
        else:
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    @staticmethod
    def build(source_file, build_directory, verbose=True, exe="tRIBS", parallel="ON", cxx_flags="-O2"):
        """
        Run a tRIBS model simulation with optional arguments.

        Run_simulation assumes that if relative paths are used then the binary and input file are collocated in the
        same directory. That means for any keywords that depend on a relative path, must be specified from the directory
        the tRIBS binary is executed. You can pass the location of the input file and executable as paths, in which case
        the function copies the binary and input file to same directory and then deletes both after the model run is complete.
        Optional arguments can be passed to store

        Args:
            binary_path (str): The path to the binary model executable.
            control_file_path (str): The path to the input control file for the binary.
            optional_args (str): Optional arguments to pass to the binary.

        Returns:
            int: The return code of the binary model simulation.
        """
        # TODO: add check if build directory already exists, prompt user if they want to remove
        if os.path.exists(build_directory):
            print(f"The directory '{build_directory}' exists.")
            user_input = input("Would you like to remove it? [y/n]: ")
            if user_input:
                try:
                    # Remove the directory and its contents
                    shutil.rmtree(build_directory)
                    print(f"Directory '{build_directory}' and its contents have been removed.")
                except FileNotFoundError:
                    print(f"Directory '{build_directory}' does not exist.")
                except PermissionError:
                    print(f"Permission denied while attempting to remove '{build_directory}'.")
                except Exception as e:
                    print(f"An error occurred: {str(e)}")

        # Allow modification of CMakeList.txt
        modified_lines = []
        source_file = os.path.expanduser(source_file)
        file_path = os.path.join(source_file, "CMakeLists.txt")

        # Define the variables to search for and their corresponding replacements
        variables_to_replace = {
            "exe": exe,
            "parallel": parallel,
            "cxx_flags": cxx_flags
        }

        # Read the contents of the CMakeLists.txt file and modify lines as needed
        with open(file_path, 'r') as file:
            for line in file:
                for variable, value in variables_to_replace.items():
                    if line.strip().startswith(f'set({variable} "'):
                        line = f'set({variable} "{value}")\n'
                modified_lines.append(line)

        # Write the modified lines back to the file
        with open(file_path, 'w') as file:
            file.writelines(modified_lines)

        if verbose:
            cmake_configure_command = ["cmake", "-B", build_directory, "-S", source_file]
            subprocess.run(cmake_configure_command)

            cmake_build_command = ["cmake", "--build", build_directory, "--target", "all"]
            result = subprocess.run(cmake_build_command)
        else:
            cmake_configure_command = ["cmake", "-B", build_directory, "-S", source_file]
            subprocess.run(cmake_configure_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            cmake_build_command = ["cmake", "--build", build_directory, "--target", "all"]
            result = subprocess.run(cmake_build_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        return result.returncode

    def clean(self):
        pass
