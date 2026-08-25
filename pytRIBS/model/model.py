import os
import datetime
import sys
from pytRIBS.model.run_docker import tRIBSDocker

class ModelProcessor:
    "Base class for Model Class"
    @staticmethod
    def run_tribs_docker(volume_path, input_file, execution_mode='serial', num_processes=None):
        """Main function to run the TRIBSDocker class."""
        docker_instance = tRIBSDocker(volume_path, input_file, execution_mode, num_processes)
        docker_instance.start_docker_desktop()
        docker_instance.initialize_docker_client()
        docker_instance.pull_image()
        docker_instance.run_container()
        docker_instance.execute()
        docker_instance.cleanup_container()
    def check_paths(instance):
        """
        Check the existence of specified input/output paths and verify the presence of required files for the tRIBS model.

        This method performs the following tasks:
        - Checks if the paths specified in the `instance.options` dictionary exist and outputs warnings for non-existent paths.
        - Verifies the existence of station descriptor paths for precipitation and meteorological stations.
        - Checks the existence of grid files for land use, soil types, and hydro-meteorological data if specified in the model options.
        - Ensures that individual grid files for hydro-meteorological data are continuous across the simulation time period, reporting any missing files.

        :param instance: An instance of the class that contains the options, grid data files, and other required attributes.
        :type instance: object


        :return: A list of missing files for hydro-meteorological grid data if any files are missing; otherwise, returns None.
        :rtype: list of str or None
        """
        data = instance.options  # Assuming m.options is a dictionary with sub-dictionaries
        exists = []
        doesnt = []

        # Filter sub-dictionaries where "io" is in the "tags" list
        result = [item for item in data.values() if any('io' in tag for tag in item.get("tags", []))]

        # Display the filtered sub-dictionaries
        for item in result:

            # special look at outfilename because:
            # (1) includes base name and not actual path
            # (2) tRIBS will error out if it doesn't exist, but won't tell you explicitly why.

            if item["keyword"] == "OUTFILENAME:":
                print("Checking OUTFILENAME:")
                path = item["value"]
                index = path.rfind('/')  # Find the last occurrence of '/'

                if index != -1:
                    path = path[:index + 1]  # Include the '/' in the result
                else:
                    path = path  # Handle the case where there is no '/'

                flag = os.path.exists(path)

                if flag:
                    print("Path for OUTFILENAME: exists")
                else:
                    print("Warning!!! Path for OUTFILENAME: does not exist")

            if isinstance(item["value"], str):
                flag = os.path.exists(item["value"])
                if flag:
                    exists.append(item)
                else:
                    doesnt.append(item)

        print("\nThe following tRIBS inputs do not have paths that exist: \n")
        for item in doesnt:
            print(f"{item['keyword']} {item['describe']}")

        print("\nChecking if station descriptor paths exist.\n")
        rain = instance.read_precip_sdf()

        flags = []
        if rain is not None:
            for station in rain:
                flag = os.path.exists(station["file_path"])
                flags.append(flag)

                if not flag:
                    print(f"{station['file_path']} does not exist")

            if all(flags):
                print("All rain gauge paths exist.")
        else:
            print("No rain gauges are specified.")

        met = instance.read_met_sdf()

        flags = []
        if met is not None:
            for station in met:
                flag = os.path.exists(station["file_path"])
                flags.append(flag)

                if not flag:
                    print(station["file_path"] + " does not exist")

            if all(flags):
                print("All met station paths exist.")

        else:
            print("No met stations are specified.")

        print("\nChecking if grid files exist.\n")

        if int(instance.options["optlanduse"]["value"]) in (1, 2):
            print("Model is set to read landuse grid files: checking paths and .gdf file")
            instance.read_grid_data_file("land")

        if int(instance.options["optsoiltype"]["value"]) == 1:
            print("Model is set to read soil grid files: checking paths and .gdf file")
            instance.read_grid_data_file("soil")

        if int(instance.options["metdataoption"]["value"]) == 2:
            print("Model is set to hydro-met grid files: checking paths and .gdf file\n")
            wgdf = instance.read_grid_data_file("weather")

            date_format = '%m/%d/%Y/%H/%M'

            missing_files = []

            print("\nChecking that individual grid files are continuous (1 hr time steps) across the model simulation "
                  "time period\n")

            for params in wgdf["Parameters"]:
                directory = params['Raster Path']
                ext = params['Raster Extension']
                var = params['Variable Name']
                files = os.listdir(directory)

                expected_time = datetime.datetime.strptime(instance.startdate['value'], date_format)
                end_time = expected_time + datetime.timedelta(hours=int(instance.runtime['value']))

                if directory is None:
                    print(f"No files were checked for {var}.")
                    continue

                print(f"Checking {directory} :")

                previous_month = None

                while expected_time <= end_time and directory is not None:
                    expected_filename = f"{var}{expected_time.strftime('%m%d%Y%H')}.{ext}"

                    current_month = expected_time.month

                    if current_month != previous_month:
                        print(expected_time, end="\r")
                        previous_month = current_month
                        sys.stdout.flush()
                        datetime.time.sleep(0.001)

                    if expected_filename not in files:
                        print(f"Missing file: {expected_filename}")
                        missing_files.append(expected_filename)

                    dt = datetime.timedelta(hours=1)
                    expected_time += dt

                if len(missing_files) == 0:
                    print(f"No missing files for {var}")
                elif len(missing_files) > 0:
                    print(f"Missing files for {var}")

            if len(missing_files) > 0:
                print(f"Returing list of missing files for hydrometgrid option")
                return missing_files
