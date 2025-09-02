import numpy as np
import rasterio
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from pytRIBS.shared.aux import Aux
from pytRIBS.shared.inout import InOut


class LandProcessor(InOut):
    @staticmethod
    def discrete_colormap(N, base_cmap=None):
        cmap = Aux.discrete_cmap(N, base_cmap)
        return cmap
    @staticmethod
    def update_landfiles_with_dates(file_path, date_str):
        Aux.rename_file_with_date(file_path,date_str)
        return
    @staticmethod
    def unsupervised_classification_naip(image_path, output_file_path, method='NDVI', n_clusters=4,
                                         plot_result=True):
        """
        Perform unsupervised classification on a NAIP image using K-means clustering.

        Parameters
        ----------
        image_path : str
            Path to the NAIP image file.
        output_file_path : str
            Path to save the classified image.
        method : str, optional
            Method to use for classification, either 'NDVI' or 'true_color'. Default is 'NDVI'.
        n_clusters : int, optional
            Number of clusters for K-means clustering. Default is 5.
        plot_result : bool, optional
            If True, the classified image will be plotted. Default is True.

        Returns
        -------
        np.ndarray
            The classified image with the same dimensions as the input image.

        Examples
        --------
        To classify an image using NDVI and 5 clusters:

        >>> classified_image = perform_kmeans_classification("path/to/naip_image.tif", "path/to/output.tif")
        >>> print(classified_image.shape)

        To classify an image using true color with 3 clusters and no plotting:

        >>> classified_image = perform_kmeans_classification("path/to/naip_image.tif", "path/to/output.tif", method='true_color', n_clusters=3, plot_result=False)

        """

        def classify_ndvi(image):
            """
            Calculate NDVI from NAIP image.
            """
            # Calculate NDVI
            red = image[0].astype(float)
            nir = image[1].astype(float)

            mask = (red == 0) & (nir == 0)

            red[mask] = np.nan
            nir[mask] = np.nan

            ndvi = (nir - red) / (nir + red)
            ndvi[np.isnan(ndvi)] = -9999  # Use a nodata value that can be handled

            return ndvi, mask

        with rasterio.open(image_path) as src:
            image = src.read()
            profile = src.profile

        if method == 'NDVI':
            data, mask = classify_ndvi(image)
            profile.update(count=1)
        elif method == 'true_color':
            n_bands, n_rows, n_cols = image.shape
            data = image.reshape(n_bands, -1).T
            mask = image[0] == 0
        else:
            raise ValueError("Method must be 'NDVI' or 'true_color'")

        if method == 'NDVI':
            reshaped_data = data.reshape(-1, 1)
        else:
            reshaped_data = data.reshape(n_bands, -1).T

        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(reshaped_data)
        labels = kmeans.labels_

        if method == 'NDVI':
            classified_image = labels.reshape(data.shape)
        else:
            classified_image = labels.reshape(n_rows, n_cols)

        profile.update(
            dtype=rasterio.uint8,
            count=1,
            compress='lzw'
        )

        InOut.write_ascii({'data': classified_image, 'profile': profile}, output_file_path)

        if plot_result:
            plt.figure(figsize=(10, 10))
            classified_image_masked = np.ma.masked_where(mask, classified_image)
            plt.imshow(classified_image_masked, cmap='viridis')
            plt.title('Classified Image')
            plt.axis('off')
            plt.show()

        classes = np.unique(classified_image)
        class_list = []

        for cl in classes:
            class_list.append({
                'ID': cl,
                'a': None,
                'b1': None,
                'P': None,
                'S': None,
                'K': None,
                'b2': None,
                'Al': None,
                'h': None,
                'Kt': None,
                'Rs': None,
                'V': None,
                'LAI': None,
                'theta*_s': None,
                'theta*_t': None
            })

        return classified_image, class_list

    @staticmethod
    def classify_vegetation_height(raster_path, thresholds, output_path, plot_result=True):
        """
        Classifies vegetation height raster based on user-defined thresholds.

        Parameters
        ----------
        raster_path : str
            Path to the input tree height raster.
        thresholds : list of tuple
            Each tuple defines a range (min, max, class) and its class value.
            For example: ``[(0, 5, 1), (5, 10, 2), (10, 15, 3)]`` will classify heights
            from 0-5 as class 1, 5-10 as class 2, and so on.
            - The min and max values must be increasing.
            - On the first iteration, min is allowed to equal max; on subsequent iterations,
              min must be greater than the previous max.
        output_path : str
            Path to save the classified raster.
        plot_result : bool, optional
            If True, the classified image will be plotted. Default is True.

        Returns
        -------
        classified_image : np.ndarray
            The classified raster array.
        class_list : list of dict
            List of class attributes, where each dictionary represents a class range and its value.

        Raises
        ------
        ValueError
            If the min value is not greater than the previous max value, or if min equals
            max on any iteration other than the first.

        Examples
        --------
        To classify vegetation height based on predefined thresholds:

        >>> thresholds = [(0, 5, 1), (5, 10, 2), (10, 15, 3)]
        >>> classified_data, class_list = classify_vegetation_height(
        ...     raster_path="path/to/raster.tif",
        ...     thresholds=thresholds,
        ...     output_path="path/to/output.tif"
        ... )
        >>> print(classified_data.shape)

        """

        with rasterio.open(raster_path) as src:
            height_data = src.read(1)
            profile = src.profile

            classified_data = np.zeros_like(height_data, dtype=np.uint8)

            prev_max_val = None

            for i, (min_val, max_val, class_val) in enumerate(thresholds):
                if prev_max_val is not None:
                    if not (min_val >= prev_max_val or (i == 0 and min_val == max_val)):
                        raise ValueError(
                            "Min value must be greater than previous max value, or equal only on the first iteration.")

                if min_val != max_val:
                    classified_data[(height_data > min_val) & (height_data <= max_val)] = int(class_val)
                elif i == 0 and min_val == max_val:
                    classified_data[(height_data == max_val)] = int(class_val)
                else:
                    raise ValueError("Min and max can only be equal on the first iteration.")

                prev_max_val = max_val

            profile.update(dtype=rasterio.uint8, count=1, compress='lzw')

            InOut.write_ascii({'data': classified_data, 'profile': profile}, output_path)

        if plot_result:
            plt.figure(figsize=(10, 10))
            plt.imshow(classified_data, cmap='viridis')
            plt.title('Classified Vegetation Height')
            plt.axis('off')
            plt.show()

        classes = np.unique(classified_data)
        class_list = []

        for cl in classes:
            class_list.append({
                'ID': cl,
                'a': None,
                'b1': None,
                'P': None,
                'S': None,
                'K': None,
                'b2': None,
                'Al': None,
                'h': None,
                'Kt': None,
                'Rs': None,
                'V': None,
                'LAI': None,
                'theta*_s': None,
                'theta*_t': None
            })

        return classified_data, class_list
    def _polygon_centroid_to_geographic(self, polygon, utm_crs=None, geographic_crs="EPSG:4326"):
        lat,lon, gmt = Aux.polygon_centroid_to_geographic(self,polygon,utm_crs=utm_crs,geographic_crs=geographic_crs)
        return lat, lon, gmt

    def create_gdf_content(self, parameters, watershed):
        """
        Create a dictionary containing geographic and parameter information for a watershed.

        This function computes the geographic centroid of the given watershed polygon,
        converts it to geographic coordinates (latitude and longitude), and then creates
        a dictionary containing the number of parameters, the centroid's geographic location,
        the GMT time zone, and a list of parameters.

        Parameters
        ----------
        parameters : list of list
            A list where each element is a list containing:
            - parameter name (str): The name of the parameter (e.g., 'VH').
            - raster path (str): The file path to the specified raster.
            - file extension (str): The extension of the raster file (e.g., '.tif').
        watershed : GeoDataFrame or shapely.geometry.Polygon
            The watershed boundary used to compute the centroid and derive geographic coordinates.

        Returns
        -------
        dict
            A dictionary containing the following key-value pairs:
            - 'Number of Parameters' : int
                The number of parameters provided.
            - 'Latitude' : float
                The latitude of the watershed centroid in geographic coordinates.
            - 'Longitude' : float
                The longitude of the watershed centroid in geographic coordinates.
            - 'GMT Time Zone' : int
                The GMT time zone derived from the geographic coordinates.
            - 'Parameters' : list
                The original list of parameters provided.

        Examples
        --------
        To create a dictionary of geographic and parameter information for a watershed:

        >>> parameters = [
        ...     ['VH', 'path/to/raster1.tif', '.tif'],
        ...     ['NDVI', 'path/to/raster2.tif', '.tif']
        ... ]
        >>> watershed = geopandas.read_file('path/to/watershed.shp')
        >>> gdf_content = create_gdf_content(parameters, watershed)
        >>> print(gdf_content['Latitude'], gdf_content['Longitude'])

        """

        num_params = len(parameters)
        lat, lon, gmt = self._polygon_centroid_to_geographic(watershed)

        land_gdf_content = {'Number of Parameters': num_params,
                            'Latitude': lat,  # update
                            'Longitude': lon,
                            'GMT Time Zone': gmt,
                            'Parameters': parameters
                            }
        return  land_gdf_content
