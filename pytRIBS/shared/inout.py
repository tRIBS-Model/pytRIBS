import os
import collections
from datetime import datetime
import getpass
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import math
from shapely.geometry import Point, LineString, Polygon
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

    @staticmethod
    def write_mesh_files(output_prefix, output_path, vertices, triangles, node_codes):
        """
        Write the four tRIBS mesh files describing a complete, pre-triangulated TIN.

        Produces ``{output_path}{output_prefix}.z``, ``.nodes``, ``.edges`` and
        ``.tri``. Together these fully specify the mesh connectivity (node spokes,
        directed edges with CCW ``nextid`` ordering, and per-triangle edge and
        neighbour tables) so that tRIBS can read the mesh directly without running
        the meshbuilder.

        Parameters
        ----------
        output_prefix : str
            Base name for the four output files (without extension).
        output_path : str
            Directory prefix prepended to ``output_prefix``. Should include a
            trailing separator (e.g. ``'data/model/mesh/'``).
        vertices : numpy.ndarray
            Array of shape (nnodes, 3) of node x, y, z coordinates.
        triangles : numpy.ndarray
            Array of shape (ntri, 3) of node indices for each triangle.
        node_codes : array_like
            Per-node boundary code (0=interior, 1=boundary, 2=outlet, 3=stream).

        Returns
        -------
        None
        """
        print("\n--- Preparing to write tRIBS mesh files ---")
        nnodes, ntri = len(vertices), len(triangles)

        # Ensure CCW winding
        for i in range(ntri):
            p0, p1, p2 = triangles[i]
            area = 0.5 * (
                vertices[p0, 0] * (vertices[p1, 1] - vertices[p2, 1]) +
                vertices[p1, 0] * (vertices[p2, 1] - vertices[p0, 1]) +
                vertices[p2, 0] * (vertices[p0, 1] - vertices[p1, 1])
            )
            if area < 0:
                triangles[i] = [p0, p2, p1]

        undirected_edges = {
            tuple(sorted((tri[i], tri[(i + 1) % 3])))
            for tri in triangles for i in range(3)
        }
        edge_list, directed_edge_to_id = [], {}
        for p1, p2 in sorted(undirected_edges):
            directed_edge_to_id[(p1, p2)] = len(edge_list); edge_list.append([p1, p2])
            directed_edge_to_id[(p2, p1)] = len(edge_list); edge_list.append([p2, p1])
        nedges = len(edge_list)

        spokes = collections.defaultdict(list)
        for i, edge in enumerate(edge_list):
            spokes[edge[0]].append(i)

        node_edgid = -np.ones(nnodes, dtype=int)
        edge_nextid = -np.ones(nedges, dtype=int)

        for node_id, edge_ids in spokes.items():
            angles = [
                (np.arctan2(
                    vertices[edge_list[eid][1], 1] - vertices[node_id, 1],
                    vertices[edge_list[eid][1], 0] - vertices[node_id, 0]
                ), eid)
                for eid in edge_ids
            ]
            angles.sort()
            sorted_eids = [eid for _, eid in angles]
            node_edgid[node_id] = sorted_eids[0]
            for i in range(len(sorted_eids)):
                edge_nextid[sorted_eids[i]] = sorted_eids[(i + 1) % len(sorted_eids)]

        undirected_edge_to_tris = collections.defaultdict(list)
        for i, tri in enumerate(triangles):
            for j in range(3):
                key = tuple(sorted((tri[j], tri[(j + 1) % 3])))
                undirected_edge_to_tris[key].append(i)

        tri_neighbors = -np.ones((ntri, 3), dtype=int)
        for i, tri in enumerate(triangles):
            for j in range(3):
                key = tuple(sorted((tri[j], tri[(j + 1) % 3])))
                nbrs = undirected_edge_to_tris[key]
                if len(nbrs) == 2:
                    tri_neighbors[i, j] = nbrs[1] if nbrs[0] == i else nbrs[0]

        with open(f"{output_path}{output_prefix}.z", "w") as f:
            f.write("0.000000\n")
            f.write(f"{nnodes}\n")
            np.savetxt(f, vertices[:, 2], fmt='%.6f')

        with open(f"{output_path}{output_prefix}.nodes", "w") as f:
            f.write("0.000000\n")
            f.write(f"{nnodes}\n")
            for i in range(nnodes):
                if node_edgid[i] == -1:
                    raise RuntimeError(f"Node {i} is isolated (no edges).")
                f.write(f"{vertices[i, 0]:.6f} {vertices[i, 1]:.6f} {node_edgid[i]} {node_codes[i]}\n")

        with open(f"{output_path}{output_prefix}.edges", "w") as f:
            f.write("0.000000\n")
            f.write(f"{nedges}\n")
            for i in range(nedges):
                f.write(f"{edge_list[i][0]} {edge_list[i][1]} {edge_nextid[i]}\n")

        with open(f"{output_path}{output_prefix}.tri", "w") as f:
            f.write("0.000000\n")
            f.write(f"{ntri}\n")
            for i in range(ntri):
                p0, p1, p2 = triangles[i]
                n0 = tri_neighbors[i, 1]   # opposite p0 → shares edge p1-p2
                n1 = tri_neighbors[i, 2]   # opposite p1 → shares edge p2-p0
                n2 = tri_neighbors[i, 0]   # opposite p2 → shares edge p0-p1
                e0 = directed_edge_to_id[(p0, p2)]  # origin=p0, dest=p2
                e1 = directed_edge_to_id[(p1, p0)]  # origin=p1, dest=p0
                e2 = directed_edge_to_id[(p2, p1)]  # origin=p2, dest=p1
                f.write(f"{p0} {p1} {p2} {n0} {n1} {n2} {e0} {e1} {e2}\n")

        print("\n--- All tRIBS mesh files have been generated successfully. ---")

    @staticmethod
    def write_mesh_diagnostics(output_base, vertices, triangles, node_codes, crs=None):
        """
        Write diagnostic shapefiles for visual inspection of a generated mesh.

        Produces three shapefiles: ``{output_base}_triangles.shp``,
        ``{output_base}_nodes.shp`` and ``{output_base}_edges.shp``. Each feature
        carries a ``code`` attribute (0=Interior, 1=Boundary, 2=Outlet, 3=Stream)
        for checking that boundary, outlet and stream nodes are placed correctly.

        Parameters
        ----------
        output_base : str
            Base path (without extension) for the three shapefiles.
        vertices : numpy.ndarray
            Array of shape (nnodes, 3) of node x, y, z coordinates.
        triangles : numpy.ndarray
            Array of shape (ntri, 3) of node indices for each triangle.
        node_codes : array_like
            Per-node boundary code (0=interior, 1=boundary, 2=outlet, 3=stream).
        crs : optional
            Coordinate reference system passed through to GeoPandas.

        Returns
        -------
        None
        """
        print(f"\n--- Writing diagnostic shapefiles to {output_base}_*.shp ---")

        polys, tri_codes = [], []
        for tri_indices in triangles:
            polys.append(Polygon(vertices[tri_indices][:, :2]))
            codes_in_tri = node_codes[tri_indices]
            if 2 in codes_in_tri:
                tri_codes.append(2)
            elif 3 in codes_in_tri:
                tri_codes.append(3)
            elif 1 in codes_in_tri:
                tri_codes.append(1)
            else:
                tri_codes.append(0)
        gpd.GeoDataFrame({'code': tri_codes}, geometry=polys, crs=crs).to_file(
            f"{output_base}_triangles.shp", driver='ESRI Shapefile'
        )

        pts = [Point(v[0], v[1]) for v in vertices]
        gpd.GeoDataFrame(
            {'code': node_codes, 'elev': vertices[:, 2]},
            geometry=pts, crs=crs
        ).to_file(f"{output_base}_nodes.shp", driver='ESRI Shapefile')

        _priority = {0: 1, 1: 2, 3: 0, 2: 3}
        seen = set()
        edge_lines, edge_codes = [], []
        for tri in triangles:
            for k in range(3):
                i, j = tri[k], tri[(k + 1) % 3]
                key = (min(i, j), max(i, j))
                if key in seen:
                    continue
                seen.add(key)
                edge_lines.append(LineString([vertices[i, :2], vertices[j, :2]]))
                ci, cj = int(node_codes[i]), int(node_codes[j])
                edge_codes.append(ci if _priority.get(ci, 0) >= _priority.get(cj, 0) else cj)
        gpd.GeoDataFrame({'code': edge_codes}, geometry=edge_lines, crs=crs).to_file(
            f"{output_base}_edges.shp", driver='ESRI Shapefile'
        )
        print(f"  Wrote {len(polys)} triangles, {len(pts)} nodes, {len(edge_lines)} edges.")

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

    @staticmethod
    def read_sdf(file_path):
        """
        Reads a tRIBS station descriptor file (*.sdf) and returns a list of station
        dictionaries. Both the precipitation and hydrometeorological station files share this
        format: a single descriptive header line that tRIBS skips, followed by comma-delimited
        rows of ID,DataFile,Northing,Easting,Elevation:

        ID,DataFile,Northing,Easting,Elevation
        1,data/model/precip/precip_U.mdf,3891469.290931,400109.778323,2188.5

        :param file_path: Path to the *.sdf file.
        :return: List of dictionaries, one per station, with keys 'station_id', 'file_path',
            'y' (northing), 'x' (easting), and 'elevation'.
        """
        station_list = []

        with open(file_path, 'r') as file:
            lines = file.readlines()

        lines.pop(0)  # discard the descriptive header line (skipped by tRIBS)

        for l in lines:
            if not l.strip():
                continue

            info = [v.strip() for v in l.strip().split(',')]
            if len(info) == 5:
                station_id, data_file, northing, easting, elevation = info
                station_list.append({
                    "station_id": station_id,
                    "file_path": data_file,
                    "y": float(northing),
                    "x": float(easting),
                    "elevation": float(elevation)
                })
            else:
                print(f"Skipping row in {file_path}: expected 5 comma-separated values, "
                      f"got {len(info)}.")

        return station_list

    def read_precip_sdf(self, file_path=None):
        """
        Returns list of precip stations read from the *.sdf referenced by the GAUGESTATIONS
        option (or a separately specified file_path). See read_sdf for the file format.
        :param file_path: Defaults to options["gaugestations"]["value"].
        :return: List of dictionaries.
        """
        if file_path is None:
            file_path = self.options["gaugestations"]["value"]

            if file_path is None:
                print(self.options["gaugestations"]["keyword"] + " is not specified.")
                return None

        return self.read_sdf(file_path)

    @staticmethod
    def read_precip_station(file_path):
        """
        Returns pandas dataframe of precipitation from a station specified by file_path.
        :param file_path: tRIBS precip data file (*.mdf), comma-delimited with header Y,M,D,H,R
        :return: Pandas dataframe
        """
        # TODO add var for specifying Station ID
        df = pd.read_csv(file_path, header=0, sep=',')
        df.rename(columns={'Y': 'year', 'M': 'month', 'D': 'day', 'H': 'hour'}, inplace=True)
        df['date'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
        df.drop(['year', 'month', 'day', 'hour'], axis=1, inplace=True)

        return df

    @staticmethod
    def write_sdf(station_list, output_file_path):
        """
        Writes a list of station dictionaries to a tRIBS station descriptor file (*.sdf):
        a single descriptive header line (skipped by tRIBS) followed by comma-delimited rows of
        ID,DataFile,Northing,Easting,Elevation. Used for both precipitation and
        hydrometeorological station files.

        :param station_list: List of dictionaries with keys 'station_id', 'file_path',
            'y' (northing), 'x' (easting), and 'elevation'.
        :param output_file_path: Output *.sdf path.
        """
        with open(output_file_path, 'w') as file:
            file.write("ID,DataFile,Northing,Easting,Elevation\n")
            for station in station_list:
                file.write(f"{station['station_id']},{station['file_path']},"
                           f"{station['y']},{station['x']},{station['elevation']}\n")

    @staticmethod
    def write_precip_sdf(station_list, output_file_path):
        """
        Writes a list of precip stations to a *.sdf file. See write_sdf for the file format.
        :param station_list: List of dictionaries containing station information.
        :param output_file_path: Output *.sdf path.
        """
        InOut.write_sdf(station_list, output_file_path)

    @staticmethod
    def write_precip_station(df, output_file_path):
        """
        Converts a DataFrame with 'date' and 'R' columns to the tRIBS precip data file
        (*.mdf) format: comma-delimited with header Y,M,D,H,R.
        :param df: Pandas DataFrame with 'date' and 'R' columns.
        :param output_file_path: Output *.mdf path.
        """
        # Extract Y, M, D, and H from the 'date' column
        df['Y'] = df['date'].dt.year
        df['M'] = df['date'].dt.month
        df['D'] = df['date'].dt.day
        df['H'] = df['date'].dt.hour

        # Reorder columns
        df = df[['Y', 'M', 'D', 'H', 'R']]

        # Write DataFrame to flat file
        df.to_csv(output_file_path, sep=',', index=False)

    def read_met_sdf(self, file_path=None):
        """
        Returns list of met stations read from the *.sdf referenced by the HYDROMETSTATIONS
        option (or a separately specified file_path). See read_sdf for the file format.
        :param file_path: Defaults to options["hydrometstations"]["value"].
        :return: List of dictionaries.
        """
        if file_path is None:
            file_path = self.options["hydrometstations"]["value"]

            if file_path is None:
                print(self.options["hydrometstations"]["keyword"] + " is not specified.")
                return None

        return self.read_sdf(file_path)

    @staticmethod
    def read_met_station(file_path):
        """
        Reads a meteorological station data file and processes it into a pandas DataFrame with a datetime index.

        Parameters
        ----------
        file_path : str
            Path to the *.mdf file. The file is comma-delimited with the header
            Year,Month,Day,Hour,PA_mb,RH_pct,XC_tenths,US_m/s,TA_C,IS_W/m2,TS_C.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with a single 'date' column built from Year/Month/Day/Hour, and the
            meteorological variables under their short names: PA, RH, XC, US, TA, IS, TS.

        Notes
        -----
        - The descriptive, unit-bearing header columns are mapped to the short variable names.
        - Year/Month/Day/Hour are combined into a single 'date' column and dropped.
        """
        df = pd.read_csv(file_path, header=0, sep=',')
        df.rename(columns={'Year': 'year', 'Month': 'month', 'Day': 'day', 'Hour': 'hour',
                           'PA_mb': 'PA', 'RH_pct': 'RH', 'XC_tenths': 'XC', 'US_m/s': 'US',
                           'TA_C': 'TA', 'IS_W/m2': 'IS', 'TS_C': 'TS'}, inplace=True)
        df['date'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
        df = df.drop(['year', 'month', 'day', 'hour'], axis=1)
        return df

    @staticmethod
    def write_met_station(df, output_file_path):
        """
        Converts a DataFrame with a 'date' column and meteorological variables to the tRIBS 
         meteorological data file (*.mdf) format: comma-delimited with the header
        Year,Month,Day,Hour,PA_mb,RH_pct,XC_tenths,US_m/s,TA_C,IS_W/m2,TS_C.

        Relative humidity (RH) is the only accepted humidity input in v6.0.0; the pre-v6.0.0
        TD/VP alternatives and the unused net radiation (NR) column have been removed. Cloud
        cover (XC) and surface temperature (TS) are required columns but are typically 9999.99;
        if absent from df they are written as 9999.99.

        :param df: Pandas DataFrame with a 'date' column and at least 'PA', 'RH', 'US', 'TA',
            and 'IS' columns.
        :param output_file_path: Output *.mdf path.
        """
        if 'RH' not in df.columns:
            print("Error: 'RH' (relative humidity) is required for the met data file.")
            return

        df = df.copy()

        # Extract Year, Month, Day, Hour from the 'date' column
        df['Year'] = df['date'].dt.year
        df['Month'] = df['date'].dt.month
        df['Day'] = df['date'].dt.day
        df['Hour'] = df['date'].dt.hour

        # XC and TS are required columns but are typically the 9999.99 fill value
        if 'XC' not in df.columns:
            df['XC'] = 9999.99
        if 'TS' not in df.columns:
            df['TS'] = 9999.99

        columns = ['Year', 'Month', 'Day', 'Hour', 'PA', 'RH', 'XC', 'US', 'TA', 'IS', 'TS']
        header = 'Year,Month,Day,Hour,PA_mb,RH_pct,XC_tenths,US_m/s,TA_C,IS_W/m2,TS_C'

        with open(output_file_path, 'w') as f:
            f.write(header + '\n')
            df[columns].to_csv(f, header=False, index=False)


    @staticmethod
    def write_met_sdf(station_list, output_file_path):
        """
        Writes a list of meteorological stations to a *.sdf file. See write_sdf for the file
        format.

        Note: the argument order changed in v1.0.0 to (station_list, output_file_path) to match
        write_precip_sdf.

        :param station_list: List of dictionaries containing station information.
        :param output_file_path: Output *.sdf path.
        """
        InOut.write_sdf(station_list, output_file_path)

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
                print(self.landtablename["keyword"] + " is not specified.")
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

        if not landuse_list:
            print(f"Warning: no land use entries were read from {file_path}. Confirm it is in the "
                  f"tRIBS format (one descriptive header line, then comma-delimited rows). "
                  f"Pre-v6.0.0 tables (a count line with whitespace-delimited rows) are not "
                  f"compatible and must be converted.")

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
        Returns dictionary with content of a specified Grid Data File (.gdf).

        The .gdf has a single descriptive header line that tRIBS skips,
        followed by comma-delimited rows of Variable,BasePath,FileExtension:

        Variable,BasePath,FileExtension
        KS,data/model/soil/KS,asc
        TS,data/model/soil/TS,asc

        :param grid_type: string set to "weather", "soil", or "land", with each corresponding to HYDROMETGRID, SCGRID, LUGRID
        :return: dictionary with keys "Number of Parameters" and "Parameters" (a list of dicts)
        """

        if grid_type == "weather":
            option = self.options["hydrometgrid"]["value"]
        elif grid_type == "soil":
            option = self.options["scgrid"]["value"]
        elif grid_type == "land":
            option = self.options["lugrid"]["value"]

        parameters = []

        with open(option, 'r') as file:
            lines = file.readlines()

        lines.pop(0)  # discard the descriptive header line (skipped by tRIBS)

        for line in lines:
            if not line.strip():
                continue

            parts = [v.strip() for v in line.strip().split(',')]
            if len(parts) == 3:
                variable_name, raster_path, raster_extension = parts
                parameters.append({
                    'Variable Name': variable_name,
                    'Raster Path': raster_path,
                    'Raster Extension': raster_extension
                })
            else:
                print(f"Skipping invalid line: {line}")

        return {
            'Number of Parameters': len(parameters),
            'Parameters': parameters
        }

    @staticmethod
    def write_grid_data_file(grid_file, data):
        """
        Writes the content of a dictionary to a specified Grid Data File (.gdf) in the tRIBS
        format: a single descriptive header line (skipped by tRIBS) followed by
        comma-delimited rows of Variable,BasePath,FileExtension.

        :param grid_file: path to write out grid file to.
        :param data: dictionary containing key "Parameters" (a list of dicts with keys
            'Variable Name', 'Raster Path', 'Raster Extension')
        :return: None
        """

        with open(grid_file, 'w') as file:
            file.write("Variable,BasePath,FileExtension\n")

            for param in data['Parameters']:
                file.write(f"{param['Variable Name']},{param['Raster Path']},"
                           f"{param['Raster Extension']}\n")

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
    def write_node_file(nodes, file_path, coords=False):
        """
        Writes a tRIBS node-list file (*.nol) used by the NODEOUTPUTLIST,
        HYDRONODELIST, and OUTLETNODELIST options.

        As of tRIBS v6.0.0 these files are CSV with a single header line that is
        either ``ID`` or ``X,Y``, followed by one row per node (an ID, or an X,Y
        coordinate pair).

        :param nodes: Sequence of node IDs (when ``coords=False``), or a sequence of
            ``(x, y)`` coordinate pairs (when ``coords=True``).
        :param file_path: Output file path.
        :param coords: If True, write an ``X,Y`` coordinate file; otherwise write an
            ``ID`` file. Defaults to False.
        """
        with open(file_path, 'w') as file:
            if coords:
                file.write("X,Y\n")
                for x, y in nodes:
                    file.write(f"{x},{y}\n")
            else:
                file.write("ID\n")
                for node_id in nodes:
                    file.write(f"{node_id}\n")

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