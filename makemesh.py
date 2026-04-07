import pandas as pd
import numpy as np
import geopandas as gpd
import triangle as tr
from scipy.spatial import cKDTree
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import unary_union, snap, linemerge
import collections
import rasterio
import os
from pathlib import Path


def sample_elevations_from_dem(points_xy, dem_path):
    """Samples elevation values for a list of (x, y) points from a DEM."""
    print(f"  - Reading DEM: {dem_path}")
    with rasterio.open(dem_path) as src:
        coords = [tuple(p) for p in points_xy]
        elevations = np.array([val[0] for val in src.sample(coords)])
        nodata = src.nodata
        if nodata is not None:
            elevations[elevations == nodata] = np.nan
        print(f"  - Sampled {len(elevations)} points.")
        return elevations


def _normalize_coords(xy, target_span=1e4):
    xy = np.asarray(xy, float)
    origin = xy.min(axis=0)
    shifted = xy - origin
    span = shifted.max(axis=0)
    scale = max(span.max() / target_span, 1.0)
    return shifted / scale, origin, scale


def _denormalize_coords(xy_norm, origin, scale):
    return np.asarray(xy_norm) * scale + origin


def _cumlens(coords):
    d = np.diff(coords, axis=0)
    seg = np.sqrt((d * d).sum(axis=1))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    return s, seg


def _remove_spikes(coords, angle_tol_deg):
    if len(coords) <= 2:
        return coords
    keep = [coords[0]]
    cos_tol = np.cos(np.deg2rad(angle_tol_deg))
    for i in range(1, len(coords) - 1):
        a, b, c = coords[i - 1], coords[i], coords[i + 1]
        u, v = b - a, c - b
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        if nu == 0 or nv == 0:
            continue
        cosang = float(np.dot(u, v) / (nu * nv))
        # Keep if not near-collinear (cosang ~ 1) and not a spike reversal (cosang ~ -1)
        if not (cosang > cos_tol or cosang < -cos_tol):
            keep.append(b)
    keep.append(coords[-1])
    return np.asarray(keep)


def _node_streams(lines, tau_snap=1.0):
    """Snap near-coincident vertices and node all intersections once."""
    ml = linemerge(unary_union(snap(unary_union(lines), unary_union(lines), tau_snap)))
    if isinstance(ml, LineString):
        return [ml]
    return list(ml.geoms)


def _resample_with_min_spacing(line, spacing, m_min, angle_tol_deg=2.0, keep_original_vertices=False):
    """
    Resample a LineString at regular spacing, enforcing a minimum consecutive spacing.
    If keep_original_vertices=False (default), only endpoints + uniform grid points are kept.
    """
    coords = np.asarray(line.coords, float)
    if coords.ndim == 2 and coords.shape[1] > 2:
        coords = coords[:, :2]
    coords = _remove_spikes(coords, angle_tol_deg) if angle_tol_deg > 0 else coords
    s, seg = _cumlens(coords)
    L = float(s[-1])
    if L == 0:
        return coords[:1]

    s_grid = np.arange(0.0, L + 1e-9, spacing)
    s_orig = s if keep_original_vertices else np.array([0.0, L])
    all_s = np.unique(np.round(np.concatenate([s_orig, s_grid]), 9))

    kept = []
    seg_idx = 0
    for t in all_s:
        while seg_idx < len(seg) - 1 and s[seg_idx + 1] < t - 1e-12:
            seg_idx += 1
        k = np.where(np.isclose(s, t, rtol=0, atol=1e-9))[0]
        if k.size:
            pt = coords[k[0]]
        else:
            t0, t1 = s[seg_idx], s[seg_idx + 1]
            if t1 <= t0:
                continue
            a = (t - t0) / (t1 - t0)
            pt = coords[seg_idx] * (1 - a) + coords[seg_idx + 1] * a

        if not kept or np.linalg.norm(pt - kept[-1]) >= m_min - 1e-12:
            kept.append(pt)

    if np.linalg.norm(kept[-1] - coords[-1]) > 1e-9:
        kept.append(coords[-1])

    out = [kept[0]]
    for p in kept[1:]:
        if np.linalg.norm(p - out[-1]) >= m_min - 1e-12:
            out.append(p)
    if np.linalg.norm(out[-1] - coords[-1]) > 1e-9:
        out.append(coords[-1])
    return np.asarray(out)


def _split_line_at_point(line, point):
    """
    Splits a shapely LineString at a projected point.
    Returns a list of one or two new LineStrings.
    """
    coords = list(line.coords)
    if point.coords[0] in coords:
        return [line]

    distance = line.project(point)
    if distance <= 1e-9 or distance >= line.length - 1e-9:
        return [line]

    pre_coords = []
    post_coords = [point.coords[0]]

    current_dist = 0.0
    for i in range(len(coords) - 1):
        p1 = Point(coords[i])
        p2 = Point(coords[i + 1])
        segment = LineString([p1, p2])

        if current_dist <= distance < current_dist + segment.length:
            pre_coords.extend(coords[:i + 1])
            pre_coords.append(point.coords[0])
            post_coords.extend(coords[i + 1:])
            return [LineString(pre_coords), LineString(post_coords)]

        current_dist += segment.length

    return [line]


def _build_stream_pslg(resampled_edges, m_min, dedupe_radius):
    """
    Global dedupe within ~dedupe_radius and emit (points, segments).
    Always returns (points_xy[N,2], segments[List[Tuple[int,int]]]).
    """
    clean_edges = []
    for arr in (resampled_edges or []):
        if arr is None:
            continue
        a = np.asarray(arr, float)
        if a.ndim != 2 or a.shape[0] < 2:
            continue
        if a.shape[1] > 2:
            a = a[:, :2]
        clean_edges.append(a)

    if not clean_edges:
        return np.empty((0, 2), dtype=float), []

    P = np.vstack(clean_edges)

    kdt = cKDTree(P)
    pairs = kdt.query_pairs(r=max(float(dedupe_radius), 1e-9))

    parent = np.arange(len(P))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, j in pairs:
        union(i, j)

    rep_to_new = {}
    new_index = np.empty(len(P), dtype=int)
    uniq = []
    for i in range(len(P)):
        r = find(i)
        if r not in rep_to_new:
            rep_to_new[r] = len(uniq)
            uniq.append(P[r])
        new_index[i] = rep_to_new[r]
    uniq = np.asarray(uniq, float)

    segs = set()
    cursor = 0
    for arr in clean_edges:
        n = len(arr)
        idxs = new_index[cursor:cursor + n]
        cursor += n
        prev = idxs[0]
        for cur in idxs[1:]:
            if cur != prev:
                segs.add((prev, cur))
            prev = cur

    tiny = max(0.5, 0.25 * float(dedupe_radius))
    segments = []
    for i, j in segs:
        if np.linalg.norm(uniq[i] - uniq[j]) >= tiny - 1e-12:
            segments.append((i, j))

    return uniq, segments


def _sample_watershed_boundary(watershed_poly, buffer_dist, spacing):
    """
    Buffers the watershed polygon outward and samples points along the
    exterior ring at regular spacing. Returns points in polygon order
    and the buffered exterior ring geometry.
    """
    buffered = watershed_poly.buffer(buffer_dist)
    exterior = buffered.exterior
    total_len = exterior.length
    distances = np.arange(0, total_len, spacing)
    pts = np.array([[exterior.interpolate(d).x, exterior.interpolate(d).y]
                    for d in distances])
    return pts, exterior


def _find_outlet_on_buffer(lines, watershed_poly, buffered_exterior):
    """
    Auto-detects the outlet as the stream endpoint nearest to the watershed
    polygon exterior, then projects it onto the buffered exterior ring.
    Returns the outlet Point on the buffered boundary.
    """
    ws_exterior = watershed_poly.exterior
    min_dist = float('inf')
    outlet_stream_pt = None

    for ln in lines:
        for coord in [ln.coords[0], ln.coords[-1]]:
            pt = Point(coord[0], coord[1])  # ensure 2D
            d = ws_exterior.distance(pt)
            if d < min_dist:
                min_dist = d
                outlet_stream_pt = pt

    outlet_on_buffer = buffered_exterior.interpolate(
        buffered_exterior.project(outlet_stream_pt)
    )
    outlet_on_buffer = Point(outlet_on_buffer.x, outlet_on_buffer.y)  # ensure 2D

    print(f"  Auto-detected outlet near stream endpoint "
          f"({outlet_stream_pt.x:.1f}, {outlet_stream_pt.y:.1f}), "
          f"{min_dist:.1f}m from watershed boundary.")
    print(f"  Outlet placed on buffered boundary at "
          f"({outlet_on_buffer.x:.1f}, {outlet_on_buffer.y:.1f}).")
    return outlet_on_buffer


# ==============================================================================
# The Main Meshing Function
# ==============================================================================

def generate_mesh_from_points(
    points_file,
    stream_shapefile,
    dem_file,
    watershed_shapefile,
    boundary_buffer=50.0,
    boundary_spacing=50.0,
    tree_centroids_shapefile=None,
    tree_remove_radius=5.0,
    stream_clearance_radius=15.0,
    stream_point_spacing=25.0,
    mesh_quality_opts='p'
):
    # -------------------------------------------------------------------------
    # Step 1: Build boundary from watershed polygon
    # -------------------------------------------------------------------------
    print("\nStep 1: Building boundary from watershed polygon...")
    watershed_gdf = gpd.read_file(watershed_shapefile)
    watershed_poly = watershed_gdf.geometry.union_all()
    if watershed_poly.geom_type == 'MultiPolygon':
        watershed_poly = max(watershed_poly.geoms, key=lambda g: g.area)

    boundary_pts_xy, buffered_exterior = _sample_watershed_boundary(
        watershed_poly, boundary_buffer, boundary_spacing
    )
    print(f"  Generated {len(boundary_pts_xy)} boundary points "
          f"(buffer={boundary_buffer}m, spacing={boundary_spacing}m).")

    # -------------------------------------------------------------------------
    # Step 2: Load and node stream network
    # -------------------------------------------------------------------------
    print("\nStep 2: Loading and noding stream network...")
    streams_gdf = gpd.read_file(stream_shapefile)
    raw_lines = []
    for g in streams_gdf.geometry.dropna():
        if isinstance(g, LineString):
            raw_lines.append(g)
        elif g.geom_type == "MultiLineString":
            raw_lines.extend(list(g.geoms))

    lines = _node_streams(raw_lines, tau_snap=1.0)
    print(f"  {len(raw_lines)} raw lines -> {len(lines)} noded edges.")

    # -------------------------------------------------------------------------
    # Step 3: Detect outlet and integrate into stream network
    #         (must happen BEFORE resampling so the connector gets included)
    # -------------------------------------------------------------------------
    print("\nStep 3: Detecting outlet and integrating into stream network...")
    outlet_pt = _find_outlet_on_buffer(lines, watershed_poly, buffered_exterior)

    try:
        min_dist = float('inf')
        closest_line_idx = -1
        for i, ln in enumerate(lines):
            d = ln.distance(outlet_pt)
            if d < min_dist:
                min_dist = d
                closest_line_idx = i

        if closest_line_idx != -1:
            line_to_split = lines[closest_line_idx]
            q_point = line_to_split.interpolate(line_to_split.project(outlet_pt))
            q_point = Point(q_point.x, q_point.y)  # ensure 2D
            connector_line = LineString([q_point, outlet_pt])
            split_parts = _split_line_at_point(line_to_split, q_point)
            lines.pop(closest_line_idx)
            lines.extend(split_parts)
            lines.append(connector_line)
            print(f"  Stream network connected to outlet (stream distance: {min_dist:.1f}m).")
    except Exception as e:
        print(f"  [WARNING] Outlet stream integration failed: {e}")

    # -------------------------------------------------------------------------
    # Step 4: Resample stream network and build stream PSLG
    # -------------------------------------------------------------------------
    print("\nStep 4: Resampling stream network...")
    MIN_STREAM_SPACING = 5.0
    DEDUPE_RADIUS = max(1.5, 0.05 * float(stream_point_spacing))
    resampled_edges = []
    for ln in lines:
        pts = _resample_with_min_spacing(
            ln, spacing=stream_point_spacing, m_min=MIN_STREAM_SPACING,
            angle_tol_deg=2.0, keep_original_vertices=False
        )
        if len(pts) >= 2:
            resampled_edges.append(pts)

    stream_nodes_xy, stream_segments_local = _build_stream_pslg(
        resampled_edges, m_min=MIN_STREAM_SPACING, dedupe_radius=DEDUPE_RADIUS
    )
    stream_nodes_df = pd.DataFrame(stream_nodes_xy, columns=['x', 'y'])
    stream_nodes_df['code'] = 3

    unified_stream_geom = linemerge(unary_union(lines))
    total_len_km = sum(ln.length for ln in lines) / 1000.0
    print(f"  {len(stream_nodes_df)} stream nodes, ~{stream_point_spacing}m spacing, "
          f"{total_len_km:.2f} km total stream length.")

    # Warn if streams are very close together (can cause quality cascade failures)
    if len(stream_nodes_xy) > 1:
        stream_kdt = cKDTree(stream_nodes_xy)
        close_pairs = stream_kdt.query_pairs(r=stream_point_spacing * 0.5)
        if close_pairs:
            print(f"  WARNING: {len(close_pairs)} stream node pairs are closer than "
                  f"{stream_point_spacing * 0.5:.1f}m. Consider removing redundant stream "
                  f"lines or reducing stream_point_spacing to avoid mesh quality failures.")

    # -------------------------------------------------------------------------
    # Step 5: Read interior terrain points from points file (code=0 only)
    # -------------------------------------------------------------------------
    print("\nStep 5: Reading interior terrain points from points file...")
    base_df = pd.read_csv(points_file, sep=r"\s+", skiprows=1, header=None,
                          names=['x', 'y', 'z', 'code'], comment='#')
    interior_pts = base_df[base_df['code'] == 0][['x', 'y', 'code']].copy().reset_index(drop=True)
    print(f"  {len(interior_pts)} interior terrain points loaded.")

    # Remove interior points that are too close to each other
    INTERIOR_REMOVE_RADIUS = 5.0
    if len(interior_pts) > 1:
        interior_tree = cKDTree(interior_pts[['x', 'y']].values)
        close_pairs = interior_tree.query_pairs(r=INTERIOR_REMOVE_RADIUS)
        drop_indices = {j for _, j in close_pairs}
        if drop_indices:
            print(f"  Removing {len(drop_indices)} interior points within "
                  f"{INTERIOR_REMOVE_RADIUS}m of another.")
            interior_pts = interior_pts.drop(index=list(drop_indices)).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Step 6: Build boundary DataFrame with outlet
    #         Replace the sampled boundary point nearest to the outlet with
    #         the exact outlet coordinates and mark it code=2.
    # -------------------------------------------------------------------------
    boundary_kdt = cKDTree(boundary_pts_xy)
    _, outlet_boundary_idx = boundary_kdt.query([outlet_pt.x, outlet_pt.y])
    boundary_pts_xy[outlet_boundary_idx] = [outlet_pt.x, outlet_pt.y]

    boundary_codes = np.ones(len(boundary_pts_xy), dtype=int)
    boundary_codes[outlet_boundary_idx] = 2

    boundary_df = pd.DataFrame(boundary_pts_xy, columns=['x', 'y'])
    boundary_df['code'] = boundary_codes
    n_boundary = len(boundary_df)

    # -------------------------------------------------------------------------
    # Step 7: Handle tree centroids (optional)
    # -------------------------------------------------------------------------
    tree_nodes_df = pd.DataFrame(columns=['x', 'y', 'code'])
    if tree_centroids_shapefile:
        print("\nStep 7: Processing tree centroids...")
        trees_gdf = gpd.read_file(tree_centroids_shapefile)
        tree_points_xy = np.array([[p.x, p.y] for p in trees_gdf.geometry])
        if tree_points_xy.size > 0:
            tree_kdt = cKDTree(tree_points_xy)
            if not interior_pts.empty:
                indices_to_drop = tree_kdt.query_ball_point(
                    interior_pts[['x', 'y']].values, r=tree_remove_radius
                )
                drop_set = {interior_pts.index[i] for i, n in enumerate(indices_to_drop) if n}
                if drop_set:
                    print(f"  Removing {len(drop_set)} interior points within "
                          f"{tree_remove_radius}m of a tree centroid.")
                    interior_pts = interior_pts.drop(list(drop_set))
            tree_nodes_df = pd.DataFrame(tree_points_xy, columns=['x', 'y'])
            tree_nodes_df['code'] = 0
            print(f"  Added {len(tree_nodes_df)} tree centroid points.")

    # -------------------------------------------------------------------------
    # Step 8: Enforce stream clearance zone on interior points
    # -------------------------------------------------------------------------
    all_interior_pts = pd.concat([interior_pts, tree_nodes_df], ignore_index=True)
    if stream_clearance_radius > 0 and not all_interior_pts.empty:
        print(f"\nStep 8: Applying {stream_clearance_radius}m stream clearance zone...")
        interior_geoseries = gpd.GeoSeries.from_xy(
            all_interior_pts['x'], all_interior_pts['y']
        )
        distances = interior_geoseries.distance(unified_stream_geom)
        too_close = all_interior_pts.index[distances < stream_clearance_radius]
        if not too_close.empty:
            print(f"  Removing {len(too_close)} interior points within clearance zone.")
            all_interior_pts = all_interior_pts.drop(too_close)

    # -------------------------------------------------------------------------
    # Step 9: Combine all points and sample elevations from DEM
    # -------------------------------------------------------------------------
    print("\nStep 9: Combining all points and sampling elevations from DEM...")
    # boundary_df first so its indices 0..n_boundary-1 align with boundary_segs_indices
    final_non_stream_pts = pd.concat([boundary_df, all_interior_pts], ignore_index=True)
    final_input_df = pd.concat(
        [final_non_stream_pts, stream_nodes_df], ignore_index=True
    ).reset_index(drop=True)

    final_input_df['z'] = sample_elevations_from_dem(
        final_input_df[['x', 'y']].values, dem_file
    )

    # -------------------------------------------------------------------------
    # Step 10: Define mesh constraints (PSLG segments)
    # -------------------------------------------------------------------------
    print("\nStep 10: Defining mesh constraints...")
    n_non_stream = len(final_non_stream_pts)

    # Boundary: polygon order means segments are simply consecutive pairs
    boundary_segs_indices = [(i, (i + 1) % n_boundary) for i in range(n_boundary)]

    # Stream: shift local indices by n_non_stream to index into final_input_df
    stream_segs_indices = [
        (i + n_non_stream, j + n_non_stream) for (i, j) in stream_segments_local
    ]

    all_segs = set(tuple(sorted(s)) for s in (boundary_segs_indices + stream_segs_indices))
    segments = [list(s) for s in all_segs]
    print(f"  {len(boundary_segs_indices)} boundary segments, "
          f"{len(stream_segs_indices)} stream segments.")

    # -------------------------------------------------------------------------
    # Step 11: Triangulate
    # -------------------------------------------------------------------------
    print("\nStep 11: Triangulating...")
    V = final_input_df[['x', 'y']].values
    Vn, origin_xy, scale_xy = _normalize_coords(V)
    mesh_data = {'vertices': Vn, 'segments': segments}

    tri_opts = f"p{mesh_quality_opts}D"
    mesh = tr.triangulate(mesh_data, tri_opts)
    mesh_vertices_xy = _denormalize_coords(np.asarray(mesh['vertices']), origin_xy, scale_xy)
    print(f"  Mesh: {len(mesh_vertices_xy)} vertices, {len(mesh['triangles'])} triangles.")

    # -------------------------------------------------------------------------
    # Step 12: Assign elevations to all mesh vertices (including Steiner points)
    # -------------------------------------------------------------------------
    print("\nStep 12: Sampling elevations for all mesh vertices...")
    mesh_vertices_z = sample_elevations_from_dem(mesh_vertices_xy, dem_file)
    final_vertices_3d = np.column_stack([mesh_vertices_xy, mesh_vertices_z])

    # -------------------------------------------------------------------------
    # Step 13: Assign node codes
    # -------------------------------------------------------------------------
    print("\nStep 13: Assigning node codes...")
    n_mesh_verts = len(mesh_vertices_xy)
    final_node_codes = np.zeros(n_mesh_verts, dtype=int)

    mesh_kdt = cKDTree(mesh_vertices_xy)
    _, mesh_indices_for_originals = mesh_kdt.query(final_input_df[['x', 'y']].values)

    for original_idx, mesh_idx in enumerate(mesh_indices_for_originals):
        final_node_codes[mesh_idx] = final_input_df['code'].iloc[original_idx]

    is_original_node = np.zeros(n_mesh_verts, dtype=bool)
    is_original_node[mesh_indices_for_originals] = True

    # Classify Steiner points by proximity to stream/boundary segments
    # Use list(s) not tuple(s) to avoid pandas .loc MultiIndex ambiguity
    final_boundary_segs = [
        LineString(final_input_df.loc[list(s), ['x', 'y']].values)
        for s in boundary_segs_indices
    ]
    final_stream_segs = [
        LineString(final_input_df.loc[list(s), ['x', 'y']].values)
        for s in stream_segs_indices
    ]

    for i in range(n_mesh_verts):
        if is_original_node[i]:
            continue
        p = Point(mesh_vertices_xy[i])
        if any(p.distance(seg) < 0.1 for seg in final_stream_segs):
            final_node_codes[i] = 3
        elif any(p.distance(seg) < 0.1 for seg in final_boundary_segs):
            final_node_codes[i] = 1

    print(f"  Code counts — 0 (Interior): {np.sum(final_node_codes == 0)}, "
          f"1 (Boundary): {np.sum(final_node_codes == 1)}, "
          f"2 (Outlet): {np.sum(final_node_codes == 2)}, "
          f"3 (Stream): {np.sum(final_node_codes == 3)}")

    return final_vertices_3d, np.array(mesh['triangles']), final_node_codes


# ==============================================================================
# tRIBS Mesh File Writing Functions
# ==============================================================================

def write_diagnostic_shapefile(output_file, vertices_3d, triangles, node_codes, crs=None):
    print(f"\n--- Writing diagnostic shapefile to {output_file} ---")
    polys, tri_codes = [], []
    for tri_indices in triangles:
        polys.append(Polygon(vertices_3d[tri_indices][:, :2]))
        codes_in_tri = node_codes[tri_indices]
        if 2 in codes_in_tri:
            tri_codes.append(2)
        elif 3 in codes_in_tri:
            tri_codes.append(3)
        elif 1 in codes_in_tri:
            tri_codes.append(1)
        else:
            tri_codes.append(0)
    gdf = gpd.GeoDataFrame({'code': tri_codes}, geometry=polys, crs=crs)
    gdf.to_file(output_file, driver='ESRI Shapefile')
    print("Diagnostic file written.")


def write_tRIBS_mesh_files(output_prefix, output_path, vertices, triangles, node_codes):
    print(f"\n--- Preparing to write tRIBS mesh files ---")
    # Work on a copy so we don't mutate the caller's array
    triangles = np.array(triangles, copy=True)
    nnodes, ntri = len(vertices), len(triangles)

    # Ensure triangle winding is counter-clockwise
    for i in range(ntri):
        p0, p1, p2 = triangles[i]
        area = 0.5 * (vertices[p0, 0] * (vertices[p1, 1] - vertices[p2, 1]) +
                      vertices[p1, 0] * (vertices[p2, 1] - vertices[p0, 1]) +
                      vertices[p2, 0] * (vertices[p0, 1] - vertices[p1, 1]))
        if area < 0:
            triangles[i] = [p0, p2, p1]

    undirected_edges = {tuple(sorted((tri[i], tri[(i + 1) % 3]))) for tri in triangles for i in range(3)}
    edge_list, directed_edge_to_id = [], {}
    for p1, p2 in sorted(list(undirected_edges)):
        directed_edge_to_id[(p1, p2)] = len(edge_list); edge_list.append([p1, p2])
        directed_edge_to_id[(p2, p1)] = len(edge_list); edge_list.append([p2, p1])
    nedges = len(edge_list)

    spokes = collections.defaultdict(list)
    for i, edge in enumerate(edge_list):
        spokes[edge[0]].append(i)
    node_edgid = -np.ones(nnodes, dtype=int)
    edge_nextid = -np.ones(nedges, dtype=int)

    for node_id, edge_ids in spokes.items():
        if not edge_ids:
            continue
        angles = [
            (np.arctan2(vertices[edge_list[eid][1], 1] - vertices[node_id, 1],
                        vertices[edge_list[eid][1], 0] - vertices[node_id, 0]), eid)
            for eid in edge_ids
        ]
        angles.sort()
        sorted_edge_ids = [eid for _, eid in angles]
        node_edgid[node_id] = sorted_edge_ids[0]
        for i in range(len(sorted_edge_ids)):
            edge_nextid[sorted_edge_ids[i]] = sorted_edge_ids[(i + 1) % len(sorted_edge_ids)]

    undirected_edge_to_tris = collections.defaultdict(list)
    for i, tri in enumerate(triangles):
        for j in range(3):
            p1, p2 = tri[j], tri[(j + 1) % 3]
            undirected_edge_to_tris[tuple(sorted((p1, p2)))].append(i)
    tri_neighbors = -np.ones((ntri, 3), dtype=int)

    for i, tri in enumerate(triangles):
        for j in range(3):
            p1, p2 = tri[j], tri[(j + 1) % 3]
            neighbor_tris = undirected_edge_to_tris[tuple(sorted((p1, p2)))]
            if len(neighbor_tris) == 2:
                tri_neighbors[i, j] = neighbor_tris[1] if neighbor_tris[0] == i else neighbor_tris[0]

    with open(f"{output_path}{output_prefix}.z", "w") as f:
        f.write("0.000000\n"); f.write(f"{nnodes}\n")
        np.savetxt(f, vertices[:, 2], fmt='%.6f')
    with open(f"{output_path}{output_prefix}.nodes", "w") as f:
        f.write("0.000000\n"); f.write(f"{nnodes}\n")
        for i in range(nnodes):
            if node_edgid[i] == -1:
                raise RuntimeError(f"Node {i} is isolated.")
            f.write(f"{vertices[i, 0]:.6f} {vertices[i, 1]:.6f} {node_edgid[i]} {node_codes[i]}\n")
    with open(f"{output_path}{output_prefix}.edges", "w") as f:
        f.write("0.000000\n"); f.write(f"{nedges}\n")
        for i in range(nedges):
            f.write(f"{edge_list[i][0]} {edge_list[i][1]} {edge_nextid[i]}\n")
    with open(f"{output_path}{output_prefix}.tri", "w") as f:
        f.write("0.000000\n"); f.write(f"{ntri}\n")
        for i in range(ntri):
            p0, p1, p2 = triangles[i]
            n0, n1, n2 = tri_neighbors[i, 1], tri_neighbors[i, 2], tri_neighbors[i, 0]
            e0, e1, e2 = directed_edge_to_id[(p0, p2)], directed_edge_to_id[(p1, p0)], directed_edge_to_id[(p2, p1)]
            f.write(f"{p0} {p1} {p2} {n0} {n1} {n2} {e0} {e1} {e2}\n")
    print("\n--- All tRIBS mesh files have been generated successfully. ---")


# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == "__main__":
    # --- FILE PATHS ---
    name = "CloverCanyon"
    output_path = f'data/model/mesh/'
    points_file = f'data/model/mesh/{name}.points'          # Interior terrain points from pytRIBS wavelet transform
    stream_shapefile = f'data/preprocessing/{name}_stream.shp'
    watershed_shapefile = f'data/preprocessing/{name}_watershed.shp'
    dem_file = "../../GIS/data/raster/USGS_1m_2019.tif"
    tree_file = "../../GIS/data/shp/Tree_Points/TreePoints_CC.shp"  # Set to None to disable

    print("--- Generating tRIBS mesh ---")

    # --- KEY PARAMETERS TO TUNE ---
    boundary_buffer = 50.0        # meters. Buffer distance outward from watershed polygon.
                                  # Must be large enough that boundary nodes are clearly outside
                                  # the watershed. 50m is a reasonable starting point.
    boundary_spacing = 50.0       # meters. Distance between boundary nodes along the buffered polygon.
    stream_spacing = 20.0         # meters. Distance between generated stream nodes.
    stream_clear_radius = 10.0    # meters. Removes interior/tree points this close to streams.
    tree_cull_radius = 5.0        # meters. Removes original interior points this close to trees.
    quality_opts = 'q10a15000'    # Triangle quality options.
        # q: Min angle in degrees. Adds Steiner points to fix triangles below this angle.
        # a: Max triangle area in map units (m² if CRS is projected). Adds Steiner points
        #    to break up large triangles. Be careful — too small causes very large meshes.
        # D: Conforming Delaunay (always on).
        # YY: Prevent Steiner points on input segments. Use if tRIBS has stream connectivity
        #     issues, but note it relaxes quality guarantees.
        # https://www.cs.cmu.edu/~quake/triangle.switch.html

    # Set working directory to script location so relative paths work
    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)
    print(f"Working directory: {os.getcwd()}")

    try:
        final_vertices, final_triangles, final_node_codes = generate_mesh_from_points(
            points_file=points_file,
            stream_shapefile=stream_shapefile,
            dem_file=dem_file,
            watershed_shapefile=watershed_shapefile,
            boundary_buffer=boundary_buffer,
            boundary_spacing=boundary_spacing,
            tree_centroids_shapefile=tree_file,
            tree_remove_radius=tree_cull_radius,
            stream_clearance_radius=stream_clear_radius,
            stream_point_spacing=stream_spacing,
            mesh_quality_opts=quality_opts
        )

        write_diagnostic_shapefile(
            output_file=f'data/preprocessing/{name}_tin_{quality_opts}.shp',
            vertices_3d=final_vertices,
            triangles=final_triangles,
            node_codes=final_node_codes,
            crs="EPSG:26912"
        )

        write_tRIBS_mesh_files(f'{name}_mesh', output_path, final_vertices, final_triangles, final_node_codes)

    except Exception as e:
        import traceback
        print(f"\nPROCESS FAILED: {e}")
        traceback.print_exc()
