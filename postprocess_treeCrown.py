import os
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.ops import unary_union
from tqdm import tqdm  # Import tqdm for progress tracking

# CRS used in your workflow
TARGET_CRS = 25832
DENSITY_LAYER = "tree_crown_min10_density"
FINAL_LAYER = "tree_crown_merged_final"


def safe_remove(path):
    if os.path.exists(path):
        try:
            os.remove(path)
        except PermissionError:
            raise PermissionError(f"Cannot remove {path}. Close any program using it (e.g., QGIS).")


# Function to check and remove small polygons based on overlap threshold
def remove_small_polygons_based_on_overlap(gdf, overlap_threshold=0.3):
    """
    This function checks for overlaps between polygons in different density classes
    (low, medium, and high). If a small polygon overlaps by more than overlap_threshold,
    it will be removed.
    """
    # Initialize spatial index for efficient search
    sindex = gdf.sindex

    # Create a dictionary to store results
    overlap_results = {'low': [], 'medium': [], 'high': []}

    # Separate the polygons into low, medium, and high density classes
    low_density_polygons = gdf[gdf['density_class'] == 'low']
    medium_density_polygons = gdf[gdf['density_class'] == 'medium']
    high_density_polygons = gdf[gdf['density_class'] == 'high']

    def check_overlaps(source_gdf, target_gdf, source_label, target_label):
        for idx, source_geom in source_gdf.iterrows():

            source_poly = source_geom['geometry']

            # 🔴 FIX 1: clean invalid geometry
            if not source_poly.is_valid:
                source_poly = source_poly.buffer(0)

            # Find candidate polygons
            cand_idx = list(sindex.intersection(source_poly.bounds))
            if not cand_idx:
                continue

            cand_idx = [i for i in cand_idx if i in target_gdf.index]
            if not cand_idx:
                continue

            cands = target_gdf.loc[cand_idx]

            for id_, candidate in cands.iterrows():
                candidate_poly = candidate['geometry']

                # 🔴 FIX 2: clean invalid geometry
                if not candidate_poly.is_valid:
                    candidate_poly = candidate_poly.buffer(0)

                try:
                    if source_poly.intersects(candidate_poly):

                        # 🔴 FIX 3: safe intersection
                        intersection = source_poly.intersection(candidate_poly)

                        # 🔴 FIX 4: correct overlap logic
                        ratio_source = intersection.area / source_poly.area
                        ratio_target = intersection.area / candidate_poly.area

                        overlap_ratio = max(ratio_source, ratio_target)

                        if overlap_ratio >= overlap_threshold:

                            # 🔴 FIX 5: remove SMALLER polygon
                            if source_poly.area < candidate_poly.area:
                                remove_id = source_geom['id']
                            else:
                                remove_id = candidate['id']

                            overlap_results[source_label].append((remove_id, overlap_ratio))

                except Exception as e:
                    print(f"Error processing intersection: {e}")

                try:
                    if source_poly.intersects(candidate_poly):
                        intersection = source_poly.intersection(candidate_poly)

                        # --- NEW: symmetric overlap ---
                        ratio_source = intersection.area / source_poly.area
                        ratio_target = intersection.area / candidate_poly.area # new

                        overlap_ratio = max(ratio_source, ratio_target)

                        # --- decide which polygon to remove ---
                        if overlap_ratio >= overlap_threshold:

                            if source_poly.area < candidate_poly.area:
                                remove_id = source_geom['id']
                            else:
                                remove_id = candidate['id']

                            overlap_results[source_label].append((remove_id, overlap_ratio))

                except Exception as e:
                    print(f"Error processing intersection: {e}")

    # Check overlaps between low and medium polygons
    check_overlaps(low_density_polygons, medium_density_polygons, 'low', 'medium')
    # Check overlaps between low and high polygons
    check_overlaps(low_density_polygons, high_density_polygons, 'low', 'high')

    # Check overlaps between medium and high polygons
    check_overlaps(medium_density_polygons, high_density_polygons, 'medium', 'high')

    # Create a list to store polygons to remove
    polygons_to_remove = []

    # Iterate over the overlap results and add polygons to the removal list
    for label in ['low', 'medium', 'high']:
        for overlap in overlap_results[label]:
            remove_id, overlap_ratio = overlap
            polygons_to_remove.append(remove_id)

    # Remove polygons in the removal list from the GeoDataFrame
    gdf_filtered = gdf[~gdf['id'].isin(polygons_to_remove)]

    return gdf_filtered


def stage2_compute_density(poly_out_5, poly_out_10, poly_out_15,
                           density_out,
                           cell_size=20,
                           use_median=False):

    print("\n=== STAGE 2: Density Mask Creation (AVERAGE density) ===")

    g5 = gpd.read_file(poly_out_5).to_crs(TARGET_CRS)
    g10 = gpd.read_file(poly_out_10).to_crs(TARGET_CRS)
    g15 = gpd.read_file(poly_out_15).to_crs(TARGET_CRS)

    print("Loaded min5:", len(g5))
    print("Loaded min10:", len(g10))
    print("Loaded min15:", len(g15))

    density_tables = []

    for g in [g5, g10, g15]:
        cent = g.geometry.centroid
        cx = cent.x.to_numpy()
        cy = cent.y.to_numpy()

        gx = np.floor_divide(cx.astype(np.int64), cell_size).astype(np.int64)
        gy = np.floor_divide(cy.astype(np.int64), cell_size).astype(np.int64)

        tmp = pd.DataFrame({"gx": gx, "gy": gy})
        cell_counts = tmp.groupby(["gx", "gy"]).size()

        density_tables.append(cell_counts)

    density_df = pd.concat(density_tables, axis=1).fillna(0)

    if use_median:
        density_df["density_avg"] = density_df.median(axis=1)
        print("Using MEDIAN density")
    else:
        density_df["density_avg"] = density_df.mean(axis=1)
        print("Using MEAN density")

    base = g5.copy()

    cent = base.geometry.centroid
    cx = cent.x.to_numpy()
    cy = cent.y.to_numpy()

    gx = np.floor_divide(cx.astype(np.int64), cell_size).astype(np.int64)
    gy = np.floor_divide(cy.astype(np.int64), cell_size).astype(np.int64)

    tmp = pd.DataFrame({"gx": gx, "gy": gy})

    base["density"] = tmp.join(
        density_df["density_avg"],
        on=["gx", "gy"]
    )["density_avg"].to_numpy()

    # Calculate percentiles for classification
    low_th = np.percentile(base["density"], 44)  # 33rd percentile
    med_th = np.percentile(base["density"], 95)  # 66th percentile

    print("\nDensity statistics:")
    print(base["density"].describe())

    print(f"\nThresholds (percentiles):")
    print(f"low < {low_th}")
    print(f"medium < {med_th}")
    print(f"high >= {med_th}")

    # Use percentiles for density classification
    base["density_class"] = np.where(
        base["density"] < low_th, "low",
        np.where(base["density"] < med_th, "medium", "high")
    )

    safe_remove(density_out)

    base.to_file(
        density_out,
        driver="SQLite",
        layer=DENSITY_LAYER
    )

    print("Saved density mask →", density_out)


def has_tile_seam(poly, max_straight=3, axis_tol=0.05):
    """
    Detect long straight edges typical for tile seams.
    Works for Polygon and MultiPolygon.
    """

    # handle MultiPolygon
    if poly.geom_type == "MultiPolygon":
        geoms = poly.geoms
    else:
        geoms = [poly]

    for g in geoms:
        coords = list(g.exterior.coords)

        for i in range(len(coords) - 1):
            x1, y1 = coords[i]
            x2, y2 = coords[i + 1]

            length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

            if length > max_straight:
                if abs(x1 - x2) < axis_tol or abs(y1 - y2) < axis_tol:
                    return True

    return False


def replace_seam_with_g10(final_gdf, g10_gdf, sindex, max_straight=3, axis_tol=0.05, overlap_threshold=0.3):
    """
    Detect polygons with tile seams in final_gdf and replace them with the best matching polygons from g10_gdf.
    :param final_gdf: GeoDataFrame containing the final polygons.
    :param g10_gdf: GeoDataFrame containing the polygons from g10.
    :param sindex: Spatial index of g10_gdf for fast lookup.
    :param max_straight: Maximum length to detect a straight edge typical for tile seams.
    :param axis_tol: Tolerance to detect horizontal or vertical straight edges typical for tile seams.
    :param overlap_threshold: Minimum overlap ratio for replacement.
    :return: Updated GeoDataFrame with replaced polygons.
    """

    def has_tile_seam(poly, max_straight=max_straight, axis_tol=axis_tol):
        """Detect long straight edges typical for tile seams."""
        if poly is None or poly.is_empty:
            return False

        if poly.geom_type == "MultiPolygon":
            geoms = poly.geoms
        else:
            geoms = [poly]

        for g in geoms:
            if g.is_empty:
                continue

            coords = list(g.exterior.coords)

            for i in range(len(coords) - 1):
                x1, y1 = coords[i]
                x2, y2 = coords[i + 1]

                length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

                if length > max_straight:
                    if abs(x1 - x2) < axis_tol or abs(y1 - y2) < axis_tol:
                        return True

        return False

    def get_best_overlap_geom(geom, polygons_gdf, sindex):
        """Find the polygon in g10 with the maximum intersection area."""
        cand_idx = list(sindex.intersection(geom.bounds))
        if not cand_idx:
            return None, None

        cands = polygons_gdf.iloc[cand_idx]
        inter = cands.intersection(geom)
        areas = inter.area

        if len(areas) == 0 or areas.max() <= 0:
            return None, None

        best_pos = areas.values.argmax()
        best_row = cands.iloc[best_pos]
        return best_row["id"], best_row.geometry

    replaced_geoms = []

    for _, row in tqdm(final_gdf.iterrows(), total=len(final_gdf)):
        geom = row.geometry

        # If no seam is detected, keep the original geometry
        if not has_tile_seam(geom):
            replaced_geoms.append(geom)
            continue

        # Find the best matching polygon from g10
        g10_id, g10_geom = get_best_overlap_geom(geom, g10_gdf, sindex)

        if g10_geom is None:
            replaced_geoms.append(geom)
            continue

        # Replace the geometry with the best matching g10 geometry if a seam is detected
        replaced_geoms.append(g10_geom)

    # Create a new GeoDataFrame with replaced geometries
    final_replaced_gdf = final_gdf.copy()
    final_replaced_gdf["geometry"] = replaced_geoms

    return final_replaced_gdf


def get_best_overlap_geom(geom, polygons_gdf, sindex):
    """Find the polygon in g10 with the maximum intersection area."""
    cand_idx = list(sindex.intersection(geom.bounds))
    if not cand_idx:
        return None, None

    cands = polygons_gdf.iloc[cand_idx]
    inter = cands.intersection(geom)
    areas = inter.area

    if len(areas) == 0 or areas.max() <= 0:
        return None, None

    best_pos = areas.values.argmax()
    best_row = cands.iloc[best_pos]
    return best_row["id"], best_row.geometry


# -------------------------------------------------------------------
# Helper: replace geometry by best-overlap candidate
# -------------------------------------------------------------------
def replace_by_overlap(mask_gdf, polygons_gdf, label, batch_size=200_000, overlap_threshold=0.30):
    """
    For each polygon in mask_gdf, pick the polygon in polygons_gdf that has
    the maximum intersection area with it. If nothing overlaps, keep the original polygon.
    After replacement, if the replaced polygon intersects with other small polygons
    by more than overlap_threshold of the smaller polygon's area, remove the smaller polygon.

    batch_size controls memory (important for millions of features).
    """
    print(f"\n{label}: overlap-based geometry replacement (batched)")

    polygons_gdf = polygons_gdf[["geometry"]].copy()
    sindex = polygons_gdf.sindex

    out_parts = []
    n = len(mask_gdf)

    # Using tqdm to track progress
    for start in tqdm(range(0, n, batch_size), desc="Processing Batches", unit="batch"):
        end = min(start + batch_size, n)
        part = mask_gdf.iloc[start:end].copy()

        new_geom = []
        for geom in part.geometry.values:
            cand_idx = list(sindex.intersection(geom.bounds))
            if not cand_idx:
                new_geom.append(geom)
                continue

            cands = polygons_gdf.iloc[cand_idx]
            inter = cands.intersection(geom)

            areas = inter.area
            if areas.max() <= 0:
                new_geom.append(geom)
            else:
                best_pos = areas.values.argmax()
                best_geom = cands.iloc[best_pos].geometry

                # Keep the current replacement logic
                new_geom.append(best_geom)

                # Check for tile seam artifact
                if has_tile_seam(best_geom):
                    new_geom[-1] = geom  # fallback to min5 if tile seam detected

        part.geometry = new_geom
        out_parts.append(part)

        print(f"  processed {end:,}/{n:,}")

    return gpd.GeoDataFrame(pd.concat(out_parts, ignore_index=True), crs=mask_gdf.crs)


def stage3_merge_three_levels(density_out, poly_out_10, poly_out_15, final_out):
    print("\n=== STAGE 3: Three-level merge (min5 + min10 + min15) ===")

    mask = gpd.read_file(density_out, layer=DENSITY_LAYER).to_crs(TARGET_CRS)

    high = mask[mask["density_class"] == "high"].copy()
    medium = mask[mask["density_class"] == "medium"].copy()
    low = mask[mask["density_class"] == "low"].copy()

    g10 = gpd.read_file(poly_out_10).to_crs(TARGET_CRS)
    g15 = gpd.read_file(poly_out_15).to_crs(TARGET_CRS)

    print("High crowns (min5):", len(high))
    print("Medium crowns (min10 candidates):", len(medium))
    print("Low crowns (min15 candidates):", len(low))

    # --------------------------------------------------
    # Replace geometry
    # --------------------------------------------------

    medium_final = replace_by_overlap(medium, g10, "MEDIUM → min10")
    low_final = replace_by_overlap(low, g15, "LOW → min15")

    # --------------------------------------------------
    # Simplify
    # --------------------------------------------------

    medium_final["geometry"] = medium_final.geometry.simplify(
        tolerance=0.2,
        preserve_topology=True
    )

    low_final["geometry"] = low_final.geometry.simplify(
        tolerance=0.2,
        preserve_topology=True
    )

    high["geometry"] = high.geometry.simplify(
        tolerance=0.2,
        preserve_topology=True
    )

    # --------------------------------------------------
    # Merge result
    # --------------------------------------------------

    final = gpd.GeoDataFrame(
        pd.concat([high, medium_final, low_final], ignore_index=True),
        crs=TARGET_CRS
    )

    print("Loading g10...")
    g10_gdf = g10[["id", "geometry"]].copy().reset_index(drop=True)
    g10_sindex = g10_gdf.sindex
    final_gdf = replace_seam_with_g10(final, g10_gdf, g10_sindex)

    # Remove small polygons based on overlap
    final_cleaned = remove_small_polygons_based_on_overlap(final_gdf, overlap_threshold=0.2)


    # Remove duplicates based on the geometry
    final_cleaned_ = final_cleaned.drop_duplicates(subset='geometry')

    safe_remove(final_out)

    final_cleaned_.to_file(
        final_out,
        driver="SQLite",
        layer=FINAL_LAYER
    )

    print("Saved final merged →", final_out)


if __name__ == "__main__":
    # Paths to input and output data
    tile_dirs = [
        r"D:\DeepTree",  # Modify this with your correct directory
    ]

    for tile_dir in tile_dirs:
        poly_out_5 = os.path.join(tile_dir, "Frankfurt_2021_polygon_min7")
        poly_out_10 = os.path.join(tile_dir, "Frankfurt_2021_polygon_min15")
        poly_out_15 = os.path.join(tile_dir, "D:\DeepTree\Frankfurt_2021_polygon_min17___________.sqlite")
        density_out = os.path.join(tile_dir, "min5_with_density_final_1_buffer20_44_95_02_15_17.sqlite")
        final_out = os.path.join(tile_dir, "tree_crown_merged_final_1_buffer20_44_95_02_15_17.sqlite")

        print("\nProcessing directory:", tile_dir)

        stage2_compute_density(
            poly_out_5,
            poly_out_10,
            poly_out_15,
            density_out,
            cell_size=20
        )
        stage3_merge_three_levels(
            density_out,
            poly_out_10,
            poly_out_15,
            final_out
        )

    print("\n🎉 DONE")
