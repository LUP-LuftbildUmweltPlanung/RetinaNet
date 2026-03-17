

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

    :param gdf: The GeoDataFrame containing the polygons (with 'density_class' and 'geometry' columns)
    :param overlap_threshold: The threshold for significant overlap (default is 30%)
    :return: Updated GeoDataFrame with small polygons removed
    """
    # Initialize spatial index for efficient search
    sindex = gdf.sindex

    # Create a dictionary to store results
    overlap_results = {'low': [], 'medium': [], 'high': []}

    # Separate the polygons into low, medium, and high density classes
    low_density_polygons = gdf[gdf['density_class'] == 'low']
    medium_density_polygons = gdf[gdf['density_class'] == 'medium']
    high_density_polygons = gdf[gdf['density_class'] == 'high']

    # Function to check and store overlaps
    def check_overlaps(source_gdf, target_gdf, source_label, target_label):
        for idx, source_geom in source_gdf.iterrows():
            source_poly = source_geom['geometry']

            # Find candidate polygons that intersect with the source polygon
            cand_idx = list(sindex.intersection(source_poly.bounds))
            if not cand_idx:
                continue  # No candidates, skip this polygon

            # Ensure that cand_idx contains valid indices for target_gdf
            cand_idx = [i for i in cand_idx if i in target_gdf.index]
            if not cand_idx:
                continue  # Skip if no valid candidates in target_gdf

            # Get the intersecting polygons from the target group
            cands = target_gdf.loc[cand_idx]

            for id_, candidate in cands.iterrows():
                candidate_poly = candidate['geometry']

                # Check if the source polygon intersects with the candidate polygon
                if source_poly.intersects(candidate_poly):
                    intersection = source_poly.intersection(candidate_poly)
                    overlap_ratio = intersection.area / source_poly.area

                    # If overlap is greater than the threshold, consider it for removal
                    if overlap_ratio >= overlap_threshold:  # Threshold of 30% overlap
                        overlap_results[source_label].append((source_geom['id'], candidate['id'], overlap_ratio))

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
            source_id, target_id, overlap_ratio = overlap
            polygons_to_remove.append(target_id)  # Mark the smaller polygon for removal

    # Remove polygons in the removal list from the GeoDataFrame
    gdf_filtered = gdf[~gdf['id'].isin(polygons_to_remove)]

    return gdf_filtered

def stage2_compute_density(poly_out_5, poly_out_10, poly_out_15,
                           density_out,
                           cell_size=20,
                           low_th=40,
                           med_th=140,
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
        on=["gx","gy"]
    )["density_avg"].to_numpy()

    print("\nDensity statistics:")
    print(base["density"].describe())

    print(f"\nThresholds:")
    print("low <", low_th)
    print("medium <", med_th)
    print("high >=", med_th)

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

    # Remove small polygons based on overlap
    final_cleaned = remove_small_polygons_based_on_overlap(final, overlap_threshold=0.01)
    #check_and_remove_overlap

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
        poly_out_15 = os.path.join(tile_dir, "Frankfurt_2021_polygon_min20")
        density_out = os.path.join(tile_dir, "min5_with_density_final_1.sqlite")
        final_out = os.path.join(tile_dir, "tree_crown_merged_final_1.sqlite")

        print("\nProcessing directory:", tile_dir)

        # # Stage 2: Compute density (fast grid-based)
        # stage2_compute_density(poly_out_5, density_out, cell_size=50)
        #
        stage2_compute_density(
            poly_out_5,
            poly_out_10,
            poly_out_15,
            density_out,
            cell_size=50
        )
        # Stage 3: Merge based on density classes
        # stage3_merge_three_levels(density_out, poly_out_10, poly_out_15, final_out)
        stage3_merge_three_levels(
            density_out,
            poly_out_5,
            poly_out_15,
            final_out
        )

    print("\n🎉 DONE")
