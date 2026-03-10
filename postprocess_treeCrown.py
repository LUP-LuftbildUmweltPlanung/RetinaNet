
import os
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.ops import unary_union
from tqdm import tqdm  # Import tqdm for progress tracking


# CRS used in your workflow
TARGET_CRS = 25832
DENSITY_LAYER = "tree_crown_min5_density"
FINAL_LAYER = "tree_crown_merged_final"


def safe_remove(path):
    if os.path.exists(path):
        try:
            os.remove(path)
        except PermissionError:
            raise PermissionError(f"Cannot remove {path}. Close any program using it (e.g., QGIS).")


# -------------------------------------------------------------------
# STAGE 2 (FAST): density proxy using a grid (no KDTree, no O(n^2))
# -------------------------------------------------------------------
def stage2_compute_density(poly_out_5, density_out, cell_size=50):
    """
    cell_size in meters. 50m is similar to your original r=50 neighborhood idea,
    but computed in O(n) using grid counts.
    """
    print("\n=== STAGE 2: Density Mask Creation (FAST grid density) ===")

    g = gpd.read_file(poly_out_5).to_crs(TARGET_CRS).reset_index(drop=True)
    print("Loaded min5 polygons:", len(g))

    cent = g.geometry.centroid
    cx = cent.x.to_numpy()
    cy = cent.y.to_numpy()

    # grid cell indices
    gx = np.floor_divide(cx.astype(np.int64), cell_size).astype(np.int64)
    gy = np.floor_divide(cy.astype(np.int64), cell_size).astype(np.int64)

    tmp = pd.DataFrame({"gx": gx, "gy": gy})
    cell_counts = tmp.groupby(["gx", "gy"]).size().rename("density")

    g["density"] = tmp.join(cell_counts, on=["gx", "gy"])["density"].to_numpy()

    low_th = float(pd.Series(g["density"]).quantile(0.33))
    med_th = float(pd.Series(g["density"]).quantile(0.55))

    g["density_class"] = np.where(
        g["density"] < low_th, "low",
        np.where(g["density"] < med_th, "medium", "high")
    )

    safe_remove(density_out)
    g.to_file(density_out, driver="SQLite", layer=DENSITY_LAYER)
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

        for i in range(len(coords)-1):
            x1, y1 = coords[i]
            x2, y2 = coords[i+1]

            length = ((x2-x1)**2 + (y2-y1)**2)**0.5

            if length > max_straight:
                if abs(x1-x2) < axis_tol or abs(y1-y2) < axis_tol:
                    return True

    return False
# -------------------------------------------------------------------
# Helper: replace geometry by best-overlap candidate
# -------------------------------------------------------------------
def replace_by_overlap(mask_gdf, polygons_gdf, label, batch_size=200_000):
    """
    For each polygon in mask_gdf, pick the polygon in polygons_gdf that has
    the maximum intersection area with it. If nothing overlaps, keep original.

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

                # check for tile seam artifact
                if has_tile_seam(best_geom):
                    new_geom.append(geom)  # fallback to min5
                else:
                    new_geom.append(best_geom)

        part.geometry = new_geom
        out_parts.append(part)

        print(f"  processed {end:,}/{n:,}")

    return gpd.GeoDataFrame(pd.concat(out_parts, ignore_index=True), crs=mask_gdf.crs)

from shapely.ops import unary_union

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
    # Remove overlaps with HIGH crowns
    # --------------------------------------------------

    print("\nRemoving overlaps with HIGH crowns...")

    sindex_high = high.sindex
    new_geoms = []

    for i, geom in enumerate(medium_final.geometry):

        if i % 50000 == 0:
            print(f" processed {i:,}/{len(medium_final):,}")

        candidates = list(sindex_high.intersection(geom.bounds))

        if not candidates:
            new_geoms.append(geom)
            continue

        local_union = unary_union(high.iloc[candidates].geometry)

        new_geom = geom.difference(local_union)

        if not new_geom.is_empty:
            new_geoms.append(new_geom)

    medium_final["geometry"] = new_geoms
    medium_final = medium_final.explode(index_parts=False).reset_index(drop=True)
    medium_final = medium_final[medium_final.geometry.area > 1]

    # --------------------------------------------------
    # Remove overlaps with HIGH + MEDIUM crowns
    # --------------------------------------------------

    print("\nRemoving overlaps with HIGH + MEDIUM crowns...")

    priority_union = unary_union(
        list(high.geometry) + list(medium_final.geometry)
    )

    new_geoms = []

    for i, geom in enumerate(low_final.geometry):

        if i % 50000 == 0:
            print(f" processed {i:,}/{len(low_final):,}")

        new_geom = geom.difference(priority_union)

        if not new_geom.is_empty:
            new_geoms.append(new_geom)

    low_final["geometry"] = new_geoms
    low_final = low_final.explode(index_parts=False).reset_index(drop=True)
    low_final = low_final[low_final.geometry.area > 1]

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

    safe_remove(final_out)

    final.to_file(
        final_out,
        driver="SQLite",
        layer=FINAL_LAYER
    )

    print("Saved final merged →", final_out)

    print("Saved final merged →", final_out)
if __name__ == "__main__":
    # Paths to input and output data
    tile_dirs = [
        r"D:\DeepTree",  # Modify this with your correct directory
    ]

    for tile_dir in tile_dirs:
        poly_out_5 = os.path.join(tile_dir, "Frankfurt_2021_polygon_min5")
        poly_out_10 = os.path.join(tile_dir, "Frankfurt_2021_polygon_min10_2")
        poly_out_15 = os.path.join(tile_dir, "polygons_min15")
        density_out = os.path.join(tile_dir, "min5_with_density_new_param_3.sqlite")
        final_out = os.path.join(tile_dir, "tree_crown_merged_final_new_param_3.sqlite")

        print("\nProcessing directory:", tile_dir)

        # Stage 2: Compute density (fast grid-based)
        stage2_compute_density(poly_out_5, density_out, cell_size=50)

        # Stage 3: Merge based on density classes
        stage3_merge_three_levels(density_out, poly_out_10, poly_out_15, final_out)

    print("\n🎉 DONE")

