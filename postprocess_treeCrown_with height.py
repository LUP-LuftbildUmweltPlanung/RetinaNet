import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize
from tqdm import tqdm


# ============================================================
# SETTINGS
# ============================================================
TARGET_CRS = "EPSG:25832"
CROWNS_LAYER_IN = "tree_crown_merged_final"   # crowns input layer (sqlite/gpkg)
CROWNS_LAYER_OUT = "crowns_final"            # crowns output layer in gpkg
HEIGHT_ATTR = "height"


# ============================================================
# UTILS
# ============================================================
def safe_remove(path):
    if os.path.exists(path):
        try:
            os.remove(path)
        except PermissionError:
            raise PermissionError(f"Cannot remove {path}. Close apps using it (QGIS).")


def fix_local_cs(crs):
    if crs is None:
        return TARGET_CRS
    s = str(crs)
    if "LOCAL_CS" in s:
        return TARGET_CRS
    return crs


def estimate_bytes(h, w, dtype_bytes):
    return int(h) * int(w) * int(dtype_bytes)


def human_gb(nbytes):
    return nbytes / (1024**3)


def layer_exists_vector(path, layer):
    """
    Works for GPKG and SQLite via Fiona/GeoPandas.
    Returns True if layer exists.
    """
    try:
        import fiona
        layers = fiona.listlayers(path)
        return layer in layers
    except Exception:
        return False


# ============================================================
# STEP 0: CREATE nDOM = BDOM - DGM (optional)
# ============================================================
def create_ndom(bdom_path, dgm_path, ndom_out):
    print(">>> Creating nDOM (BDOM - DGM)...")

    with rasterio.open(bdom_path) as bsrc:
        profile = bsrc.profile
        bdom = bsrc.read(1)
        dst_transform = bsrc.transform
        dst_crs = fix_local_cs(bsrc.crs)

    with rasterio.open(dgm_path) as dsrc:
        dgm = dsrc.read(1)
        dgm_resampled = np.zeros_like(bdom, dtype=np.float32)

        reproject(
            source=dgm,
            destination=dgm_resampled,
            src_transform=dsrc.transform,
            src_crs=fix_local_cs(dsrc.crs),
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
        )

    ndom = bdom.astype(np.float32) - dgm_resampled
    ndom[~np.isfinite(ndom)] = np.nan

    profile.update(dtype="float32", count=1, compress="deflate", nodata=np.nan, crs=dst_crs)

    safe_remove(ndom_out)
    with rasterio.open(ndom_out, "w", **profile) as dst:
        dst.write(ndom, 1)

    print(">>> nDOM saved:", ndom_out)
    return ndom_out


# ============================================================
# STEP 1: READ CROWNS + CRS ALIGN
# ============================================================
def read_crowns(crowns_path, raster_path, layer=CROWNS_LAYER_IN):
    print("\n>>> Loading crown polygons…")
    gdf = gpd.read_file(crowns_path, layer=layer)

    with rasterio.open(raster_path) as src:
        raster_crs = fix_local_cs(src.crs)

    if str(gdf.crs) != str(raster_crs):
        print(f"  Reprojecting crowns {gdf.crs} → {raster_crs}")
        gdf = gdf.to_crs(raster_crs)
    else:
        print("  CRS already aligned — no reprojection.")

    gdf = gdf.reset_index(drop=True)
    if "pid" not in gdf.columns:
        gdf["pid"] = np.arange(1, len(gdf) + 1, dtype=np.int32)

    print("  Crowns loaded:", len(gdf))
    return gdf, raster_crs


# ============================================================
# MODE B: TILE-BASED HISTOGRAM PERCENTILE (scales to huge rasters)
# ============================================================
def make_hist_bins(vmin=0.0, vmax=60.0, step=0.25):
    return np.arange(vmin, vmax + step, step, dtype=np.float32)


def zonal_percentile_tiled_hist(
    ndom_path,
    crowns_gdf,
    percentile=80,
    chunk_size=512,
    vmin=0.0,
    vmax=60.0,
    step=0.25,
):
    print(f"\n>>> Computing {percentile}th percentile height (TILED histogram)…")
    print(f"    chunk_size = {chunk_size}, bins = {vmin}..{vmax} step {step}")

    bins = make_hist_bins(vmin, vmax, step)
    nbins = len(bins) - 1

    crowns_gdf = crowns_gdf.reset_index(drop=True)
    max_pid = int(crowns_gdf["pid"].max())

    sindex = crowns_gdf.sindex
    hists = {}

    with rasterio.open(ndom_path) as src:
        H, W = src.height, src.width
        transform = src.transform
        nodata = src.nodata

        for r0 in tqdm(range(0, H, chunk_size), desc="Raster rows"):
            for c0 in range(0, W, chunk_size):

                win = Window(
                    col_off=c0,
                    row_off=r0,
                    width=min(chunk_size, W - c0),
                    height=min(chunk_size, H - r0),
                )

                minx, miny, maxx, maxy = rasterio.windows.bounds(win, transform)
                idxs = list(sindex.intersection((minx, miny, maxx, maxy)))
                if not idxs:
                    continue

                gsub = crowns_gdf.iloc[idxs][["geometry", "pid"]]

                arr = src.read(1, window=win).astype(np.float32)
                if nodata is not None:
                    arr[arr == nodata] = np.nan

                win_transform = src.window_transform(win)

                pid_r = rasterize(
                    shapes=[(geom, int(pid)) for geom, pid in zip(gsub.geometry, gsub.pid) if geom is not None],
                    out_shape=arr.shape,
                    transform=win_transform,
                    fill=0,
                    dtype="int32",
                    all_touched=False,
                )

                m = (pid_r > 0) & np.isfinite(arr)
                if not np.any(m):
                    continue

                pid_vals = pid_r[m]
                h_vals = arr[m]

                b = np.searchsorted(bins, h_vals, side="right") - 1
                valid = (b >= 0) & (b < nbins)
                if not np.any(valid):
                    continue

                pid_vals = pid_vals[valid]
                b = b[valid]

                order = np.argsort(pid_vals)
                pid_sorted = pid_vals[order]
                b_sorted = b[order]

                cuts = np.where(pid_sorted[1:] != pid_sorted[:-1])[0] + 1
                pid_groups = np.split(pid_sorted, cuts)
                b_groups = np.split(b_sorted, cuts)

                for pg, bg in zip(pid_groups, b_groups):
                    pid = int(pg[0])
                    if pid not in hists:
                        hists[pid] = np.zeros(nbins, dtype=np.int32)
                    hists[pid] += np.bincount(bg, minlength=nbins).astype(np.int32)

    heights = np.full(max_pid + 1, np.nan, dtype=np.float32)
    target_q = percentile / 100.0

    for pid, hist in tqdm(hists.items(), desc="Percentiles"):
        cdf = np.cumsum(hist, dtype=np.int64)
        total = cdf[-1]
        if total <= 0:
            continue
        k = target_q * total
        idx = int(np.searchsorted(cdf, k))
        idx = max(0, min(idx, nbins - 1))
        heights[pid] = float(bins[idx])

    return heights


# ============================================================
# STEP 4: SAVE CLEAN CROWNS (skip if already exists)
# ============================================================
def save_clean_crowns(gdf, out_gpkg, height_clean=0.1):
    print("\n>>> Cleaning crowns…")
    before = len(gdf)
    gdf_clean = gdf[np.isfinite(gdf[HEIGHT_ATTR]) & (gdf[HEIGHT_ATTR] > height_clean)].copy()
    after = len(gdf_clean)
    print("  Crowns before filter:", before)
    print("  Crowns after  filter:", after)

    safe_remove(out_gpkg)
    gdf_clean.to_file(out_gpkg, driver="GPKG", layer=CROWNS_LAYER_OUT)
    print(">>> Crowns saved:", out_gpkg)

    return gdf_clean


# ============================================================
# STEP 5: RASTERIZE HEIGHT (SAFE FOR HUGE RASTERS)
# -> write output in tiles/windows, never allocate full raster array
# ============================================================
def rasterize_crowns_height_tiled(
    gpkg_path,
    raster_template,
    output_raster,
    layer=CROWNS_LAYER_OUT,
    attribute=HEIGHT_ATTR,
    chunk_size=1024,
    dtype="float32",
    all_touched=False,
):
    print("\n>>> Rasterizing crown height (TILED, PROJ-safe)…")

    gdf = gpd.read_file(gpkg_path, layer=layer)

    with rasterio.open(raster_template) as src:
        profile = src.profile.copy()
        transform = src.transform
        H, W = src.height, src.width
        raster_crs = src.crs            # ← copy CRS object directly
        nodata = np.nan

    # Reproject only if absolutely necessary (string compare)
    if str(gdf.crs) != str(raster_crs):
        print("CRS mismatch detected but PROJ is broken. Forcing crowns CRS to raster CRS.")
        gdf = gdf.set_crs(raster_crs, allow_override=True)

    # spatial index for chunk filtering
    sindex = gdf.sindex

    # IMPORTANT: do NOT touch CRS in profile — keep template CRS
    profile.update(
        dtype=dtype,
        count=1,
        compress="deflate",
        nodata=nodata
    )

    safe_remove(output_raster)

    with rasterio.open(output_raster, "w", **profile) as dst:

        for r0 in tqdm(range(0, H, chunk_size), desc="Write rows"):
            for c0 in range(0, W, chunk_size):

                win = Window(
                    col_off=c0,
                    row_off=r0,
                    width=min(chunk_size, W - c0),
                    height=min(chunk_size, H - r0),
                )

                minx, miny, maxx, maxy = rasterio.windows.bounds(win, transform)
                idxs = list(sindex.intersection((minx, miny, maxx, maxy)))

                # start with nodata tile
                tile = np.full((int(win.height), int(win.width)), np.nan, dtype=np.float32)

                if idxs:
                    gsub = gdf.iloc[idxs]
                    shapes = [
                        (geom, float(val))
                        for geom, val in zip(gsub.geometry, gsub[attribute])
                        if geom is not None and np.isfinite(val)
                    ]

                    if shapes:
                        tile = rasterize(
                            shapes=shapes,
                            out_shape=tile.shape,
                            transform=rasterio.windows.transform(win, transform),
                            fill=np.nan,
                            dtype=dtype,
                            all_touched=all_touched,
                        )

                dst.write(tile, 1, window=win)

    print(">>> Height raster saved:", output_raster)


# ============================================================
# MAIN PIPELINE (resume-safe)
# ============================================================
def process_treecrowns_height_only(
    bdom_path,
    dgm_path,
    crowns_path,
    ndom_out,
    final_out,
    height_raster_out,
    percentile=80,
    height_clean=0.1,
    chunk_size=512,
    hist_vmin=0.0,
    hist_vmax=60.0,
    hist_step=0.25,
    rasterize_chunk_size=1024,
):
    print("====================================================")
    print(" TREE CROWN POST-PROCESSING (HEIGHT ONLY + RASTER)")
    print("====================================================")

    # STEP 0: nDOM
    if bdom_path and dgm_path and (not os.path.exists(ndom_out)):
        create_ndom(bdom_path, dgm_path, ndom_out)
    else:
        print(">>> Using existing nDOM / nDSM:", ndom_out)

    # If crowns_final already exists, skip height computation
    crowns_final_exists = os.path.exists(final_out) and layer_exists_vector(final_out, CROWNS_LAYER_OUT)

    if crowns_final_exists:
        print(f"\n>>> Found existing crowns output: {final_out} (layer '{CROWNS_LAYER_OUT}')")
        print(">>> Skipping height computation and cleaning step.")
    else:
        # STEP 1: read crowns
        gdf, _ = read_crowns(crowns_path, ndom_out, layer=CROWNS_LAYER_IN)

        # STEP 3: compute height (tiled histogram; safe for huge rasters)
        heights = zonal_percentile_tiled_hist(
            ndom_path=ndom_out,
            crowns_gdf=gdf,
            percentile=percentile,
            chunk_size=chunk_size,
            vmin=hist_vmin,
            vmax=hist_vmax,
            step=hist_step,
        )

        gdf[HEIGHT_ATTR] = heights[gdf["pid"].to_numpy()]

        valid = np.isfinite(gdf[HEIGHT_ATTR].to_numpy())
        print("\n  Height stats:",
              "min =", float(np.nanmin(gdf.loc[valid, HEIGHT_ATTR])) if valid.any() else np.nan,
              "max =", float(np.nanmax(gdf.loc[valid, HEIGHT_ATTR])) if valid.any() else np.nan,
              "valid =", int(valid.sum()),
              "/", len(gdf))

        # STEP 4: save crowns
        save_clean_crowns(gdf, final_out, height_clean=height_clean)

    # STEP 5: rasterize height (tiled, safe for huge rasters)
    if os.path.exists(height_raster_out):
        print(f"\n>>> Height raster already exists: {height_raster_out}")
        print(">>> Skipping rasterization.")
    else:
        rasterize_crowns_height_tiled(
            gpkg_path=final_out,
            raster_template=ndom_out,
            output_raster=height_raster_out,
            layer=CROWNS_LAYER_OUT,
            attribute=HEIGHT_ATTR,
            chunk_size=rasterize_chunk_size,
            dtype="float32",
            all_touched=False,
        )

    print("\n====================================================")
    print("  DONE")
    print("====================================================")


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    process_treecrowns_height_only(
        bdom_path=r"",   # leave empty if nDOM already exists
        dgm_path=r"",
        crowns_path=r"D:\DeepTree\tree_crown_merged_final_4.sqlite",
        ndom_out=r"X:\Austausch\Shadi\2025_12_04_DeepTree\Stadtgrenze\Frankfurt_2021_nDSM_mosaic_1m.tif",
        final_out=r"D:\DeepTree\crowns_final_Frankfurt_2021.gpkg",
        height_raster_out=r"D:\DeepTree\crowns_final_Frankfurt_2021.tif",
        percentile=80,
        height_clean=0.25,
        chunk_size=512,             # for height extraction
        hist_vmin=0.0,
        hist_vmax=60.0,
        hist_step=1.0,
        rasterize_chunk_size=1024,  # for writing the height raster (larger is faster if RAM allows)
    )
