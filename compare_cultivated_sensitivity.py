#!/usr/bin/env python3
"""
Compare cultivated-land sensitivity rasters exported by napa_cabbage_gee_v6.py.

Example:
    python compare_cultivated_sensitivity.py \
        --c015 napa_cabbage_ca_standard_cultivated_15_v6_raster.tif \
        --c030 napa_cabbage_ca_standard_cultivated_30_v6_raster.tif \
        --c050 napa_cabbage_ca_standard_cultivated_50_v6_raster.tif \
        --c070 napa_cabbage_ca_standard_cultivated_70_v6_raster.tif \
        --out cultivated_jaccard.csv

The script:
  1. Reads the 'objective' band from each raster.
  2. Defines that raster's top 5% among valid pixels.
  3. Computes pairwise Jaccard similarity:
         J(A,B) = |A ∩ B| / |A ∪ B|
  4. Writes a pairwise matrix and a compact summary vs the 0.50 baseline.

All rasters must have the same grid, shape and transform.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


def read_objective(path: str):
    with rasterio.open(path) as src:
        names = list(src.descriptions)
        if "objective" in names:
            band_index = names.index("objective") + 1
        else:
            # v6 exports objective as the first band.
            band_index = 1

        arr = src.read(band_index, masked=True).astype("float64")
        profile = {
            "shape": arr.shape,
            "transform": src.transform,
            "crs": src.crs,
        }
    return arr, profile


def top_fraction_mask(arr, fraction=0.05):
    valid = (~np.ma.getmaskarray(arr)) & np.isfinite(arr.filled(np.nan))
    values = np.asarray(arr.filled(np.nan))[valid]
    if values.size == 0:
        raise ValueError("Raster contains no valid objective pixels.")

    threshold = np.quantile(values, 1.0 - fraction)
    return valid & (np.asarray(arr.filled(np.nan)) >= threshold), threshold


def jaccard(a, b):
    union = np.logical_or(a, b).sum()
    if union == 0:
        return np.nan
    intersection = np.logical_and(a, b).sum()
    return intersection / union


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c015", required=True)
    parser.add_argument("--c030", required=True)
    parser.add_argument("--c050", required=True)
    parser.add_argument("--c070", required=True)
    parser.add_argument("--out", default="cultivated_jaccard.csv")
    parser.add_argument("--top-fraction", type=float, default=0.05)
    args = parser.parse_args()

    paths = {
        "Cmin_0.15": args.c015,
        "Cmin_0.30": args.c030,
        "Cmin_0.50": args.c050,
        "Cmin_0.70": args.c070,
    }

    arrays = {}
    profiles = {}
    masks = {}
    thresholds = {}

    for label, path in paths.items():
        arr, profile = read_objective(path)
        arrays[label] = arr
        profiles[label] = profile
        mask, threshold = top_fraction_mask(
            arr, fraction=args.top_fraction
        )
        masks[label] = mask
        thresholds[label] = threshold

    labels = list(paths)

    reference = profiles[labels[0]]
    for label in labels[1:]:
        current = profiles[label]
        if current["shape"] != reference["shape"]:
            raise ValueError(
                f"Raster shape mismatch: {label} has "
                f"{current['shape']}, expected {reference['shape']}."
            )
        if current["crs"] != reference["crs"]:
            raise ValueError(f"CRS mismatch for {label}.")
        if current["transform"] != reference["transform"]:
            raise ValueError(f"Transform mismatch for {label}.")

    matrix = pd.DataFrame(index=labels, columns=labels, dtype=float)
    for a in labels:
        for b in labels:
            matrix.loc[a, b] = jaccard(masks[a], masks[b])

    out = Path(args.out)
    matrix.to_csv(out)

    baseline = "Cmin_0.50"
    summary = pd.DataFrame(
        {
            "scenario": labels,
            "objective_top5_threshold": [thresholds[x] for x in labels],
            "top5_pixel_count": [int(masks[x].sum()) for x in labels],
            "jaccard_vs_Cmin_0.50": [
                jaccard(masks[x], masks[baseline]) for x in labels
            ],
        }
    )
    summary_path = out.with_name(out.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("\nPairwise Jaccard matrix:")
    print(matrix.round(4).to_string())
    print("\nSummary versus Cmin=0.50:")
    print(summary.round(4).to_string(index=False))
    print(f"\nSaved: {out}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
