#!/usr/bin/env python3
"""
Napa cabbage suitability model for California
==============================================

End-to-end Google Earth Engine implementation of a spatiotemporal
multi-criteria suitability model.

Outputs:
    1. risk-adjusted suitability
    2. historical mean suitability
    3. interannual standard deviation
    4. coefficient of variation
    5. optimal planting day-of-year
    6. long-run component suitability bands
    7. Monte-Carlo weight-robustness score
    8. CSV of top candidate pixels with coordinates

Core model:
    S_y(x,t) = Π_i mu_i(x,t,y) ^ w_i

    J(x,t) = E_y[S_y(x,t)] - lambda * SD_y[S_y(x,t)]

    (x*, t*) = argmax_{x,t} J(x,t)

Robustness:
    Random weight vectors are drawn from a Dirichlet distribution.
    For each weight vector, a long-run geometric suitability is calculated
    for every planting date. A pixel receives 1 if it belongs to the top
    percentile of eligible California land, otherwise 0. Robustness is the
    fraction of random weight vectors for which it remains in that top set.

Default data:
    Climate: OREGONSTATE/PRISM/ANd
    ETo: IDAHO_EPSCOR/GRIDMET, band "eto"
    Soil: projects/soilgrids-isric/*
    Boundary: TIGER/2018/States
    Terrain: USGS/SRTMGL1_003
    Land cover: ESA/WorldCover/v200

Important:
    The biological threshold values are model parameters and should be
    justified and sensitivity-tested in the final research paper.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Sequence, Tuple

import ee
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

COMPONENT_BANDS = [
    "mu_temp",
    "mu_heat",
    "mu_water",
    "mu_ph",
    "mu_texture",
    "mu_soc",
]


@dataclass(frozen=True)
class Config:
    # Historical analysis
    start_year: int = 2004
    end_year: int = 2024

    # Planting-window search
    growing_days: int = 60
    planting_start_doy: int = 1
    planting_end_doy: int = 305
    planting_step_days: int = 14

    # Output grid
    output_crs: str = "EPSG:3310"     # California Albers
    output_scale_m: int = 4000        # 4 km: close to PRISM/gridMET native scale

    # Temperature suitability (degrees C)
    temp_a: float = 8.0
    temp_b: float = 15.6
    temp_c: float = 21.1
    temp_d: float = 27.0

    # Heat stress
    heat_threshold_c: float = 24.0
    heat_alpha: float = 3.0

    # Water deficit
    effective_rain_fraction: float = 0.80
    water_tau_mm: float = 200.0

    # pH trapezoid
    ph_a: float = 5.0
    ph_b: float = 6.0
    ph_c: float = 6.5
    ph_d: float = 7.5

    # Texture suitability target, expressed as top-30-cm %
    texture_target_sand_pct: float = 55.0
    texture_target_clay_pct: float = 15.0
    texture_sigma_sand: float = 20.0
    texture_sigma_clay: float = 10.0

    # SOC saturation parameter, g/kg
    soc_c0_gkg: float = 20.0

    # Risk aversion
    risk_lambda: float = 0.50

    # Hard feasibility mask
    use_feasibility_mask: bool = True
    max_slope_deg: float = 15.0

    # Baseline weights
    weights: Tuple[float, float, float, float, float, float] = (
        0.25,  # temperature
        0.15,  # heat
        0.20,  # water
        0.15,  # pH
        0.15,  # texture
        0.10,  # SOC
    )

    # Monte Carlo weight robustness
    run_robustness: bool = True
    robustness_draws: int = 25
    robustness_dirichlet_alpha: float = 1.0
    robustness_top_percent: float = 5.0
    random_seed: int = 42

    # Export
    drive_folder: str = "napa_cabbage_california"
    export_prefix: str = "napa_cabbage_ca"

    # Numerical
    epsilon: float = 1e-6


# Crop-coefficient stage fractions.  These reproduce 15/15/20/10 days
# when growing_days=60, while scaling coherently for sensitivity runs.
KC_STAGE_FRACTIONS = [
    (0.00, 0.25, 0.45),
    (0.25, 0.50, 0.75),
    (0.50, 5.0 / 6.0, 1.05),
    (5.0 / 6.0, 1.00, 0.90),
]

def kc_stages(growing_days: int):
    edges = []
    for f0, f1, kc in KC_STAGE_FRACTIONS:
        start = int(round(f0 * growing_days))
        end = int(round(f1 * growing_days))
        end = max(end, start + 1)
        edges.append((start, min(end, growing_days), kc))
    # Ensure the last stage reaches the requested season length.
    a, _, k = edges[-1]
    edges[-1] = (a, growing_days, k)
    return edges


# ---------------------------------------------------------------------------
# Earth Engine setup
# ---------------------------------------------------------------------------

def initialize_ee(project: str | None) -> None:
    """Initialize Earth Engine; authenticate interactively if necessary."""
    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
    except Exception:
        print("Earth Engine authentication is required.")
        ee.Authenticate()
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()


def california_geometry() -> ee.Geometry:
    states = ee.FeatureCollection("TIGER/2018/States")
    return states.filter(ee.Filter.eq("STUSPS", "CA")).geometry()


# ---------------------------------------------------------------------------
# SoilGrids helpers
# ---------------------------------------------------------------------------

SOIL_ASSETS = {
    "phh2o": "projects/soilgrids-isric/phh2o_mean",
    "sand": "projects/soilgrids-isric/sand_mean",
    "silt": "projects/soilgrids-isric/silt_mean",
    "clay": "projects/soilgrids-isric/clay_mean",
    "soc": "projects/soilgrids-isric/soc_mean",
}


def _soil_depth_band_names(asset_id: str) -> Tuple[str, str, str]:
    """
    Discover the three bands representing 0-5, 5-15 and 15-30 cm.
    This avoids hard-coding the exact SoilGrids band prefix.
    """
    img = ee.Image(asset_id)
    names = img.bandNames().getInfo()
    wanted = []
    for depth in ("0-5cm", "5-15cm", "15-30cm"):
        matches = [name for name in names if depth in name]
        if len(matches) != 1:
            raise RuntimeError(
                f"Could not uniquely identify SoilGrids band for {depth} in "
                f"{asset_id}. Available bands: {names}"
            )
        wanted.append(matches[0])
    return tuple(wanted)


def top30_soil(property_name: str, cfg: Config) -> ee.Image:
    """
    Thickness-weighted 0-30 cm SoilGrids mean.

    SoilGrids conversion factor for the properties used here is 10:
      pH: raw / 10 -> pH
      sand, silt, clay: raw / 10 -> %
      SOC: raw / 10 -> g/kg
    """
    asset = SOIL_ASSETS[property_name]
    img = ee.Image(asset)
    b0, b1, b2 = _soil_depth_band_names(asset)

    top30_raw = (
        img.select(b0).multiply(5.0)
        .add(img.select(b1).multiply(10.0))
        .add(img.select(b2).multiply(15.0))
        .divide(30.0)
    )

    conventional = top30_raw.divide(10.0).rename(property_name)

    # Aggregate 250 m soil predictions to the analysis grid.
    return (
        conventional
        .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=4096)
        .reproject(crs=cfg.output_crs, scale=cfg.output_scale_m)
    )


# ---------------------------------------------------------------------------
# Suitability functions
# ---------------------------------------------------------------------------

def trapezoid(
    x: ee.Image,
    a: float,
    b: float,
    c: float,
    d: float,
    name: str,
) -> ee.Image:
    """Trapezoidal fuzzy membership function in [0, 1]."""
    result = ee.Image.constant(0.0)
    result = result.where(
        x.gt(a).And(x.lt(b)),
        x.subtract(a).divide(b - a),
    )
    result = result.where(
        x.gte(b).And(x.lte(c)),
        1.0,
    )
    result = result.where(
        x.gt(c).And(x.lt(d)),
        ee.Image.constant(d).subtract(x).divide(d - c),
    )
    return result.clamp(0.0, 1.0).rename(name)


def build_static_components(cfg: Config) -> ee.Image:
    """pH, texture and SOC suitability, independent of year/planting date."""
    ph = top30_soil("phh2o", cfg)
    sand = top30_soil("sand", cfg)
    silt = top30_soil("silt", cfg)
    clay = top30_soil("clay", cfg)
    soc = top30_soil("soc", cfg)

    mu_ph = trapezoid(
        ph, cfg.ph_a, cfg.ph_b, cfg.ph_c, cfg.ph_d, "mu_ph"
    )

    # Normalize texture fractions, because modelled sand+silt+clay can differ
    # slightly from 100 due to rounding/model prediction.
    total_texture = sand.add(silt).add(clay).max(cfg.epsilon)
    sand_pct = sand.divide(total_texture).multiply(100.0)
    clay_pct = clay.divide(total_texture).multiply(100.0)

    # Smooth distance from a researcher-defined sandy-loam-like target.
    texture_distance = (
        sand_pct.subtract(cfg.texture_target_sand_pct)
        .divide(cfg.texture_sigma_sand)
        .pow(2)
        .add(
            clay_pct.subtract(cfg.texture_target_clay_pct)
            .divide(cfg.texture_sigma_clay)
            .pow(2)
        )
    )
    mu_texture = texture_distance.multiply(-0.5).exp().rename("mu_texture")

    # Saturating SOC benefit.
    mu_soc = (
        ee.Image.constant(1.0)
        .subtract(soc.divide(cfg.soc_c0_gkg).multiply(-1.0).exp())
        .clamp(0.0, 1.0)
        .rename("mu_soc")
    )

    return (
        mu_ph.addBands(mu_texture).addBands(mu_soc)
        .addBands(ph.rename("soil_ph"))
        .addBands(sand_pct.rename("soil_sand_pct"))
        .addBands(silt.divide(total_texture).multiply(100.0).rename("soil_silt_pct"))
        .addBands(clay_pct.rename("soil_clay_pct"))
        .addBands(soc.rename("soil_soc_gkg"))
    )


def build_feasibility_mask(cfg: Config) -> ee.Image:
    """
    Exclude clearly infeasible land:
      - open water
      - built-up land
      - permanent snow/ice
      - slope above configured threshold

    WorldCover classes:
      50 = built-up
      70 = snow/ice
      80 = permanent water
    """
    if not cfg.use_feasibility_mask:
        return ee.Image.constant(1).rename("feasible").toByte()

    worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
    land_ok = (
        worldcover.neq(50)
        .And(worldcover.neq(70))
        .And(worldcover.neq(80))
    )

    # Use modal land-cover class at the output resolution.
    land_ok = (
        land_ok
        .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=4096)
        .reproject(crs=cfg.output_crs, scale=cfg.output_scale_m)
        .gte(0.5)
    )

    dem = ee.Image("USGS/SRTMGL1_003")
    slope = ee.Terrain.slope(dem)
    mean_slope = (
        slope
        .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=65536)
        .reproject(crs=cfg.output_crs, scale=cfg.output_scale_m)
    )
    slope_ok = mean_slope.lte(cfg.max_slope_deg)

    return land_ok.And(slope_ok).rename("feasible").toByte()


# ---------------------------------------------------------------------------
# Climate model
# ---------------------------------------------------------------------------

PRISM = ee.ImageCollection("OREGONSTATE/PRISM/ANd")
GRIDMET = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET")


def _at_output_grid(img: ee.Image, cfg: Config) -> ee.Image:
    """Bilinearly place a continuous climate image on the output grid."""
    return (
        img.resample("bilinear")
        .reproject(crs=cfg.output_crs, scale=cfg.output_scale_m)
    )


def _window_dates(year: int, doy: int, cfg: Config) -> Tuple[ee.Date, ee.Date]:
    start = ee.Date.fromYMD(year, 1, 1).advance(doy - 1, "day")
    end = start.advance(cfg.growing_days, "day")
    return start, end


def climate_components(year: int, doy: int, cfg: Config) -> ee.Image:
    """Temperature, heat-stress and water-deficit suitability for one year/window."""
    start, end = _window_dates(year, doy, cfg)

    prism_window = PRISM.filterDate(start, end)

    mean_temp = prism_window.select("tmean").mean()

    hot_fraction = (
        prism_window.select("tmax")
        .map(lambda img: ee.Image(img).gt(cfg.heat_threshold_c).rename("hot"))
        .mean()
    )

    mu_temp = trapezoid(
        _at_output_grid(mean_temp, cfg),
        cfg.temp_a,
        cfg.temp_b,
        cfg.temp_c,
        cfg.temp_d,
        "mu_temp",
    )

    mu_heat = (
        _at_output_grid(hot_fraction, cfg)
        .multiply(-cfg.heat_alpha)
        .exp()
        .clamp(0.0, 1.0)
        .rename("mu_heat")
    )

    # Stage-wise irrigation deficit:
    # D = sum_stage max(0, Kc * sum(ETo) - eta * sum(P))
    deficit = ee.Image.constant(0.0)

    for stage_start, stage_end, kc in kc_stages(cfg.growing_days):
        # If the user changes growing_days below a stage boundary, trim safely.
        if stage_start >= cfg.growing_days:
            continue
        clipped_end = min(stage_end, cfg.growing_days)

        s = start.advance(stage_start, "day")
        e = start.advance(clipped_end, "day")

        eto_sum = (
            GRIDMET.filterDate(s, e)
            .select("eto")
            .sum()
            .multiply(kc)
        )
        p_eff = (
            PRISM.filterDate(s, e)
            .select("ppt")
            .sum()
            .multiply(cfg.effective_rain_fraction)
        )

        stage_deficit = eto_sum.subtract(p_eff).max(0.0)
        deficit = deficit.add(stage_deficit)

    deficit = _at_output_grid(deficit.rename("deficit_mm"), cfg)

    mu_water = (
        deficit.divide(cfg.water_tau_mm)
        .multiply(-1.0)
        .exp()
        .clamp(0.0, 1.0)
        .rename("mu_water")
    )

    return mu_temp.addBands(mu_heat).addBands(mu_water)


def components_for_year_doy(
    year: int,
    doy: int,
    cfg: Config,
    static_components: ee.Image,
    mask: ee.Image,
) -> ee.Image:
    climate = climate_components(year, doy, cfg)
    all_components = climate.addBands(static_components)
    return all_components.updateMask(mask)


# ---------------------------------------------------------------------------
# Score and temporal aggregation
# ---------------------------------------------------------------------------

def validate_weights(weights: Sequence[float]) -> None:
    if len(weights) != len(COMPONENT_BANDS):
        raise ValueError("Weight vector must have six values.")
    if any(w < 0 for w in weights):
        raise ValueError("Weights must be non-negative.")
    if not np.isclose(sum(weights), 1.0, atol=1e-8):
        raise ValueError(f"Weights must sum to 1. Got {sum(weights)}")


def weighted_geometric_score(
    components: ee.Image,
    weights: Sequence[float],
    out_name: str = "score",
) -> ee.Image:
    validate_weights(weights)

    score = ee.Image.constant(1.0)
    for band, weight in zip(COMPONENT_BANDS, weights):
        score = score.multiply(
            components.select(band).clamp(0.0, 1.0).pow(float(weight))
        )
    return score.rename(out_name)


def planting_doys(cfg: Config) -> List[int]:
    return list(
        range(
            cfg.planting_start_doy,
            cfg.planting_end_doy + 1,
            cfg.planting_step_days,
        )
    )


def summarize_planting_doy(
    doy: int,
    years: Sequence[int],
    cfg: Config,
    static_components: ee.Image,
    mask: ee.Image,
) -> ee.Image:
    """
    For one planting DOY:
      - annual suitability scores
      - historical mean, SD, CV
      - risk-adjusted objective
      - geometric mean of each component across years

    The geometric component means are later reused for weight-robustness.
    """
    annual_components = [
        components_for_year_doy(year, doy, cfg, static_components, mask)
        for year in years
    ]

    annual_scores = [
        weighted_geometric_score(img, cfg.weights, "score")
        for img in annual_components
    ]

    score_col = ee.ImageCollection.fromImages(annual_scores)
    mean_score = score_col.mean().rename("mean_score")
    sd_score = score_col.reduce(ee.Reducer.stdDev()).rename("sd_score")

    cv = (
        sd_score.divide(mean_score.max(cfg.epsilon))
        .rename("cv")
    )

    objective = (
        mean_score.subtract(sd_score.multiply(cfg.risk_lambda))
        .clamp(0.0, 1.0)
        .rename("objective")
    )

    # Exact geometric mean of each suitability component over years:
    # exp(mean(log(mu_i))).
    log_component_images = [
        img.select(COMPONENT_BANDS).clamp(cfg.epsilon, 1.0).log()
        for img in annual_components
    ]
    geo_components = (
        ee.ImageCollection.fromImages(log_component_images)
        .mean()
        .exp()
        .rename([f"g_{b}" for b in COMPONENT_BANDS])
    )

    doy_img = ee.Image.constant(float(doy)).rename("planting_doy")

    return (
        objective
        .addBands(mean_score)
        .addBands(sd_score)
        .addBands(cv)
        .addBands(doy_img)
        .addBands(geo_components)
        .updateMask(mask)
        .set("planting_doy", doy)
    )


# ---------------------------------------------------------------------------
# Monte Carlo weight robustness
# ---------------------------------------------------------------------------

def geo_score_for_summary(
    summary: ee.Image,
    weights: Sequence[float],
) -> ee.Image:
    validate_weights(weights)
    score = ee.Image.constant(1.0)
    for band, weight in zip(COMPONENT_BANDS, weights):
        score = score.multiply(
            summary.select(f"g_{band}")
            .clamp(0.0, 1.0)
            .pow(float(weight))
        )
    return score.rename("score")


def monte_carlo_robustness(
    summaries: Sequence[ee.Image],
    region: ee.Geometry,
    mask: ee.Image,
    cfg: Config,
) -> ee.Image:
    """
    Robustness = fraction of random weight vectors for which a pixel lies
    in the top configured percentage of eligible California pixels.

    Each random weighting is allowed to choose its own optimal planting DOY.
    """
    if not cfg.run_robustness or cfg.robustness_draws <= 0:
        return ee.Image.constant(0.0).rename("robustness").updateMask(mask)

    rng = np.random.default_rng(cfg.random_seed)

    alpha = np.full(
        len(COMPONENT_BANDS),
        cfg.robustness_dirichlet_alpha,
        dtype=float,
    )
    weight_draws = rng.dirichlet(alpha, size=cfg.robustness_draws)

    binary_images = []
    percentile = 100.0 - cfg.robustness_top_percent

    for k, weights in enumerate(weight_draws):
        weighted_by_doy = [
            geo_score_for_summary(img, weights)
            for img in summaries
        ]
        best_score = (
            ee.ImageCollection.fromImages(weighted_by_doy)
            .max()
            .rename("score")
            .updateMask(mask)
        )

        p = best_score.reduceRegion(
            reducer=ee.Reducer.percentile([percentile]),
            geometry=region,
            scale=cfg.output_scale_m,
            crs=cfg.output_crs,
            maxPixels=100_000_000,
            tileScale=4,
            bestEffort=False,
        )

        threshold = ee.Number(p.get(f"score_p{int(percentile)}"))
        is_top = (
            best_score.gte(threshold)
            .rename("robust")
            .set("draw", k)
        )
        binary_images.append(is_top)

    return (
        ee.ImageCollection.fromImages(binary_images)
        .mean()
        .rename("robustness")
        .updateMask(mask)
    )


# ---------------------------------------------------------------------------
# Build model
# ---------------------------------------------------------------------------

def build_model(cfg: Config) -> Tuple[ee.Image, ee.FeatureCollection]:
    validate_weights(cfg.weights)

    region = california_geometry()
    mask = build_feasibility_mask(cfg)
    static_components = build_static_components(cfg)

    years = list(range(cfg.start_year, cfg.end_year + 1))
    doys = planting_doys(cfg)

    print(
        f"Building {len(doys)} planting windows × {len(years)} years "
        f"({len(doys) * len(years)} climate-window evaluations)."
    )

    summaries = [
        summarize_planting_doy(
            doy,
            years,
            cfg,
            static_components,
            mask,
        )
        for doy in doys
    ]

    summary_col = ee.ImageCollection.fromImages(summaries)

    # Pixelwise optimal planting date under baseline weights and risk criterion.
    best = summary_col.qualityMosaic("objective")

    robustness = monte_carlo_robustness(
        summaries=summaries,
        region=region,
        mask=mask,
        cfg=cfg,
    )

    result = (
        best.select(
            [
                "objective",
                "mean_score",
                "sd_score",
                "cv",
                "planting_doy",
                *[f"g_{b}" for b in COMPONENT_BANDS],
            ]
        )
        .addBands(robustness)
        .addBands(static_components.select([
            "soil_ph", "soil_sand_pct", "soil_silt_pct",
            "soil_clay_pct", "soil_soc_gkg"
        ]))
        .updateMask(mask)
        .clip(region)
    )

    # Candidate ranking gives robustness priority, then objective score.
    rank_metric = (
        robustness.multiply(10.0)
        .add(result.select("objective"))
        .rename("rank_metric")
    )

    candidate_image = (
        ee.Image.pixelLonLat()
        .addBands(result)
        .addBands(rank_metric)
    )

    candidates = (
        candidate_image.sample(
            region=region,
            scale=cfg.output_scale_m,
            projection=ee.Projection(cfg.output_crs),
            geometries=True,
            tileScale=4,
            dropNulls=True,
        )
        .sort("rank_metric", False)
        .limit(200)
    )

    return result, candidates


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_to_drive(
    result: ee.Image,
    candidates: ee.FeatureCollection,
    cfg: Config,
) -> Tuple[ee.batch.Task, ee.batch.Task]:
    region = california_geometry()

    image_task = ee.batch.Export.image.toDrive(
        image=result,
        description=f"{cfg.export_prefix}_raster",
        folder=cfg.drive_folder,
        fileNamePrefix=f"{cfg.export_prefix}_raster",
        region=region,
        scale=cfg.output_scale_m,
        crs=cfg.output_crs,
        maxPixels=100_000_000,
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": True},
    )

    table_task = ee.batch.Export.table.toDrive(
        collection=candidates,
        description=f"{cfg.export_prefix}_top_candidates",
        folder=cfg.drive_folder,
        fileNamePrefix=f"{cfg.export_prefix}_top_candidates",
        fileFormat="CSV",
        selectors=[
            "longitude",
            "latitude",
            "objective",
            "mean_score",
            "sd_score",
            "cv",
            "planting_doy",
            "g_mu_temp",
            "g_mu_heat",
            "g_mu_water",
            "g_mu_ph",
            "g_mu_texture",
            "g_mu_soc",
            "robustness",
            "soil_ph",
            "soil_sand_pct",
            "soil_silt_pct",
            "soil_clay_pct",
            "soil_soc_gkg",
            "rank_metric",
        ],
    )

    image_task.start()
    table_task.start()

    return image_task, table_task


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def config_for_mode(mode: str) -> Config:
    base = Config()

    if mode == "quick":
        # Fast sanity check before launching the research-scale run.
        return replace(
            base,
            start_year=2018,
            end_year=2024,
            planting_step_days=28,
            robustness_draws=8,
            export_prefix="napa_cabbage_ca_quick",
        )

    if mode == "full":
        # Research-scale settings.
        return replace(
            base,
            start_year=2004,
            end_year=2024,
            planting_step_days=7,
            robustness_draws=100,
            export_prefix="napa_cabbage_ca_full",
        )

    if mode == "standard":
        return base

    raise ValueError(f"Unknown mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="California Napa cabbage suitability model"
    )
    parser.add_argument(
        "--project",
        default=os.environ.get("EE_PROJECT"),
        help=(
            "Google Cloud project enabled for Earth Engine. "
            "Can also be supplied via EE_PROJECT environment variable."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["quick", "standard", "full"],
        default="quick",
        help=(
            "quick: validation run; standard: balanced run; "
            "full: 7-day planting grid and 100 robustness draws."
        ),
    )
    parser.add_argument(
        "--no-robustness",
        action="store_true",
        help="Skip Monte Carlo robustness for faster testing.",
    )
    args = parser.parse_args()

    initialize_ee(args.project)

    cfg = config_for_mode(args.mode)
    if args.no_robustness:
        cfg = replace(cfg, run_robustness=False, robustness_draws=0)

    print("Configuration:")
    print(cfg)

    result, candidates = build_model(cfg)
    image_task, table_task = export_to_drive(result, candidates, cfg)

    print("\nEarth Engine export tasks started.")
    print(f"Raster task ID: {image_task.id}")
    print(f"Candidate CSV task ID: {table_task.id}")
    print(f"Google Drive folder: {cfg.drive_folder}")
    print(
        "\nCheck task status in the Earth Engine Tasks panel or with "
        "`earthengine task list`."
    )


if __name__ == "__main__":
    main()
