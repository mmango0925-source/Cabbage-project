# Final California Napa Cabbage Research Pipeline

This folder contains the complete reproducible pipeline for the California Napa-cabbage site-selection study.

## Pipeline

1. **Baseline statewide model** — PRISM + gridMET + SoilGrids, 2004–2024, weekly planting-date search.
2. **Risk adjustment** — historical mean suitability minus λ × interannual SD.
3. **Monte Carlo weight robustness** — 100 Dirichlet weight draws in full mode.
4. **Sensitivity analysis** — temperature, heat, rainfall efficiency, water penalty, season length, risk aversion, slope, and alternative weights.
5. **Spatial robustness** — Jaccard overlap of the top-5% region.
6. **CIMIS validation** — compare gridMET ETo with Spatial CIMIS ASCE ETo at candidate coordinates.
7. **SSURGO validation** — use USDA Soil Data Access to retrieve horizon pH/texture at candidate coordinates.
8. **Publication outputs** — maps, Jaccard table/figure, and top-candidate table.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Earth Engine requires an enabled Google Cloud project:

```bash
export EE_PROJECT="your-project-id"
```

CIMIS validation additionally requires a free CIMIS Web API application key:

```bash
export CIMIS_APP_KEY="your-cimis-app-key"
```

## A. Create scenario manifest

```bash
python research_pipeline.py manifest
```

This writes `scenario_manifest.csv`, which should be retained as part of the research audit trail.

## B. Launch the baseline

For a lower-cost check first:

```bash
python napa_cabbage_gee.py --mode quick --no-robustness
```

Then launch the final baseline:

```bash
python research_pipeline.py baseline
```

The full baseline uses 2004–2024, planting dates every 7 days, and 100 Monte-Carlo weight draws.

## C. Launch sensitivity scenarios

To launch one scenario:

```bash
python research_pipeline.py sensitivity --scenario tempD_25
```

To submit all configured sensitivity scenarios:

```bash
python research_pipeline.py sensitivity
```

Sensitivity runs omit Monte-Carlo robustness because the purpose is to compare deterministic top-region movement against the baseline.

## D. Download Earth Engine outputs

Earth Engine writes GeoTIFFs and candidate CSVs to the Drive folder `napa_cabbage_research`.

Arrange downloaded files locally, e.g.:

```text
results/
  baseline/baseline_full_raster.tif
  baseline/baseline_full_top_candidates.csv
  sensitivity/sens_tempD_25_raster.tif
  sensitivity/sens_heatT_22_raster.tif
  ...
```

## E. Produce final maps and Jaccard sensitivity results

```bash
python postprocess.py \
  results/baseline/baseline_full_raster.tif \
  --sensitivity-dir results/sensitivity \
  --candidate-csv results/baseline/baseline_full_top_candidates.csv \
  --outdir results/final_analysis
```

Outputs include:

- `map_objective.png`
- `map_planting_doy.png`
- `map_robustness.png`
- `map_cv.png`
- `sensitivity_jaccard.csv`
- `sensitivity_jaccard.png`
- `top30_candidates.csv`

## F. Validate ETo with Spatial CIMIS

```bash
python cimis_validation.py \
  results/baseline/baseline_full_top_candidates.csv \
  --n 15 \
  --years 2015-2024 \
  --out results/final_analysis/cimis_validation.csv
```

This queries the Spatial CIMIS coordinate service for daily ASCE ETo and independently extracts gridMET ETo through Earth Engine for the same coordinate and 60-day planting window. It writes annual paired values plus an overall bias/RMSE/correlation file.

## G. Validate topsoil with SSURGO

```bash
python ssurgo_validation.py \
  results/baseline/baseline_full_top_candidates.csv \
  --n 30 \
  --out results/final_analysis/ssurgo_validation.csv
```

The script queries USDA Soil Data Access at each candidate location and calculates component × horizon-thickness weighted 0–30 cm representative pH, sand, silt, clay, and organic matter.

## H. Paper structure

Use `METHODS_BLUEPRINT.md` as the reproducible methods outline. Keep `scenario_manifest.csv`, validation CSVs, and the exact code version with the final submission.

## Model caveats

- Suitability thresholds and weights are explicit modelling assumptions and must be sensitivity-tested.
- gridMET is the statewide ETo backbone; Spatial CIMIS is used as an independent California-specific validation.
- SoilGrids supports computationally consistent statewide mapping; SSURGO is used for higher-detail candidate validation.
- Environmental suitability is not the same as economic feasibility. Water rights, land cost, disease, market access, labor, and zoning are outside this model.
