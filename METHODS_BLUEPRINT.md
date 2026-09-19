# Methods blueprint for the final paper

## 1. Spatial domain and temporal window
California is represented on a common California Albers grid (EPSG:3310). The default analysis period is 2004–2024 and candidate planting dates are tested every 7 days. Each simulated crop has a 60-day growing window in the baseline model.

## 2. Input variables
For grid cell x, planting date t, and year y, six fuzzy suitability components are calculated:

- mean growing-season temperature, μ_T
- heat-stress frequency, μ_H
- irrigation-demand proxy, μ_W
- topsoil pH, μ_pH
- topsoil texture, μ_Q
- soil organic carbon, μ_C

Climate is derived from PRISM daily temperature/precipitation and gridMET grass-reference evapotranspiration. Soil properties use 0–30 cm thickness-weighted SoilGrids predictions. Land/water, built-up land, permanent snow/ice, and high-slope cells are removed by a feasibility mask.

## 3. Suitability functions
Temperature and pH use trapezoidal fuzzy membership functions. Heat stress uses μ_H = exp(-αH). Water stress is based on stage-wise crop water demand D = Σ max(0, Kc ETo − ηP), followed by μ_W = exp(-D/τ). Texture uses a smooth distance from a sandy-loam-like target; SOC uses μ_C = 1 − exp(-C/C0).

## 4. Multi-criteria aggregation
Annual suitability is a weighted geometric mean:

S_y(x,t) = Π μ_i(x,t,y)^(w_i),  Σw_i = 1.

The baseline weights are [0.25, 0.15, 0.20, 0.15, 0.15, 0.10] for temperature, heat, water, pH, texture, and SOC respectively.

## 5. Temporal risk and planting optimization
For each location and candidate planting date:

J(x,t) = mean_y[S_y(x,t)] − λ sd_y[S_y(x,t)].

The historically robust planting date is t*(x) = argmax_t J(x,t), rather than selecting the best date separately with hindsight in each year.

## 6. Weight robustness
Weight vectors are sampled from a Dirichlet distribution. For every random weighting, the best planting date is re-selected and the top 5% of feasible California cells are recorded. Robustness is:

R(x) = number of draws where x is top 5% / total draws.

## 7. Parameter sensitivity
One-at-a-time scenarios vary temperature cutoffs, heat threshold, effective rainfall fraction, water penalty scale, crop duration, risk aversion, slope mask, and alternative weight structures. The overlap of the baseline and scenario top-5% regions is measured with:

J(A,B) = |A∩B| / |A∪B|.

## 8. External validation
### Water
At the top candidate coordinates, the model’s gridMET ETo is compared with Spatial CIMIS ASCE ETo over the same historical planting windows. Report bias, RMSE, correlation, and candidate-level discrepancies.

### Soil
Candidate coordinates are intersected with USDA SSURGO map units through Soil Data Access. Component- and horizon-weighted 0–30 cm pH and texture are calculated and compared qualitatively/quantitatively with the gridded soil estimates used in the statewide model.

## 9. Recommended results sequence
1. Baseline suitability map.
2. Optimal planting-date map.
3. Interannual CV map.
4. Monte-Carlo robustness map.
5. Top candidate table.
6. Sensitivity Jaccard plot/table.
7. CIMIS vs gridMET validation scatter/metrics.
8. SSURGO soil validation table.

## 10. Interpretation constraint
The output is a model-based environmental suitability ranking. It does not include land price, water rights, irrigation infrastructure, labor, crop disease, market access, zoning, or farm-management practices; therefore it should not be presented as a direct recommendation to purchase or farm a particular parcel.
