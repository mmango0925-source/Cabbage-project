#!/usr/bin/env python3
"""Launch baseline and sensitivity Earth Engine exports for the research pipeline."""
from __future__ import annotations
import argparse, json, os
from dataclasses import replace, asdict
from pathlib import Path
import pandas as pd

from napa_cabbage_gee import Config, build_model, export_to_drive, initialize_ee

ROOT = Path(__file__).resolve().parent


def full_config() -> Config:
    return replace(
        Config(),
        start_year=2004,
        end_year=2024,
        planting_step_days=7,
        robustness_draws=100,
        export_prefix="baseline_full",
        drive_folder="napa_cabbage_research",
    )


def launch(cfg: Config, prefix: str):
    cfg = replace(cfg, export_prefix=prefix)
    result, candidates = build_model(cfg)
    return export_to_drive(result, candidates, cfg)


def scenario_configs(base: Config):
    spec = json.loads((ROOT / "sensitivity_scenarios.json").read_text())
    scenarios = []
    for value in spec["temperature_upper"]:
        scenarios.append((f"tempD_{value:g}", replace(base, temp_d=value)))
    for value in spec["heat_threshold"]:
        scenarios.append((f"heatT_{value:g}", replace(base, heat_threshold_c=value)))
    for value in spec["rain_efficiency"]:
        scenarios.append((f"rainEta_{value:g}", replace(base, effective_rain_fraction=value)))
    for value in spec["water_tau"]:
        scenarios.append((f"waterTau_{value:g}", replace(base, water_tau_mm=value)))
    for value in spec["growing_days"]:
        scenarios.append((f"growDays_{value}", replace(base, growing_days=int(value))))
    for value in spec["risk_lambda"]:
        scenarios.append((f"riskLambda_{value:g}", replace(base, risk_lambda=value)))
    for value in spec["slope_cutoff"]:
        scenarios.append((f"slope_{value:g}", replace(base, max_slope_deg=value)))
    for name, weights in spec["weights"].items():
        scenarios.append((f"weights_{name}", replace(base, weights=tuple(weights))))
    return scenarios


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["baseline", "sensitivity", "all", "manifest"])
    ap.add_argument("--project", default=os.getenv("EE_PROJECT"))
    ap.add_argument("--standard", action="store_true", help="Use 14-day planting grid / 25 robustness draws")
    ap.add_argument("--scenario", help="Launch only one named sensitivity scenario")
    args = ap.parse_args()

    base = Config() if args.standard else full_config()
    sens_base = replace(base, run_robustness=False, robustness_draws=0)
    scenarios = scenario_configs(sens_base)

    rows = [{"scenario":"baseline", **asdict(base)}]
    rows += [{"scenario":name, **asdict(cfg)} for name, cfg in scenarios]
    pd.DataFrame(rows).to_csv(ROOT / "scenario_manifest.csv", index=False)

    if args.command == "manifest":
        print(f"Wrote {ROOT/'scenario_manifest.csv'}")
        return

    initialize_ee(args.project)

    if args.command in ("baseline", "all"):
        print("Launching baseline...")
        launch(base, "baseline_full" if not args.standard else "baseline_standard")

    if args.command in ("sensitivity", "all"):
        chosen = scenarios
        if args.scenario:
            chosen = [(n,c) for n,c in scenarios if n == args.scenario]
            if not chosen:
                raise SystemExit(f"Unknown scenario: {args.scenario}")
        for name, cfg in chosen:
            print(f"Launching sensitivity: {name}")
            launch(cfg, f"sens_{name}")

    print("Earth Engine export tasks have been submitted to the configured Google Drive folder.")

if __name__ == "__main__":
    main()
