#!/usr/bin/env python3
"""Validate gridMET ETo against Spatial CIMIS at selected candidate coordinates."""
from __future__ import annotations
import argparse, os, time
from datetime import datetime, timedelta
from pathlib import Path
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ee

from napa_cabbage_gee import initialize_ee

API = "https://et.water.ca.gov/api/data"
GRIDMET = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET")


def dates_for_doy(year: int, doy: int, days: int):
    start = datetime(year,1,1) + timedelta(days=int(doy)-1)
    end = start + timedelta(days=days-1)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def cimis_sum(app_key, lat, lon, start, end, timeout=45):
    params = {
        "appKey": app_key,
        "targets": f"lat={lat:.6f},lng={lon:.6f}",
        "startDate": start,
        "endDate": end,
        "unitOfMeasure": "M",
        "dataItems": "day-asce-eto",
    }
    r = requests.get(API, params=params, headers={"Accept":"application/json"}, timeout=timeout)
    r.raise_for_status()
    payload = r.json()
    vals=[]
    for provider in payload.get("Data",{}).get("Providers",[]):
        for rec in provider.get("Records",[]):
            obj=rec.get("DayAsceEto") or {}
            v=obj.get("Value")
            if v not in (None, ""):
                vals.append(float(v))
    if not vals:
        return np.nan
    return float(np.sum(vals))


def gridmet_sum(lat, lon, start, end, project=None):
    # filterDate end is exclusive, so advance one day.
    end_excl=(datetime.strptime(end,"%Y-%m-%d")+timedelta(days=1)).strftime("%Y-%m-%d")
    img=GRIDMET.filterDate(start,end_excl).select("eto").sum()
    val=img.reduceRegion(
        reducer=ee.Reducer.first(),
        geometry=ee.Geometry.Point([lon,lat]),
        scale=4638.3,
        maxPixels=1_000_000,
    ).get("eto").getInfo()
    return float(val) if val is not None else np.nan


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("candidate_csv")
    ap.add_argument("--project", default=os.getenv("EE_PROJECT"))
    ap.add_argument("--app-key", default=os.getenv("CIMIS_APP_KEY"))
    ap.add_argument("--years", default="2015-2024")
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--growing-days", type=int, default=60)
    ap.add_argument("--out", default="cimis_validation.csv")
    args=ap.parse_args()
    if not args.app_key:
        raise SystemExit("Set CIMIS_APP_KEY or pass --app-key. CIMIS registration/AppKey is required.")
    initialize_ee(args.project)
    a,b=map(int,args.years.split("-")); years=range(a,b+1)
    df=pd.read_csv(args.candidate_csv).head(args.n)
    rows=[]
    for idx,row in df.iterrows():
        lat=float(row["latitude"]); lon=float(row["longitude"]); doy=int(round(row["planting_doy"]))
        for year in years:
            start,end=dates_for_doy(year,doy,args.growing_days)
            try:
                c=cimis_sum(args.app_key,lat,lon,start,end)
            except Exception as exc:
                print(f"CIMIS failed candidate {idx}, {year}: {exc}")
                c=np.nan
            try:
                g=gridmet_sum(lat,lon,start,end,args.project)
            except Exception as exc:
                print(f"gridMET failed candidate {idx}, {year}: {exc}")
                g=np.nan
            rows.append(dict(candidate=idx,latitude=lat,longitude=lon,planting_doy=doy,year=year,
                             start=start,end=end,cimis_eto_mm=c,gridmet_eto_mm=g,
                             difference_mm=(g-c) if np.isfinite(c) and np.isfinite(g) else np.nan))
            time.sleep(0.05)
    out=pd.DataFrame(rows)
    out.to_csv(args.out,index=False)
    valid=out.dropna(subset=["cimis_eto_mm","gridmet_eto_mm"])
    if len(valid):
        bias=(valid.gridmet_eto_mm-valid.cimis_eto_mm).mean()
        rmse=np.sqrt(((valid.gridmet_eto_mm-valid.cimis_eto_mm)**2).mean())
        corr=valid[["gridmet_eto_mm","cimis_eto_mm"]].corr().iloc[0,1]
        metrics=pd.DataFrame([{"n":len(valid),"bias_mm":bias,"rmse_mm":rmse,"correlation":corr}])
        metrics.to_csv(Path(args.out).with_name("cimis_validation_metrics.csv"),index=False)
        fig, ax = plt.subplots(figsize=(6.5, 6.0))
        ax.scatter(valid.cimis_eto_mm, valid.gridmet_eto_mm, s=20, alpha=0.65)
        lo = float(min(valid.cimis_eto_mm.min(), valid.gridmet_eto_mm.min()))
        hi = float(max(valid.cimis_eto_mm.max(), valid.gridmet_eto_mm.max()))
        ax.plot([lo, hi], [lo, hi], linestyle="--")
        ax.set_xlabel("Spatial CIMIS ASCE ETo (mm / growing window)")
        ax.set_ylabel("gridMET grass-reference ETo (mm / growing window)")
        ax.set_title("ETo validation at candidate locations")
        ax.text(0.03, 0.97, f"Bias = {bias:.1f} mm\nRMSE = {rmse:.1f} mm\nr = {corr:.3f}",
                transform=ax.transAxes, va="top")
        fig.tight_layout()
        fig.savefig(Path(args.out).with_name("cimis_validation_scatter.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(metrics.to_string(index=False))
    print(f"Wrote {args.out}")

if __name__=="__main__": main()
