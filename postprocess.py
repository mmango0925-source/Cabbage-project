#!/usr/bin/env python3
"""Local post-processing: publication maps, Jaccard sensitivity, and summary tables."""
from __future__ import annotations
import argparse, glob
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt

BANDS = {
    "objective":1, "mean_score":2, "sd_score":3, "cv":4, "planting_doy":5,
    "g_mu_temp":6, "g_mu_heat":7, "g_mu_water":8, "g_mu_ph":9,
    "g_mu_texture":10, "g_mu_soc":11, "robustness":12,
}


def read_band(path, band):
    with rasterio.open(path) as ds:
        arr=ds.read(BANDS[band],masked=True).astype("float64")
        transform=ds.transform; crs=ds.crs
    return arr,transform,crs


def top_mask(arr, top_percent=5.0):
    vals=arr.compressed()
    if not len(vals): return np.zeros(arr.shape,dtype=bool)
    t=np.nanpercentile(vals,100-top_percent)
    return (~np.ma.getmaskarray(arr)) & (np.asarray(arr)>=t)


def jaccard(a,b):
    inter=np.logical_and(a,b).sum(); union=np.logical_or(a,b).sum()
    return inter/union if union else np.nan


def plot_band(path, band, title, out):
    arr,transform,crs=read_band(path,band)
    left=transform.c; top=transform.f
    right=left+transform.a*arr.shape[1]
    bottom=top+transform.e*arr.shape[0]
    fig,ax=plt.subplots(figsize=(7.2,8.8))
    im=ax.imshow(arr,extent=[left,right,bottom,top],origin="upper")
    ax.set_title(title); ax.set_xlabel(f"Easting ({crs})"); ax.set_ylabel("Northing")
    cb=fig.colorbar(im,ax=ax,shrink=0.8); cb.set_label(band)
    fig.tight_layout(); fig.savefig(out,dpi=300,bbox_inches="tight"); plt.close(fig)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("baseline_tif")
    ap.add_argument("--sensitivity-dir",default=None)
    ap.add_argument("--candidate-csv",default=None)
    ap.add_argument("--outdir",default="analysis_outputs")
    ap.add_argument("--top-percent",type=float,default=5.0)
    args=ap.parse_args()
    out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    for band,title in [
        ("objective","Risk-adjusted Napa cabbage suitability"),
        ("planting_doy","Optimal planting day of year"),
        ("robustness","Weight robustness"),
        ("cv","Interannual coefficient of variation"),
    ]:
        plot_band(args.baseline_tif,band,title,out/f"map_{band}.png")

    base,_t,_c=read_band(args.baseline_tif,"objective")
    bm=top_mask(base,args.top_percent)
    rows=[]
    if args.sensitivity_dir:
        for path in sorted(glob.glob(str(Path(args.sensitivity_dir)/"*.tif"))):
            try:
                a,_,_=read_band(path,"objective")
                rows.append({"scenario":Path(path).stem,"jaccard_top_region":jaccard(bm,top_mask(a,args.top_percent))})
            except Exception as exc:
                print(f"Skipping {path}: {exc}")
    if rows:
        sdf=pd.DataFrame(rows).sort_values("jaccard_top_region")
        sdf.to_csv(out/"sensitivity_jaccard.csv",index=False)
        fig,ax=plt.subplots(figsize=(9,max(4,0.3*len(sdf))))
        ax.barh(sdf.scenario,sdf.jaccard_top_region)
        ax.set_xlim(0,1); ax.set_xlabel("Jaccard index vs baseline top region")
        fig.tight_layout(); fig.savefig(out/"sensitivity_jaccard.png",dpi=300,bbox_inches="tight"); plt.close(fig)

    if args.candidate_csv:
        c=pd.read_csv(args.candidate_csv)
        keep=[x for x in ["longitude","latitude","objective","mean_score","sd_score","cv","planting_doy","robustness"] if x in c.columns]
        c[keep].head(30).to_csv(out/"top30_candidates.csv",index=False)
    print(f"Wrote analysis outputs to {out}")

if __name__=="__main__": main()
