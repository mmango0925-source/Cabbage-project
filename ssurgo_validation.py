#!/usr/bin/env python3
"""Point-based SSURGO validation of candidate soil properties via USDA Soil Data Access."""
from __future__ import annotations
import argparse, math
from pathlib import Path
import requests
import numpy as np
import pandas as pd

SDA = "https://SDMDataAccess.sc.egov.usda.gov/Tabular/post.rest"


def query_point(lon: float, lat: float):
    # A very small polygon is more robust than an exact boundary point.
    e=0.00005
    wkt=(f"POLYGON(({lon-e} {lat-e},{lon-e} {lat+e},{lon+e} {lat+e},"
         f"{lon+e} {lat-e},{lon-e} {lat-e}))")
    sql=f"""
    SELECT mu.mukey, mu.musym, mu.muname,
           c.cokey, c.compname, c.comppct_r,
           ch.chkey, ch.hzdept_r, ch.hzdepb_r,
           ch.ph1to1h2o_r, ch.sandtotal_r, ch.silttotal_r,
           ch.claytotal_r, ch.om_r
    FROM SDA_Get_Mukey_from_intersection_with_WktWgs84('{wkt}') s
    INNER JOIN mapunit mu ON mu.mukey = s.mukey
    LEFT JOIN component c ON c.mukey = mu.mukey
    LEFT JOIN chorizon ch ON ch.cokey = c.cokey
    WHERE ch.hzdept_r < 30 AND ch.hzdepb_r > 0
    """
    r=requests.post(SDA,data={"service":"query","request":"query","query":sql,"format":"JSON+COLUMNNAME"},timeout=60)
    r.raise_for_status(); payload=r.json()
    table=payload.get("Table")
    if not table or len(table)<2: return pd.DataFrame()
    cols=table[0]
    return pd.DataFrame(table[1:],columns=cols)


def top30_weighted(df: pd.DataFrame):
    if df.empty: return {}
    for c in ["comppct_r","hzdept_r","hzdepb_r","ph1to1h2o_r","sandtotal_r","silttotal_r","claytotal_r","om_r"]:
        df[c]=pd.to_numeric(df[c],errors="coerce")
    # Weight by component proportion and horizon thickness intersecting 0-30 cm.
    top=np.maximum(df.hzdept_r.fillna(0),0)
    bottom=np.minimum(df.hzdepb_r.fillna(0),30)
    thickness=np.maximum(bottom-top,0)
    comp=df.comppct_r.fillna(0)/100.0
    base_w=thickness*comp
    res={}
    for field,name in [("ph1to1h2o_r","ssurgo_ph"),("sandtotal_r","ssurgo_sand_pct"),
                       ("silttotal_r","ssurgo_silt_pct"),("claytotal_r","ssurgo_clay_pct"),
                       ("om_r","ssurgo_organic_matter_pct")]:
        vals=df[field]
        ok=vals.notna() & (base_w>0)
        res[name]=float(np.average(vals[ok],weights=base_w[ok])) if ok.any() else np.nan
    res["ssurgo_mapunits"]=";".join(sorted(set(df.mukey.astype(str))))
    return res


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("candidate_csv")
    ap.add_argument("--n",type=int,default=30)
    ap.add_argument("--out",default="ssurgo_validation.csv")
    args=ap.parse_args()
    df=pd.read_csv(args.candidate_csv).head(args.n)
    rows=[]
    for i,row in df.iterrows():
        lon=float(row.longitude); lat=float(row.latitude)
        try:
            raw=query_point(lon,lat); agg=top30_weighted(raw)
        except Exception as exc:
            print(f"SDA failed candidate {i}: {exc}"); agg={}
        rec={"candidate":i,"longitude":lon,"latitude":lat}
        # preserve SoilGrids component suitability for comparison, if present
        for key in ["g_mu_ph","g_mu_texture","g_mu_soc","objective","robustness",
                    "soil_ph","soil_sand_pct","soil_silt_pct","soil_clay_pct","soil_soc_gkg"]:
            if key in row: rec[key]=row[key]
        rec.update(agg)
        if "soil_ph" in rec and "ssurgo_ph" in rec:
            rec["ph_difference_model_minus_ssurgo"] = rec["soil_ph"] - rec["ssurgo_ph"]
        if "soil_sand_pct" in rec and "ssurgo_sand_pct" in rec:
            rec["sand_difference_pctpt"] = rec["soil_sand_pct"] - rec["ssurgo_sand_pct"]
        if "soil_clay_pct" in rec and "ssurgo_clay_pct" in rec:
            rec["clay_difference_pctpt"] = rec["soil_clay_pct"] - rec["ssurgo_clay_pct"]
        rows.append(rec)
    pd.DataFrame(rows).to_csv(args.out,index=False)
    print(f"Wrote {args.out}")

if __name__=="__main__": main()
