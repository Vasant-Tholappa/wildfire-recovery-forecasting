"""Materialize the approved in-memory California sample, entirely offline."""
import hashlib
import json
from pathlib import Path
import zipfile
import geopandas as gpd
import numpy as np
import pandas as pd
from manual_planb_eecu_forecast import scenarios

DATA = Path(__file__).resolve().parent / "data/processed"
SEED = "ca_portfolio_175_v1"
IDS_HASH = "3fc06786066229a08fbd63016826f6fd49beac1c6b43fbf09989b92e4bc5659f"


def select_ca(manifest):
    ca = manifest[manifest.fire_id.str.startswith("CA")].copy()
    ca["size_band"] = pd.cut(ca.fire_size_acres, [0,5000,20000,100000,np.inf], right=False,
        labels=["under5k","5k-20k","20k-100k","100kplus"]).astype(str)
    ca["rank"] = ca.fire_id.map(lambda x: hashlib.sha256((SEED+":"+x).encode()).hexdigest())
    groups = list(ca.groupby(["fire_year","size_band"], sort=True))
    counts = np.array([len(g) for _,g in groups])
    ideal = 175 * counts / len(ca)
    q = np.maximum(1, np.floor(ideal).astype(int))
    while q.sum() < 175:
        i = max((i for i in range(len(q)) if q[i]<counts[i]), key=lambda i:(ideal[i]-q[i],-i))
        q[i] += 1
    while q.sum() > 175:
        i = max((i for i in range(len(q)) if q[i]>1), key=lambda i:(q[i]-ideal[i],-i))
        q[i] -= 1
    selected = pd.concat([g.sort_values(["rank","fire_id"]).head(int(n)) for (_,g),n in zip(groups,q)])
    return selected.sort_values("fire_id")


def main():
    source = DATA/"fire_manifest.parquet"
    occupancy = DATA/"phase3_planb_tile120_batch_patch_counts.parquet"
    forecast = DATA/"phase3_planb_eecu_forecast_combined_daymet_candidate_a.json"
    m = gpd.read_parquet(source)
    selected = select_ca(m)
    ids = selected.fire_id.tolist()
    assert len(ids)==175 and hashlib.sha256("\n".join(ids).encode()).hexdigest()==IDS_HASH
    assert selected.split.value_counts().to_dict()=={"train":91,"validation":43,"test":41}
    frozen = m.set_index("fire_id").loc[ids].reset_index()
    b = pd.read_parquet(occupancy)
    b = b[b.fire_id.isin(ids)].sort_values(["fire_id","batch_id"])
    assert len(b)==729 and b.patch_count.sum()==2334795
    f=json.loads(forecast.read_text()); c=np.zeros(len(b)); p=c.copy()
    for prod in f["products"]:
        v=scenarios(b.patch_count.to_numpy(float), np.array(prod["measured_seconds"]))
        c+=v["piecewise_power"]/3600; p+=v["small_batch_floor"]/3600
    report={"seed":SEED,"selected_ids_sha256":IDS_HASH,"selected_fire_ids":ids,
        "production_version":"phase3_ca175_v1","fire_count":175,"spatial_batches":729,
        "candidate_patches":2334795,"export_tasks":8748,
        "year_counts":selected.fire_year.value_counts().sort_index().to_dict(),
        "split_counts":selected.split.value_counts().to_dict(),
        "central_hours":float(c.sum()),"planning_hours_with_20pct_retries":float(np.maximum(c,p).sum()*1.2),
        "inputs_sha256":{x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in (source,occupancy,forecast)}}
    stem=DATA/"phase3_ca175_population_v1"
    frozen.to_parquet(stem.with_suffix(".parquet"),index=False)
    b.to_parquet(DATA/"phase3_ca175_batches_v1.parquet",index=False)
    selected.drop(columns="geometry").to_csv(str(stem)+"_selection.csv",index=False)
    report["manifest_sha256"]=hashlib.sha256(stem.with_suffix(".parquet").read_bytes()).hexdigest()
    stem.with_suffix(".json").write_text(json.dumps(report,indent=2)+"\n")
    files=[stem.with_suffix(".parquet"),stem.with_suffix(".json"),DATA/"phase3_ca175_batches_v1.parquet",Path(str(stem)+"_selection.csv"),forecast]
    archive=DATA/"phase3_ca175_launch_inputs_v1.zip"
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for path in files:z.write(path,arcname=path.name)
    record={**report,"files_sha256":{x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in files},
        "archive_sha256":hashlib.sha256(archive.read_bytes()).hexdigest()}
    Path("notes/phase3_ca175_inputs_v1.json").write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("selected_fire_ids","inputs_sha256")},indent=2))
    print(b.iloc[(b.patch_count-500).abs().argsort()[:1]].to_string(index=False))
    print(b.iloc[(b.patch_count-3600).abs().argsort()[:1]].to_string(index=False))
    print(b.sort_values(["patch_count","fire_id","batch_id"],ascending=[False,True,True]).head(1).to_string(index=False))

if __name__=="__main__":main()
