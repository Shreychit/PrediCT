"""Build the candidate table for the expert-annotation test set (exp-011).

Pool = COCA scans never used in training or tuning: splits_v1 'unscored' + 'hard_cases',
minus the 9 exp-009 visualisation patients (their predictions were inspected).
Features per scan: predicted epi/para volume (ensemble CSV), SAM2 pericardium volume,
TotalSegmentator heart volume, and their ratio (sac/heart).
"""
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "COCA_dataset"
OUT = Path(__file__).parent / "candidates.csv"

splits = json.load(open(REPO / "data" / "splits_v1.json"))["splits"]
pool = set(splits["unscored"]) | set(splits["hard_cases"])
viz_seen = {p.name.split("_")[0] for p in (REPO / "experiments/exp-009-qualitative-viz/figures").glob("*.png")}
pool -= viz_seen

vols = {r["scan_id"].replace(".nii", ""): r for r in csv.DictReader(open(DATA / "eat_volumes_ensemble.csv"))}
sam2 = {r["scan_id"]: r for r in json.load(open(DATA / "sam2_batch_632.json"))}

rows = []
for sid in sorted(pool):
    if sid not in vols:
        continue
    meta = json.load(open(DATA / "data_canonical/images" / sid / f"{sid}_meta.json"))
    hm = nib.load(DATA / "heart_masks" / sid / "heart.nii.gz")
    heart_ml = float((np.asarray(hm.dataobj) > 0).sum()) * float(np.prod(hm.header.get_zooms()[:3])) / 1000
    peri_ml = sam2.get(sid, {}).get("pericardium_vol_ml")
    rows.append({
        "scan_id": sid,
        "hard_case": sid in splits["hard_cases"],
        "epi_mL": float(vols[sid]["epi_mL"]),
        "para_mL": float(vols[sid]["para_mL"]),
        "epi_flag": vols[sid]["epi_flag"],
        "heart_mL": round(heart_ml, 1),
        "peri_mL": peri_ml,
        "peri_heart_ratio": round(peri_ml / heart_ml, 3) if peri_ml and heart_ml else None,
        "n_slices": len(list(Path(meta["original_path"]).glob("*.dcm"))),
        "dicom_dir": meta["original_path"],
    })

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"{len(rows)} candidates -> {OUT}")
