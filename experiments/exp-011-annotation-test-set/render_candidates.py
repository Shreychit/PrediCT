"""Render contact sheets (3 axial slices per scan, mediastinal window) for shortlisted
candidates, so pericardium visibility can be judged by eye before selection (exp-011)."""
import csv
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "COCA_dataset"
HERE = Path(__file__).parent
SEED = 2026

rows = list(csv.DictReader(open(HERE / "candidates.csv")))
for r in rows:
    r["epi"] = float(r["epi_mL"]); r["ratio"] = float(r["peri_heart_ratio"]); r["n"] = int(r["n_slices"])
std = [r for r in rows if r["epi_flag"] == "ok" and 40 <= r["n"] <= 60 and r["hard_case"] == "False"]
epi = np.array([r["epi"] for r in rows])
q = lambda p: float(np.percentile(epi, p))

rng = random.Random(SEED)
groups = {
    "typical_low": rng.sample([r for r in std if q(15) <= r["epi"] < q(35)], 4),
    "typical_median": rng.sample([r for r in std if q(40) <= r["epi"] < q(60)], 4),
    "typical_high": rng.sample([r for r in std if q(65) <= r["epi"] < q(85)], 4),
    "difficult_lowfat": [r for r in rows if r["hard_case"] == "True"]
        + rng.sample([r for r in std if r["epi"] < q(10)], 6),
    "difficult_highratio": sorted(std, key=lambda r: r["ratio"])[-4:],
}

def slices(sid):
    ct = nib.load(DATA / "data_resampled" / sid / f"{sid}_img.nii.gz").get_fdata()
    hm = np.asarray(nib.load(DATA / "heart_masks" / sid / "heart.nii.gz").dataobj) > 0
    xs, ys, zs = np.where(hm)
    m = 30
    x0, x1 = max(xs.min() - m, 0), min(xs.max() + m, ct.shape[0])
    y0, y1 = max(ys.min() - m, 0), min(ys.max() + m, ct.shape[1])
    out = []
    for f in (0.3, 0.5, 0.7):
        z = int(zs.min() + f * (zs.max() - zs.min()))
        out.append((z, np.flipud(np.rot90(ct[x0:x1, y0:y1, z]))))
    return out

for name, grp in groups.items():
    fig, axes = plt.subplots(len(grp), 3, figsize=(10, 3.3 * len(grp)))
    for i, r in enumerate(grp):
        for j, (z, img) in enumerate(slices(r["scan_id"])):
            ax = axes[i, j]
            ax.imshow(img, cmap="gray", vmin=-160, vmax=240)  # mediastinal window L40 W400
            ax.set_axis_off()
            ax.set_title(f"{r['scan_id']}  z={z}" + (f"  epi={r['epi']:.0f}mL ratio={r['ratio']:.2f}" if j == 0 else ""), fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "figures" / f"candidates_{name}.png", dpi=70)
    plt.close(fig)
    print(name, [r["scan_id"] for r in grp])
