# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Experiment 013: Bound epi/para with TotalSegmentator cavity masks
#
# **Started:** 2026-10-28
# **Status:** Completed
#
# exp-012 fixed the Visual Lab GT registration: epi Dice 0.713 (recall 0.79, precision 0.66), para 0.319
# (recall 0.75, precision 0.21). Both errors are precision errors at an anatomical boundary — epi spills past
# the pericardium, para includes chest-wall fat outside the mediastinum. This experiment clips the
# **unchanged** nnU-Net predictions with TotalSegmentator `trunk_cavities` masks (pericardium, mediastinum).
# Steps 1–2 of the Visual Lab improvement plan in `docs/progress.md`.

# %% [markdown]
# ## Question
#
# Does bounding epi by the TS pericardium and para by the TS mediastinum raise strict Dice on Visual Lab, and
# do the same operations give plausible volumes on COCA?
#
# ## Hypothesis
#
# Epi precision rises with little recall loss (Dice ~0.75–0.80); para precision rises a lot (Dice ~0.45–0.60).
# It would surprise us if the TS pericardium excluded much of the expert's epicardial fat — that would mean TS
# draws the sac tighter than Rodrigues's annotators.
#
# **Guardrail:** the five variants in `config.yaml` were fixed before any Visual Lab scoring. All are reported;
# none is tuned on Visual Lab (constraint #3).

# %% [markdown]
# ## Setup

# %%
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import distance_transform_edt

EXPERIMENT_DIR = Path(__file__).parent if "__file__" in dir() else Path.cwd()
REPO = EXPERIMENT_DIR.parents[1]
sys.path.insert(0, str(REPO))
from src.evaluation.metrics import dice, recall_precision  # noqa: E402

config = yaml.safe_load(open(EXPERIMENT_DIR / "config.yaml"))
FIGURES_DIR = EXPERIMENT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
VL = REPO / "data/visual_lab_nifti"
COCA = REPO / "data/COCA_dataset"
TS = REPO / "data/ts_cavities"
LO, HI = config["data"]["hu_range"]
VARIANTS = list(config["variants"])
print(f"Experiment: {config['experiment']['id']} — {config['experiment']['name']}; variants: {VARIANTS}")


# %% [markdown]
# ## Variants
#
# Implemented exactly as written in `config.yaml`.

# %%
def apply_variants(pred, ct, P, M, zooms):
    pe, pp, fat = pred == 1, pred == 2, (ct >= LO) & (ct <= HI)
    shell = distance_transform_edt(~P, sampling=zooms) <= config["data"]["shell_mm"]
    out = {
        "V0_baseline": (pe, pp),
        "V1_para_clip": (pe, pp & M),
        "V2A_bounded": (pe & P, (pp | pe) & M & ~P),
        "V2B_fat_in_P": (fat & P, (pp | pe) & M & ~P),
        "V3_ts_only": (fat & P, fat & M & ~P),
        "V4_shell10": (fat & P, (pp | pe) & (M | shell) & ~P),
    }
    assert list(out) == VARIANTS
    return out


def load(path):
    return np.asarray(nib.load(path).dataobj)


def load_ts(set_, id_, sl=None):
    P = load(TS / set_ / id_ / "pericardium.nii.gz").astype(bool)
    M = load(TS / set_ / id_ / "mediastinum.nii.gz").astype(bool)
    return (P[sl], M[sl]) if sl else (P, M)


# %% [markdown]
# ## Data — Visual Lab (20 patients)
#
# TS ran on the full 512² CT; predictions and registered GT live on the nnU-Net crop. The crop offset is
# recovered from the affines exactly as exp-012 did.

# %%
patients = sorted(p.name for p in (TS / "vl").iterdir())
vl = {}
for p in patients:
    full = nib.load(VL / f"{p}.nii.gz")
    crop = nib.load(VL / "nnunet_input" / f"VL_{p}_0000.nii.gz")
    off = np.round(np.linalg.inv(full.affine[:3, :3]) @ (crop.affine[:3, 3] - full.affine[:3, 3])).astype(int)
    sl = tuple(slice(off[i], off[i] + crop.shape[i]) for i in range(3))
    P, M = load_ts("vl", p, sl)
    vl[p] = dict(ct=np.asarray(crop.dataobj), pred=load(VL / "nnunet_pred" / f"VL_{p}.nii.gz"),
                 gt=load(VL / "nnunet_gt_registered" / f"VL_{p}.nii.gz"), P=P, M=M,
                 zooms=crop.header.get_zooms()[:3], vox_mL=float(np.prod(crop.header.get_zooms()[:3])) / 1000)
print(len(vl), "patients loaded")

# %% [markdown]
# ### Diagnostic: how well do the TS cavities cover the expert labels?
#
# Explains the results below; not used to choose anything.

# %%
cov = pd.DataFrame({p: {
    "gt_epi_in_P": (d["gt"] == 1)[d["P"]].sum() / max((d["gt"] == 1).sum(), 1),
    "gt_para_in_M": (d["gt"] == 2)[d["M"]].sum() / max((d["gt"] == 2).sum(), 1),
    "gt_para_in_P": (d["gt"] == 2)[d["P"]].sum() / max((d["gt"] == 2).sum(), 1),
    "gt_para_in_P|M": (d["gt"] == 2)[d["P"] | d["M"]].sum() / max((d["gt"] == 2).sum(), 1),
} for p, d in vl.items()}).T
print(cov.describe().loc[["mean", "50%", "min"]].round(3))

# %% [markdown]
# ## Run + Evaluation — Visual Lab

# %%
rows = []
for p, d in vl.items():
    ge, gp = d["gt"] == 1, d["gt"] == 2
    for v, (e, q) in apply_variants(d["pred"], d["ct"], d["P"], d["M"], d["zooms"]).items():
        re, pe = recall_precision(e, ge)
        rq, pq = recall_precision(q, gp)
        rows.append(dict(patient=p, variant=v, dice_epi=dice(e, ge), dice_para=dice(q, gp),
                         dice_tat=dice(e | q, ge | gp), epi_recall=re, epi_precision=pe,
                         para_recall=rq, para_precision=pq,
                         epi_mL=e.sum() * d["vox_mL"], gt_epi_mL=ge.sum() * d["vox_mL"],
                         para_mL=q.sum() * d["vox_mL"], gt_para_mL=gp.sum() * d["vox_mL"]))
res = pd.DataFrame(rows)
summary = res.groupby("variant", sort=False).mean(numeric_only=True)
summary["epi_dice_median"] = res.groupby("variant", sort=False)["dice_epi"].median()
summary["epi_vol_r"] = res.groupby("variant", sort=False).apply(lambda g: np.corrcoef(g.epi_mL, g.gt_epi_mL)[0, 1])
summary["para_vol_r"] = res.groupby("variant", sort=False).apply(lambda g: np.corrcoef(g.para_mL, g.gt_para_mL)[0, 1])
cols = ["dice_epi", "epi_dice_median", "epi_recall", "epi_precision", "dice_para", "para_recall",
        "para_precision", "dice_tat", "epi_vol_r", "para_vol_r"]
print(summary[cols].round(3).to_string())

# %%
# Per-patient epi Dice, baseline vs bounded variants
print(res.pivot(index="patient", columns="variant", values="dice_epi")[VARIANTS].round(3).to_string())

# %% [markdown]
# ## COCA QC (10 Kettering test-set scans — no labels yet)
#
# Without GT this is a plausibility check: how much does each variant change the volumes, and do the
# overlays look anatomically right? The Kettering labels will score these same scans later.

# %%
ids = pd.read_csv(REPO / config["data"]["coca_qc"]["ids_from"])["scan_id"].tolist()
coca, qrows = {}, []
for c in ids:
    img = nib.load(COCA / "nnunet_input" / f"{c}_0000.nii.gz")
    P, M = load_ts("coca", c)
    d = dict(ct=np.asarray(img.dataobj), pred=load(COCA / "nnunet_pred_ensemble" / f"{c}.nii.gz"), P=P, M=M,
             zooms=img.header.get_zooms()[:3], vox_mL=float(np.prod(img.header.get_zooms()[:3])) / 1000)
    coca[c] = d
    for v, (e, q) in apply_variants(d["pred"], d["ct"], P, M, d["zooms"]).items():
        qrows.append(dict(scan_id=c, variant=v, epi_mL=e.sum() * d["vox_mL"], para_mL=q.sum() * d["vox_mL"]))
qc = pd.DataFrame(qrows)
qc_epi = qc.pivot(index="scan_id", columns="variant", values="epi_mL")[VARIANTS].loc[ids]
qc_para = qc.pivot(index="scan_id", columns="variant", values="para_mL")[VARIANTS].loc[ids]
print("Epi mL\n", qc_epi.round(1).to_string(), "\n\nPara mL\n", qc_para.round(1).to_string())
qc_change = (qc_epi.div(qc_epi["V0_baseline"], axis=0) - 1).median().round(3)
print("\nMedian epi volume change vs baseline:\n", qc_change.to_string())

# %%
metrics = {
    "visual_lab_summary": summary[cols].round(4).to_dict(orient="index"),
    "visual_lab_ts_coverage": cov.mean().round(4).to_dict(),
    "visual_lab_per_patient": res.round(4).to_dict(orient="records"),
    "coca_qc_epi_mL": qc_epi.round(1).to_dict(orient="index"),
    "coca_qc_para_mL": qc_para.round(1).to_dict(orient="index"),
    "coca_qc_median_epi_change_vs_baseline": qc_change.to_dict(),
}
json.dump(metrics, open(EXPERIMENT_DIR / "metrics.json", "w"), indent=2)

# %% [markdown]
# ## Visualization
#
# Fill: red = epi, green = para. Contours: yellow = TS pericardium, cyan = TS mediastinum.

# %%
def show(ax, ct, e, q, P, M, z, title, set_):
    # anterior up: VL crops store anterior along -x, COCA along -y (checked visually)
    f = (lambda a: np.rot90(np.flipud(a[:, :, z].T), -1)) if set_ == "vl" else (lambda a: a[:, :, z].T)
    ax.imshow(f(np.clip((ct + 250) / 500, 0, 1)), cmap="gray")
    rgba = np.zeros(f(e).shape + (4,))
    rgba[f(e)] = (1, 0, 0, .55)
    rgba[f(q)] = (0, 1, 0, .55)
    ax.imshow(rgba)
    ax.contour(f(P), [.5], colors="yellow", linewidths=.7)
    ax.contour(f(M), [.5], colors="cyan", linewidths=.7)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


SHOW = ["V0_baseline", "V2B_fat_in_P", "V4_shell10"]
for set_, data, picks in [("vl", vl, ["ACel", "AXav", "DLag", "FGas"]), ("coca", coca, ids[:4])]:
    fig, axes = plt.subplots(len(picks), 4 if set_ == "vl" else 3, figsize=(12 if set_ == "vl" else 9, 3 * len(picks)))
    for r, k in enumerate(picks):
        d = data[k]
        ref = (d["gt"] == 1) if set_ == "vl" else (d["pred"] == 1)
        z = int(ref.sum((0, 1)).argmax())
        vs = apply_variants(d["pred"], d["ct"], d["P"], d["M"], d["zooms"])
        cols_ = (["GT"] if set_ == "vl" else []) + SHOW
        for j, name in enumerate(cols_):
            e, q = ((d["gt"] == 1), (d["gt"] == 2)) if name == "GT" else vs[name]
            show(axes[r, j], d["ct"], e, q, d["P"], d["M"], z, f"{k} z={z} — {name}", set_)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"overlay_{set_}.png", dpi=110)
    plt.close(fig)

# %%
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
res.boxplot(column="dice_epi", by="variant", ax=ax[0], rot=20)
res.boxplot(column="dice_para", by="variant", ax=ax[1], rot=20)
for a, t in zip(ax, ["Epi Dice", "Para Dice"]):
    a.set_title(t); a.set_xlabel("")
fig.suptitle("Visual Lab, 20 patients, registered GT")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "dice_by_variant.png", dpi=110)
plt.close(fig)

# %% [markdown]
# ## Interpretation
#
# See `README.md` (written from the numbers above) and `docs/results.md` entry 2026-10-28.
