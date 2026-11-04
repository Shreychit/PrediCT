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
# # Experiment 014: Final post-processing round (no retraining)
#
# **Started:** 2026-10-28
# **Status:** Active
#
# exp-013 showed that epi is now capped by the quality of TotalSegmentator's pericardium (it holds only 80% of
# the expert EAT), and that Rodrigues's para is "fat in the mediastinal space", not TS's mediastinum class.
# This is the last post-processing round before any retraining. It tests six ideas from the references:
# pericardium shape regularization (ref1, ref4), a vote over three pericardium estimates, Commandeur's 3×3
# median, a lung-bounded mediastinum (Rodrigues), and a 3-fold + TTA ensemble.

# %% [markdown]
# ## Question
#
# Can better pericardium and mediastinum masks, built without retraining, lift Visual Lab epi above 0.75 and para
# toward 0.65?
#
# ## Hypothesis
#
# Epi 0.74–0.77 (pericardium vote and hull close the 3 mm misses). Para 0.55–0.65 (the lung-bounded mediastinum drops
# chest-wall fat and keeps retrosternal fat).
#
# **Guardrail:** variants and parameters are frozen in `config.yaml`. COCA QC runs first (`RUN_VISUAL_LAB = False`).
# Visual Lab is scored exactly once, and every variant is reported.

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
from scipy.ndimage import binary_fill_holes, distance_transform_edt, label, median_filter
from skimage.morphology import convex_hull_image

EXPERIMENT_DIR = Path(__file__).parent if "__file__" in dir() else Path.cwd()
REPO = EXPERIMENT_DIR.parents[1]
sys.path.insert(0, str(REPO))
from src.data.visual_lab import crop_slices  # noqa: E402
from src.evaluation.metrics import dice, recall_precision  # noqa: E402

RUN_VISUAL_LAB = True   # flipped once, after the COCA QC overlays were checked (see README)

config = yaml.safe_load(open(EXPERIMENT_DIR / "config.yaml"))
D = config["data"]
FIGURES_DIR = EXPERIMENT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
VL = REPO / "data/visual_lab_nifti"
COCA = REPO / "data/COCA_dataset"
TS = REPO / "data/ts_cavities"
LO, HI = D["hu_range"]
VARIANTS = list(config["variants"])
print(f"Experiment: {config['experiment']['id']}; variants: {VARIANTS}; RUN_VISUAL_LAB={RUN_VISUAL_LAB}")


# %% [markdown]
# ## Building blocks
#
# Implemented exactly as described in `config.yaml`.

# %%
def largest_cc(m):
    lab, n = label(m)
    if n <= 1:
        return m.astype(bool)
    return lab == (np.bincount(lab.ravel())[1:].argmax() + 1)


def slice_hull(m):
    out = np.zeros_like(m, dtype=bool)
    for z in range(m.shape[2]):
        if m[:, :, z].any():
            out[:, :, z] = convex_hull_image(m[:, :, z])
    return out


def slice_fill(m):
    out = np.zeros_like(m, dtype=bool)
    for z in range(m.shape[2]):
        out[:, :, z] = binary_fill_holes(m[:, :, z])
    return out


def shell(P, zooms):
    return distance_transform_edt(~P, sampling=zooms) <= D["shell_mm"]


def build(d):
    """Masks for one case. d: ct, pred0, pred, P_ts, M, T, A, heart, P_sam_raw, zooms."""
    A = d["A"]
    fat_raw = (d["ct"] >= LO) & (d["ct"] <= HI)
    fat = fat_raw & ~A
    fat_med = median_filter(fat.astype(np.uint8), size=(*D["median_kernel"], 1)).astype(bool)
    P_ts = d["P_ts"]
    P_hull = slice_fill(largest_cc(P_ts)) & ~A
    P_sam = slice_fill(largest_cc(d["P_sam_raw"])) & ~A
    P_nn = slice_fill(largest_cc((d["pred"] == 1) | d["heart"])) & ~A
    P_vote = (P_hull.astype(int) + P_sam + P_nn) >= 2
    med_lung = slice_hull(d["T"]) & ~d["T"] & ~P_vote & ~A
    fg = (d["pred"] > 0) & ~A
    M, z = d["M"], d["zooms"]
    sh_ts, sh_hull, sh_vote = shell(P_ts, z), shell(P_hull, z), shell(P_vote, z)
    out = {
        # W0/W1 reproduce exp-013 V4 exactly (no abdominal subtraction)
        "W0_v4_fold0": (fat_raw & P_ts, (d["pred0"] > 0) & (M | sh_ts) & ~P_ts),
        "W1_v4_ens": (fat_raw & P_ts, (d["pred"] > 0) & (M | sh_ts) & ~P_ts),
        "W2_hull": (fat & P_hull, fg & (M | sh_hull) & ~P_hull),
        "W3_vote": (fat & P_vote, fg & (M | sh_vote) & ~P_vote),
        "W4_vote_median": (fat_med & P_vote, fg & fat_med & (M | sh_vote) & ~P_vote),
        "W5_lungmed": (fat & P_vote, fat & med_lung),
        "W6_lungmed_shell": (fat & P_vote, fat & (med_lung | sh_vote) & ~P_vote),
    }
    assert list(out) == VARIANTS
    return out, dict(P_ts=P_ts, P_hull=P_hull, P_sam=P_sam, P_nn=P_nn, P_vote=P_vote, med_lung=med_lung)


def arr(path, sl=None):
    a = np.asarray(nib.load(path).dataobj)
    return a[sl] if sl else a


def onto(target, source_path):
    """Crop a larger same-spacing volume onto `target`'s grid using the affines."""
    src = nib.load(source_path)
    off = np.round(np.linalg.inv(src.affine[:3, :3]) @ (target.affine[:3, 3] - src.affine[:3, 3])).astype(int)
    return np.asarray(src.dataobj)[tuple(slice(off[i], off[i] + target.shape[i]) for i in range(3))]


# %% [markdown]
# ## COCA QC (10 Kettering test-set scans — no labels yet)
#
# P_sam comes from the existing COCA SAM2 masks, which are already on the nnU-Net input grid (same ROI crop).

# %%
ids = pd.read_csv(REPO / "experiments/exp-011-annotation-test-set/selection.csv")["scan_id"].tolist()
coca, qrows = {}, []
for c in ids:
    img = nib.load(COCA / "nnunet_input" / f"{c}_0000.nii.gz")
    L = lambda k: arr(TS / "coca" / c / f"{k}.nii.gz").astype(bool)
    d = dict(ct=np.asarray(img.dataobj), pred0=arr(COCA / "nnunet_pred" / f"{c}.nii.gz"),
             pred=arr(COCA / "nnunet_pred_ensemble" / f"{c}.nii.gz"), P_ts=L("pericardium"), M=L("mediastinum"),
             T=L("thoracic_cavity"), A=L("abdominal_cavity"), heart=onto(img, COCA / "heart_masks" / c / "heart.nii.gz").astype(bool),
             P_sam_raw=arr(COCA / "pericardium_masks" / c / "pericardium_roi.nii.gz").astype(bool),
             zooms=img.header.get_zooms()[:3])
    vox = float(np.prod(d["zooms"])) / 1000
    d["out"], d["masks"] = build(d)
    coca[c] = d
    for v, (e, q) in d["out"].items():
        qrows.append(dict(scan_id=c, variant=v, epi_mL=e.sum() * vox, para_mL=q.sum() * vox))
    m = d["masks"]
    qrows.append(dict(scan_id=c, variant="_P_agreement", epi_mL=dice(m["P_hull"], m["P_sam"]),
                      para_mL=dice(m["P_hull"], m["P_nn"])))
qc = pd.DataFrame(qrows)
qc_epi = qc.pivot(index="scan_id", columns="variant", values="epi_mL").loc[ids]
qc_para = qc.pivot(index="scan_id", columns="variant", values="para_mL").loc[ids]
print("Epi mL\n", qc_epi[VARIANTS].round(1).to_string())
print("\nPara mL\n", qc_para[VARIANTS].round(1).to_string())
print("\nPericardium agreement (Dice): P_hull vs P_sam =", qc_epi["_P_agreement"].round(3).tolist(),
      "\n                              P_hull vs P_nn  =", qc_para["_P_agreement"].round(3).tolist())


# %%
def show(ax, ct, e, q, masks, z, title, set_):
    f = (lambda a: np.rot90(np.flipud(a[:, :, z].T), -1)) if set_ == "vl" else (lambda a: a[:, :, z].T)
    ax.imshow(f(np.clip((ct + 250) / 500, 0, 1)), cmap="gray")
    rgba = np.zeros(f(e).shape + (4,))
    rgba[f(e)] = (1, 0, 0, .5)
    rgba[f(q)] = (0, 1, 0, .5)
    ax.imshow(rgba)
    for k, col in [("P_ts", "yellow"), ("P_sam", "magenta"), ("P_nn", "orange"), ("P_vote", "white"),
                   ("med_lung", "cyan")]:
        if f(masks[k]).any():
            ax.contour(f(masks[k]), [.5], colors=col, linewidths=.6)
    ax.set_title(title, fontsize=7)
    ax.axis("off")


def overlay(data, picks, set_, cols, fname):
    fig, axes = plt.subplots(len(picks), len(cols), figsize=(3 * len(cols), 3 * len(picks)), squeeze=False)
    for r, k in enumerate(picks):
        d = data[k]
        ref = (d["gt"] == 1) if "gt" in d else (d["pred"] == 1)
        z = int(ref.sum((0, 1)).argmax())
        for j, name in enumerate(cols):
            e, q = ((d["gt"] == 1), (d["gt"] == 2)) if name == "GT" else d["out"][name]
            show(axes[r, j], d["ct"], e, q, d["masks"], z, f"{k} z={z} — {name}", set_)
    fig.suptitle("contours: yellow P_ts, magenta P_sam, orange P_nn, white P_vote, cyan Med_lung", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / fname, dpi=100)
    plt.close(fig)


overlay(coca, ids[:5], "coca", ["W1_v4_ens", "W3_vote", "W5_lungmed", "W6_lungmed_shell"], "overlay_coca_a.png")
overlay(coca, ids[5:], "coca", ["W1_v4_ens", "W3_vote", "W5_lungmed", "W6_lungmed_shell"], "overlay_coca_b.png")

# %% [markdown]
# ## Visual Lab — scored once (20 patients, registered GT)

# %%
metrics = {"coca_qc_epi_mL": qc_epi.round(2).to_dict(orient="index"),
           "coca_qc_para_mL": qc_para.round(2).to_dict(orient="index")}
if RUN_VISUAL_LAB:
    vl, rows = {}, []
    for p in sorted(x.name for x in (TS / "vl").iterdir()):
        sl = crop_slices(p)
        crop = nib.load(VL / "nnunet_input" / f"VL_{p}_0000.nii.gz")
        L = lambda k: arr(TS / "vl" / p / f"{k}.nii.gz", sl).astype(bool)
        d = dict(ct=np.asarray(crop.dataobj), pred0=arr(VL / "nnunet_pred" / f"VL_{p}.nii.gz"),
                 pred=arr(REPO / D["predictions"]["ensemble"] / f"VL_{p}.nii.gz"),
                 gt=arr(VL / "nnunet_gt_registered" / f"VL_{p}.nii.gz"), P_ts=L("pericardium"), M=L("mediastinum"),
                 T=L("thoracic_cavity"), A=L("abdominal_cavity"), heart=arr(VL / "heart_masks" / p / "heart.nii.gz", sl).astype(bool),
                 P_sam_raw=arr(VL / "sam2_pericardium" / f"VL_{p}.nii.gz").astype(bool),
                 zooms=crop.header.get_zooms()[:3])
        vox = float(np.prod(d["zooms"])) / 1000
        d["out"], d["masks"] = build(d)
        vl[p] = d
        ge, gp = d["gt"] == 1, d["gt"] == 2
        for v, (e, q) in d["out"].items():
            re, pe = recall_precision(e, ge)
            rq, pq = recall_precision(q, gp)
            rows.append(dict(patient=p, variant=v, dice_epi=dice(e, ge), dice_para=dice(q, gp),
                             dice_tat=dice(e | q, ge | gp), epi_recall=re, epi_precision=pe, para_recall=rq,
                             para_precision=pq, epi_mL=e.sum() * vox, gt_epi_mL=ge.sum() * vox,
                             para_mL=q.sum() * vox, gt_para_mL=gp.sum() * vox))
        m = d["masks"]
        rows.append(dict(patient=p, variant="_gt_epi_coverage",
                         **{k: (ge & m[k]).sum() / max(ge.sum(), 1) for k in ["P_ts", "P_hull", "P_sam", "P_nn", "P_vote"]}))
    res = pd.DataFrame(rows)
    cover = res[res.variant == "_gt_epi_coverage"][["P_ts", "P_hull", "P_sam", "P_nn", "P_vote"]].mean()
    res = res[res.variant != "_gt_epi_coverage"].dropna(axis=1, how="all")
    g = res.groupby("variant", sort=False)
    summary = g.mean(numeric_only=True)
    summary["epi_dice_median"] = g["dice_epi"].median()
    summary["epi_vol_r"] = g.apply(lambda x: np.corrcoef(x.epi_mL, x.gt_epi_mL)[0, 1])
    summary["para_vol_r"] = g.apply(lambda x: np.corrcoef(x.para_mL, x.gt_para_mL)[0, 1])
    cols = ["dice_epi", "epi_dice_median", "epi_recall", "epi_precision", "dice_para", "para_recall",
            "para_precision", "dice_tat", "epi_vol_r", "para_vol_r"]
    print(summary[cols].round(3).to_string())
    print("\nFraction of GT epi inside each pericardium estimate:\n", cover.round(3).to_string())
    print("\nPer-patient epi Dice\n", res.pivot(index="patient", columns="variant", values="dice_epi")[VARIANTS].round(3).to_string())
    metrics.update({"visual_lab_summary": summary[cols].round(4).to_dict(orient="index"),
                    "visual_lab_gt_epi_coverage": cover.round(4).to_dict(),
                    "visual_lab_per_patient": res.round(4).to_dict(orient="records")})
    overlay(vl, ["ACel", "AEdu", "DLag", "VMar"], "vl", ["GT", "W1_v4_ens", "W3_vote", "W6_lungmed_shell"], "overlay_vl.png")
json.dump(metrics, open(EXPERIMENT_DIR / "metrics.json", "w"), indent=2)

# %% [markdown]
# ## Interpretation
#
# See `README.md` and the `docs/results.md` entry for exp-014.
