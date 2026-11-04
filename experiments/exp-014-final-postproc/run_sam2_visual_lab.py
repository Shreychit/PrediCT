"""exp-014 step A: SAM2 pericardium on the 20 Visual Lab crops, same pipeline as COCA (exp-005).

Inference only — Visual Lab is never used for training (constraint #3). Pixel parameters are converted
from the COCA values (0.7 mm) to each patient's spacing, so the physical prompt is the same.
Output: data/visual_lab_nifti/sam2_pericardium/VL_<p>.nii.gz on the nnU-Net crop grid.
"""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml

HERE = Path(__file__).parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from src.data.visual_lab import crop_slices  # noqa: E402
from src.inference.sam2_predictor import load_sam2_predictor, predict_pericardium_volume  # noqa: E402

cfg = yaml.safe_load(open(HERE / "config.yaml"))["data"]["sam2"]
VL = REPO / "data/visual_lab_nifti"
OUT = VL / "sam2_pericardium"
OUT.mkdir(exist_ok=True)

predictor = load_sam2_predictor(model_id=cfg["model"], device="mps")
for hdir in sorted((VL / "heart_masks").iterdir()):
    p = hdir.name
    out = OUT / f"VL_{p}.nii.gz"
    if out.exists():
        continue
    crop = nib.load(VL / "nnunet_input" / f"VL_{p}_0000.nii.gz")
    sl = crop_slices(p)
    heart = np.asarray(nib.load(hdir / "heart.nii.gz").dataobj).astype(np.uint8)[sl]
    sp = float(crop.header.get_zooms()[0])
    peri = predict_pericardium_volume(
        np.asarray(crop.dataobj).astype(np.float32), heart, predictor,
        box_pad=int(round(cfg["box_pad_mm"] / sp)), min_heart_vox=int(round(cfg["min_heart_vox_mm2"] / sp**2)),
        win_lo=cfg["window"][0], win_hi=cfg["window"][1],
    )
    nib.save(nib.Nifti1Image(peri, crop.affine), out)
    print(p, f"{int((peri.sum((0, 1)) > 0).sum())}/{crop.shape[2]} slices", flush=True)
