"""exp-012: Register Visual Lab GT to the CT properly, then re-score the UNCHANGED exp-008 predictions.

Finding: the GT BMPs are in Rodrigues 2016's registered frame — each patient rescaled to 0.35 mm pixels
and translated (paper §3.1). The DICOMs keep native spacing (0.26–0.39 mm). exp-008/010 matched slices
only, so the GT was offset in-plane by up to ~30 mm. The BMPs carry the CT image under the colour
labels, so the transform is recovered per patient by matching that image to the DICOM (fat window),
independent of any prediction.
"""
import json
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
from PIL import Image
from scipy.ndimage import zoom

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
DC = REPO / "data/Dicom _ Treino"
GT = REPO / "data/GroundTruth-FatRange/Ground Truth - Fat Range"
VL = REPO / "data/visual_lab_nifti"
GT_NAME = {"FPiq": "FSiq"}
T = 0.35  # Rodrigues common pixel spacing (mm)
N = 512


def fatwin(hu):
    return np.clip((hu + 200) / 170, 0, 1) * 255 * ((hu >= -200) & (hu <= -30))


def read_bmp(f):
    a = np.asarray(Image.open(f).convert("RGB")).astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    gray = (np.abs(r - g) < 8) & (np.abs(g - b) < 8)
    return np.where(gray, r, 0).astype(float), (r > g + 30) & (r > b + 30), (g > r + 30) & (g > b + 30)


def rescale(img, s, fill=-1024.0):
    z = zoom(img, s, order=1)
    m = z.shape[0]
    if m >= N:
        o = (m - N) // 2
        return z[o:o + N, o:o + N], m, -o
    out = np.full((N, N), fill)
    o = (N - m) // 2
    out[o:o + m, o:o + m] = z
    return out, m, o


def best_shift(a, b, maxs=200):
    A, B = np.fft.fft2(a - a.mean()), np.fft.fft2(b - b.mean())
    c = np.fft.fftshift(np.real(np.fft.ifft2(A * np.conj(B))))
    sub = c[N // 2 - maxs:N // 2 + maxs + 1, N // 2 - maxs:N // 2 + maxs + 1]
    i, j = np.unravel_index(sub.argmax(), sub.shape)
    return (int(i - maxs), int(j - maxs)), float(sub.max())


def dice(a, b):
    s = a.sum() + b.sum()
    return 2 * (a & b).sum() / s if s else np.nan


results = {}
for pdir in sorted(DC.iterdir()):
    p = pdir.name
    files = sorted(pdir.glob("*.dcm"))
    dss = [pydicom.dcmread(f) for f in files]
    hu_file = [d.pixel_array * float(d.RescaleSlope) + float(d.RescaleIntercept) for d in dss]
    sp = float(dss[0].PixelSpacing[0])
    bmps = sorted((GT / GT_NAME.get(p, p)).glob("*"))

    # 0+1) choose scale, slice offset and in-plane shift so GT labels land on fat HU — every Rodrigues
    #      label is fat by construction (paper §3), so this is GT/CT-intrinsic and prediction-free.
    nb = min(len(bmps), len(files))
    sample = list(range(0, nb, 3))
    bm = {k: read_bmp(bmps[k]) for k in sample}
    cands = sorted({1.0} | {round(sp / t, 4) for t in np.arange(0.344, 0.3576, 0.0015)})
    best = None
    for sc in cands:
        rs = {j: rescale(hu_file[j], sc)[0] for j in range(len(files))}
        for dz in range(-3, 4):
            ks = [k for k in sample if 0 <= k + dz < len(files)]
            votes = Counter(best_shift(bm[k][0], fatwin(rs[k + dz]))[0] for k in ks)
            sh = votes.most_common(1)[0][0]
            hits = []
            for k in ks:
                lab_k = bm[k][1] | bm[k][2]
                if lab_k.any():
                    h = np.roll(np.roll(rs[k + dz], sh[0], 0), sh[1], 1)
                    hits.append(((h[lab_k] >= -200) & (h[lab_k] <= -30)).mean())
            sc_fit = np.mean(hits) if hits else 0
            if best is None or sc_fit > best[0]:
                best = (sc_fit, sc, dz, sh, votes[sh], len(ks))
    _, s, dzz, (dy, dx), votes, nk = best
    m, o = rescale(hu_file[0], s)[1:]

    # 2) map GT into the native DICOM grid: native i -> rescaled i*(m-1)/(N-1)+o -> GT +shift
    idx = np.arange(N) * (m - 1) / (N - 1) + o
    rows = np.clip(np.round(idx + dy).astype(int), 0, N - 1)
    cols = np.clip(np.round(idx + dx).astype(int), 0, N - 1)
    zpos = [float(d.ImagePositionPatient[2]) for d in dss]
    order = np.argsort(zpos)                      # vol z index -> file index (exp-008 sorts by z)
    file_to_z = {int(fi): zi for zi, fi in enumerate(order)}
    lab = np.zeros((N, N, len(files)), np.uint8)
    lab_old = np.zeros_like(lab)  # baseline: same slice mapping, no in-plane registration (exp-008 style)
    fat_hits = []
    for k, f in enumerate(bmps[: len(files)]):
        if not 0 <= k + dzz < len(files):
            continue
        _, epi, para = read_bmp(f)
        e, q = epi[np.ix_(rows, cols)], para[np.ix_(rows, cols)]
        z = file_to_z[k + dzz]
        lab[..., z][e] = 1; lab[..., z][q] = 2
        lab_old[..., z][epi] = 1; lab_old[..., z][para] = 2
        h = hu_file[k + dzz]
        if (e | q).any():
            fat_hits.append(((h[e | q] >= -200) & (h[e | q] <= -30)).mean())

    # 3) crop exactly as exp-008 did (offset recovered from the crop's affine) and score
    full = nib.load(VL / f"{p}.nii.gz"); crop = nib.load(VL / "nnunet_input" / f"VL_{p}_0000.nii.gz")
    off = np.round(np.linalg.inv(full.affine[:3, :3]) @ (crop.affine[:3, 3] - full.affine[:3, 3])).astype(int)
    sl = tuple(slice(off[i], off[i] + crop.shape[i]) for i in range(3))
    gt_new = lab[sl]
    gt_old = lab_old[sl]
    pred = np.asarray(nib.load(VL / "nnunet_pred" / f"VL_{p}.nii.gz").dataobj)
    results[p] = {
        "spacing_mm": sp, "scale": round(s, 4), "z_offset": dzz, "shift_px": [dy, dx], "shift_votes": f"{votes}/{nk}",
        "gt_fat_at_label": round(float(np.mean(fat_hits)), 3),
        "epi_dice_old": round(dice(pred == 1, gt_old == 1), 3), "epi_dice_new": round(dice(pred == 1, gt_new == 1), 3),
        "para_dice_old": round(dice(pred == 2, gt_old == 2), 3), "para_dice_new": round(dice(pred == 2, gt_new == 2), 3),
        "gt_epi_mL_new": round(float((gt_new == 1).sum() * np.prod(full.header.get_zooms()[:3]) / 1000), 1),
        "pred_epi_mL": round(float((pred == 1).sum() * np.prod(full.header.get_zooms()[:3]) / 1000), 1),
    }
    nib.save(nib.Nifti1Image(gt_new.astype(np.int16), crop.affine), VL / "nnunet_gt_registered" / f"VL_{p}.nii.gz") if (VL / "nnunet_gt_registered").exists() else None
    print(p, results[p])

keys = ["gt_fat_at_label", "epi_dice_old", "epi_dice_new", "para_dice_old", "para_dice_new"]
summary = {k: round(float(np.nanmean([r[k] for r in results.values()])), 3) for k in keys}
print("MEAN", summary)
json.dump({"summary": summary, "per_patient": results}, open(HERE / "metrics.json", "w"), indent=2)
