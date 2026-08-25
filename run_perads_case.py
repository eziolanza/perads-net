#!/usr/bin/env python3
"""PERADS.net: run the validated research PE-RADS pipeline on one CT NIfTI.

Input: one contrast-enhanced chest CT in NIfTI format.
Output: embolus mask, RV/LV ratio, PE-RADS grade and a PNG preview.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_closing, binary_dilation, distance_transform_edt, label
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize_3d

from rvlv_v2 import calculate as calculate_rvlv


PROJECT = Path("/media/ezio/Boatta2TB/nnunet_projects")
PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT / "PERADS/models/perads-net/Dataset546_PEArteryPrior/nnUNetTrainer__nnUNetPlans__3d_fullres"
THRESHOLDS = {"central": 0.516, "lobar": 0.358, "segmental": 0.270}
OVERLAP = 0.01


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=os.environ.copy())


def save_crop(image: nib.spatialimages.SpatialImage, slices: tuple[slice, ...], path: Path) -> None:
    start = np.array([part.start for part in slices], dtype=float)
    translation = np.eye(4); translation[:3, 3] = start
    header = image.header.copy()
    nib.save(nib.Nifti1Image(np.asanyarray(image.dataobj)[slices], image.affine @ translation, header), path)


def crop_slices(mask_paths: list[Path], shape: tuple[int, int, int]) -> tuple[slice, ...]:
    union = np.zeros(shape, bool)
    for path in mask_paths:
        image = nib.load(path)
        if image.shape != shape:
            raise ValueError(f"Geometry mismatch: {path}")
        union |= np.asanyarray(image.dataobj) > 0
    points = np.argwhere(union)
    if not len(points):
        raise ValueError("Empty TotalSegmentator lung masks: cannot define Dataset546 thoracic crop.")
    return tuple(slice(int(a), int(b) + 1) for a, b in zip(points.min(0), points.max(0)))


def hierarchy(arteries: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    bridged = binary_closing(arteries, iterations=2)
    components, count = label(bridged)
    if not count:
        raise ValueError("Empty pulmonary artery mask.")
    sizes = np.bincount(components.ravel()); sizes[0] = 0
    bridged = components == sizes.argmax()
    distance = distance_transform_edt(bridged, sampling=spacing)
    centerline = np.argwhere(skeletonize_3d(bridged) > 0)
    if not len(centerline):
        raise ValueError("Pulmonary-artery skeleton is empty.")
    voxels = np.argwhere(bridged)
    closest = cKDTree(centerline * spacing).query(voxels * spacing)[1]
    ratio = distance[tuple(centerline[closest].T)] / distance.max()
    levels = np.where(ratio >= THRESHOLDS["central"], 1, np.where(ratio >= THRESHOLDS["lobar"], 2, np.where(ratio >= THRESHOLDS["segmental"], 3, 4)))
    result = np.zeros(bridged.shape, np.uint8); result[tuple(voxels.T)] = levels
    return result


def create_preview(ct: Path, embolus: Path, rv: Path, lv: Path, rvlv: dict, result: dict, destination: Path) -> None:
    # NIfTI files from the pipeline are not guaranteed to be stored as
    # (x, y, axial-z). Reorient every layer through the affine first so the
    # preview is always a true axial (R-L, A-P) view.
    def canonical_array(path: Path) -> np.ndarray:
        return np.asanyarray(nib.as_closest_canonical(nib.load(path)).dataobj)

    ct_data = canonical_array(ct)
    pe = canonical_array(embolus) > 0
    rv_data = canonical_array(rv) > 0
    lv_data = canonical_array(lv) > 0
    pe_counts = pe.sum(axis=(0, 1))
    if pe.any():
        positive_slices = np.flatnonzero(pe_counts)
        targets = np.linspace(positive_slices[0], positive_slices[-1], 5)
        pe_slices = [int(positive_slices[np.argmin(np.abs(positive_slices - target))]) for target in targets]
    else:
        pe_slices = [ct_data.shape[2] // 2] * 5
    chamber_counts = (rv_data | lv_data).sum(axis=(0, 1))
    rv_z = int(np.argmax(chamber_counts)) if chamber_counts.any() else ct_data.shape[2] // 2
    figure, axes = plt.subplots(2, 3, figsize=(14, 9), facecolor="black")
    figure.suptitle(f"PE-RADS {result['perads_grade']} — {result['anatomic_level'].capitalize()}  |  RV/LV {rvlv['ratio_axial_same_slice']:.2f}", color="white", fontsize=16, weight="bold")
    for number, (axis, z) in enumerate(zip(axes.flat[:5], pe_slices), 1):
        axis.imshow(ct_data[:, :, z].T, cmap="gray", vmin=-160, vmax=240, origin="lower")
        if pe.any(): axis.contour(pe[:, :, z].T, [0.5], colors=["#ef4444"], linewidths=2)
        axis.set_title(f"Embolo {number}/5 · z={z}", color="white"); axis.axis("off")
    rvlv_axis = axes.flat[5]
    rvlv_axis.imshow(ct_data[:, :, rv_z].T, cmap="gray", vmin=-160, vmax=240, origin="lower")
    rvlv_axis.contour(rv_data[:, :, rv_z].T, [0.5], colors=["#ef4444"], linewidths=2)
    rvlv_axis.contour(lv_data[:, :, rv_z].T, [0.5], colors=["#3b82f6"], linewidths=2)
    rvlv_axis.set_title(
        f"RV rosso · LV blu · assiale z={rv_z} · "
        f"RV {result['rv_diameter_mm']:.1f} mm · LV {result['lv_diameter_mm']:.1f} mm",
        color="white",
    ); rvlv_axis.axis("off")
    figure.tight_layout(rect=(0, 0, 1, .95)); figure.savefig(destination, dpi=160, facecolor="black"); plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input chest CT NIfTI (.nii or .nii.gz)")
    parser.add_argument("--output", required=True, type=Path, help="New output directory")
    parser.add_argument("--case-id", help="Optional output identifier; defaults to input filename")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--totalsegmentator",
        default=os.environ.get("PERADS_TOTALSEGMENTATOR", "TotalSegmentator"),
        help="TotalSegmentator executable used for lung_vessels and heartchambers_highres.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    ct_path = args.input.resolve(); output = args.output.resolve()
    case_id = args.case_id or ct_path.name.removesuffix(".nii.gz").removesuffix(".nii")
    if not ct_path.is_file(): parser.error(f"Input not found: {ct_path}")
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        parser.error(f"Output exists and is not empty: {output} (use --overwrite to reuse it)")
    if not args.model.is_dir(): parser.error(f"nnU-Net model not found: {args.model}")
    output.mkdir(parents=True, exist_ok=True)
    ts_lung, ts_heart = output / "01_totalsegmentator/lung_vessels", output / "01_totalsegmentator/heartchambers"
    pre, inference, final = output / "02_preprocessed", output / "03_embolus", output / "04_results"
    raw = nib.load(str(ct_path)); raw_shape = raw.shape
    lung_masks = [ts_lung / name for name in ("lung_airways.nii.gz", "lung_arteries.nii.gz", "lung_veins.nii.gz")]
    ts_heart_marker = output / "01_totalsegmentator/.perads_heartchambers_highres"
    if not all(path.exists() for path in lung_masks):
        ts_device = "gpu" if args.device == "cuda" else "cpu"
        run([args.totalsegmentator, "-i", str(ct_path), "-o", str(ts_lung), "-ta", "lung_vessels", "-d", ts_device])
    rv, lv = ts_heart / "heart_ventricle_right.nii.gz", ts_heart / "heart_ventricle_left.nii.gz"
    if not (rv.exists() and lv.exists()) or not ts_heart_marker.exists():
        ts_device = "gpu" if args.device == "cuda" else "cpu"
        run([
            args.totalsegmentator, "-i", str(ct_path), "-o", str(ts_heart),
            "-ta", "heartchambers_highres", "-d", ts_device,
        ])
        ts_heart_marker.write_text("TotalSegmentator task=heartchambers_highres\n")
    slices = crop_slices(lung_masks, raw_shape)
    images = pre / "imagesTs"; images.mkdir(parents=True, exist_ok=True)
    cropped_ct, cropped_prior, cropped_arteries = images / f"{case_id}_0000.nii.gz", images / f"{case_id}_0001.nii.gz", pre / "lung_arteries.nii.gz"
    save_crop(raw, slices, cropped_ct)
    arteries_raw = nib.load(str(ts_lung / "lung_arteries.nii.gz"))
    prior_raw = binary_dilation(np.asanyarray(arteries_raw.dataobj) > 0, iterations=5).astype(np.uint8)
    prior_image = nib.Nifti1Image(prior_raw, arteries_raw.affine, arteries_raw.header.copy())
    save_crop(prior_image, slices, cropped_prior); save_crop(arteries_raw, slices, cropped_arteries)
    cropped_rv, cropped_lv = pre / "heart_ventricle_right.nii.gz", pre / "heart_ventricle_left.nii.gz"
    save_crop(nib.load(str(rv)), slices, cropped_rv); save_crop(nib.load(str(lv)), slices, cropped_lv)
    inference.mkdir(parents=True, exist_ok=True)
    prediction = inference / f"{case_id}.nii.gz"
    if not prediction.exists() or args.overwrite:
        run(["nnUNetv2_predict_from_modelfolder", "-i", str(images), "-o", str(inference), "-m", str(args.model), "-f", "0", "1", "2", "3", "4", "-device", args.device, "-npp", "1", "-nps", "1"])
    final.mkdir(parents=True, exist_ok=True)
    embolus_path = final / "embolus_segmentation.nii.gz"; shutil.copy2(prediction, embolus_path)
    prediction_image = nib.load(str(prediction)); embolus = np.asanyarray(prediction_image.dataobj) > 0
    labels = hierarchy(np.asanyarray(nib.load(str(cropped_arteries)).dataobj) > 0, np.asarray(prediction_image.header.get_zooms()[:3]))
    hierarchy_path = final / "arterial_hierarchy.nii.gz"; nib.save(nib.Nifti1Image(labels, prediction_image.affine, prediction_image.header), hierarchy_path)
    total = int(embolus.sum()); names = ("central", "lobar", "segmental", "subsegmental")
    fractions = {name: (int((embolus & (labels == level)).sum()) / total if total else 0.0) for level, name in enumerate(names, 1)}
    level = next((level for level, name in enumerate(names, 1) if fractions[name] >= OVERLAP), 0)
    result = {"case_id": case_id, "perads_grade": {0: 0, 1: 4, 2: 3, 3: 2, 4: 1}[level], "anatomic_level": ("none", *names)[level], "embolic_voxels": total, "minimum_embolus_overlap_fraction": OVERLAP, "diameter_ratio_thresholds": THRESHOLDS, **{f"{key}_fraction": round(value, 5) for key, value in fractions.items()}}
    rvlv_dir = final / "rv_lv"
    rvlv = calculate_rvlv(cropped_rv, cropped_lv, case_id, rvlv_dir)
    result.update({"rv_lv_ratio": rvlv["ratio_axial_same_slice"], "rv_diameter_mm": rvlv["rv_chord"]["length_mm"], "lv_diameter_mm": rvlv["lv_chord"]["length_mm"]})
    (final / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    create_preview(cropped_ct, embolus_path, cropped_rv, cropped_lv, rvlv, result, final / "preview.png")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
