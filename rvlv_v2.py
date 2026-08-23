"""PERADS.net RV/LV v2: same-slice chamber diameters used in Dataset 120."""
from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, cKDTree


def largest_component(mask: np.ndarray) -> np.ndarray:
    labels, count = ndimage.label(mask)
    if not count:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(labels.ravel()); sizes[0] = 0
    return labels == sizes.argmax()


def longest_internal_chord(mask: np.ndarray, direction: np.ndarray, normal: np.ndarray, spacing: np.ndarray, step_mm: float = 0.5) -> dict:
    component = largest_component(mask)
    points_mm = np.argwhere(component).astype(float) * spacing
    origin = points_mm.mean(axis=0); relative = points_mm - origin
    along, across = relative @ direction, relative @ normal
    best = None
    for offset in np.arange(across.min(), across.max() + step_mm, step_mm):
        locations = origin + np.outer(np.arange(along.min(), along.max() + step_mm, step_mm), direction) + offset * normal
        pixels = locations / spacing
        sampled = ndimage.map_coordinates(component.astype(np.uint8), [pixels[:, 0], pixels[:, 1]], order=0, mode="constant").astype(bool)
        changes = np.diff(np.r_[False, sampled, False].astype(np.int8))
        for start, end in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
            candidate = {"length_mm": float((end - start - 1) * step_mm), "endpoints_px": [pixels[start].tolist(), pixels[end - 1].tolist()]}
            if best is None or candidate["length_mm"] > best["length_mm"]: best = candidate
    if best is None: raise ValueError("Unable to find an internal chamber chord")
    return best


def chamber_axes(mask: np.ndarray, spacing: np.ndarray) -> tuple[dict, dict]:
    component = largest_component(mask)
    boundary = component & ~ndimage.binary_erosion(component)
    points = np.argwhere(boundary).astype(float) * spacing
    if len(points) < 3: raise ValueError("Insufficient chamber boundary points")
    hull = points[ConvexHull(points).vertices]
    first, second = np.unravel_index(np.argmax(np.linalg.norm(hull[:, None] - hull[None, :], axis=2)), (len(hull), len(hull)))
    direction = hull[second] - hull[first]; direction /= np.linalg.norm(direction)
    normal = np.array([-direction[1], direction[0]])
    return longest_internal_chord(component, direction, normal, spacing), longest_internal_chord(component, normal, direction, spacing)


def calculate(rv_path: Path, lv_path: Path, case_id: str, output_dir: Path) -> dict:
    rv_image, lv_image = nib.load(str(rv_path)), nib.load(str(lv_path))
    rv, lv = np.asanyarray(rv_image.dataobj) > 0, np.asanyarray(lv_image.dataobj) > 0
    if rv.shape != lv.shape: raise ValueError("RV and LV masks must have matching geometry")
    spacing = np.asarray(rv_image.header.get_zooms()[:2], dtype=float)
    candidates = []
    for z in range(rv.shape[2]):
        r, l = largest_component(rv[:, :, z]), largest_component(lv[:, :, z])
        if r.any() and l.any(): candidates.append((chamber_axes(r, spacing)[0]["length_mm"], z, r, l))
    if not candidates: raise ValueError("No axial slice contains both ventricles")
    top = sorted(candidates, reverse=True, key=lambda item: item[0])[:max(1, (len(candidates) + 3) // 4)]
    _, z, r, l = min(top, key=lambda item: abs(np.log(float(item[2].sum()) / float(item[3].sum()))))
    rv_boundary, lv_boundary = r & ~ndimage.binary_erosion(r), l & ~ndimage.binary_erosion(l)
    rv_points = np.argwhere(rv_boundary); distances, _ = cKDTree(np.argwhere(lv_boundary) * spacing).query(rv_points * spacing)
    facing = rv_points[distances <= distances.min() + 5]
    centered = facing * spacing - (facing * spacing).mean(0)
    _, _, vectors = np.linalg.svd(centered, full_matrices=False)
    septum = vectors[0]; perpendicular = np.array([-septum[1], septum[0]])
    rv_chord, lv_chord = longest_internal_chord(r, perpendicular, septum, spacing), longest_internal_chord(l, perpendicular, septum, spacing)
    result = {"case_id": case_id, "selected_slice_z": int(z), "rv_voxels": int(r.sum()), "lv_voxels": int(l.sum()), "slice_selection": "top quartile of RV maximum chords, then RV/LV voxel ratio closest to 1", "septal_direction_xy": septum.tolist(), "perpendicular_direction_xy": perpendicular.tolist(), "rv_chord": rv_chord, "lv_chord": lv_chord, "ratio_axial_same_slice": round(rv_chord["length_mm"] / lv_chord["length_mm"], 4)}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rv_lv_v2_same_slice.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
