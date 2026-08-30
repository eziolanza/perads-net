#!/usr/bin/env python3
"""PERADS.net -- end-to-end PE-RADS classification pipeline.

Raw, uncropped chest CT in -> PE-RADS grade + named anatomic site + RV/LV
ratio out, in one process, one CSV row.

Steps: TotalSegmentator (lung_vessels, heartchambers_highres, total -rs
5-lobes) on the raw CT -> crop once for nnU-Net -> nnU-Net 5-fold embolus
segmentation (hard-mask, 30mm3 minimum volume) -> named-artery branch-tree
PE-RADS classification (main pulmonary trunk / right or left PA / lobar /
segmental / subsegmental, per lung lobe) -> four-chamber RV/LV ratio ->
result.json + structure_labelmap.nii.gz + one CSV row.

Every TotalSegmentator task runs on the SAME raw CT, so the arteries, lobe
and heart-chamber masks are all natively co-registered with each other and
with the CT -- only the small-crop embolus prediction needs one alignment
step back to full-resolution space.

The nnU-Net model directory and its scale (crop size, resolution) are
independent of this pipeline; point --model / PERADS_NET_MODEL at whichever
trained model directory matches the artery-prior-conditioned segmentation
task described in the accompanying publication.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path

import networkx as nx
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_closing, binary_dilation, distance_transform_edt, label
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize_3d
from skan.csr import Skeleton, summarize

from rvlv_v3_four_chamber import calculate as calculate_rvlv

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

DEFAULT_MODEL = Path(os.environ.get(
    "PERADS_NET_MODEL",
    "models/Dataset547_PEArteryPriorCrop/nnUNetTrainer__nnUNetPlans__3d_fullres",
))

MIN_EMBOLUS_VOLUME_MM3 = 30.0
OVERLAP = 0.01
MIN_BRANCH_LENGTH_MM = 3.0
LOBE_DILATION_ITERATIONS = 3
PE_CLASS_NAMES = {4: "main_right_left", 3: "lobar", 2: "segmental", 1: "subsegmental"}

LOBES_RIGHT = ["lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"]
LOBES_LEFT = ["lung_upper_lobe_left", "lung_lower_lobe_left"]
LOBE_TO_SHORT = {
    "lung_upper_lobe_right": "RUL", "lung_middle_lobe_right": "RML",
    "lung_lower_lobe_right": "RLL", "lung_upper_lobe_left": "LUL",
    "lung_lower_lobe_left": "LLL",
}
LOBE_SHORT_NAMES = list(LOBE_TO_SHORT.values())

STRUCTURE_CODES = {"main_trunk": 1, "RPA": 2, "LPA": 3}
_next_code = 4
for _lobe in LOBE_SHORT_NAMES:
    for _suffix in ("artery", "segmental", "subsegmental"):
        STRUCTURE_CODES[f"{_lobe}_{_suffix}"] = _next_code
        _next_code += 1
STRUCTURE_CODES["interlobar"] = _next_code

# Every named structure's standard PE-RADS proximity class (4=most proximal
# ... 1=most peripheral), for grading only -- the labelmap itself keeps the
# richer named-structure codes above.
PERADS_CLASS = {"main_trunk": 4, "RPA": 4, "LPA": 4, "interlobar": 3}
for _lobe in LOBE_SHORT_NAMES:
    PERADS_CLASS[f"{_lobe}_artery"] = 3
    PERADS_CLASS[f"{_lobe}_segmental"] = 2
    PERADS_CLASS[f"{_lobe}_subsegmental"] = 1


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=os.environ.copy())


# --------------------------------------------------------------------------
# Small NIfTI / geometry utilities
# --------------------------------------------------------------------------

def load_binary_mask(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj) > 0
    return data, img.affine, np.array(img.header.get_zooms()[:3])


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    components, count = label(mask)
    if not count:
        raise ValueError("Empty mask")
    sizes = np.bincount(components.ravel()); sizes[0] = 0
    return components == sizes.argmax()


def align_embolus_to_arteries(embolus_path: Path, arteries_shape: tuple,
                              arteries_affine: np.ndarray) -> np.ndarray:
    """Place the (small-crop) embolus prediction into the full-res arteries grid.

    nnU-Net runs on a tight crop for speed, so its output rarely matches the
    full-res arteries volume's shape. Recovers the voxel offset from the two
    affines (a pure-translation crop, same rotation/spacing) and pastes the
    prediction into a full-size array aligned with `arteries`.
    """
    img = nib.load(str(embolus_path))
    data = np.asanyarray(img.dataobj) > 0
    if data.shape == arteries_shape and np.allclose(img.affine, arteries_affine, atol=1e-2):
        return data

    transform = np.linalg.inv(arteries_affine) @ img.affine
    if not np.allclose(transform[:3, :3], np.eye(3), atol=1e-3):
        raise ValueError("Embolus prediction is rotated relative to arteries mask; cannot align by translation only")

    offset = np.round(transform[:3, 3]).astype(int)
    hi = offset + np.array(data.shape)
    if np.any(offset < 0) or np.any(hi > np.array(arteries_shape)):
        raise ValueError(f"Embolus prediction crop [{offset}:{hi}] falls outside arteries volume {arteries_shape}")

    full = np.zeros(arteries_shape, dtype=bool)
    full[offset[0]:hi[0], offset[1]:hi[1], offset[2]:hi[2]] = data
    return full


def save_crop(image: nib.spatialimages.SpatialImage, slices: tuple[slice, ...], path: Path) -> None:
    start = np.array([part.start for part in slices], dtype=float)
    translation = np.eye(4); translation[:3, 3] = start
    header = image.header.copy()
    nib.save(nib.Nifti1Image(np.asanyarray(image.dataobj)[slices], image.affine @ translation, header), path)


def crop_slices(mask_paths: list[Path], shape: tuple[int, int, int]) -> tuple[slice, ...]:
    union = np.zeros(shape, bool)
    for path in mask_paths:
        image = nib.load(str(path))
        if image.shape != shape:
            raise ValueError(f"Geometry mismatch: {path}")
        union |= np.asanyarray(image.dataobj) > 0
    points = np.argwhere(union)
    if not len(points):
        raise ValueError("Empty TotalSegmentator lung masks: cannot define the crop.")
    return tuple(slice(int(a), int(b) + 1) for a, b in zip(points.min(0), points.max(0)))


# --------------------------------------------------------------------------
# Branch-tree construction (skan) -- unchanged from PERADS.net C, this
# session's proven, debugged version.
# --------------------------------------------------------------------------

def build_branch_tree(arteries: np.ndarray, spacing: np.ndarray,
                      min_branch_length_mm: float = MIN_BRANCH_LENGTH_MM) -> nx.Graph:
    """skan-derived branch-level graph: one edge per vessel segment.

    Edge attrs: length_mm, radius_mean/min/max_mm, num_points, path_voxels
    (Nx3 int array, voxel-index space), node_src/node_dst (skan's original
    endpoint order, since G is undirected and orientation is only decided
    later once a root is picked).
    """
    bridged = binary_closing(arteries, iterations=2)
    components, count = label(bridged)
    if not count:
        raise ValueError("Empty arteries after closing")
    sizes = np.bincount(components.ravel()); sizes[0] = 0
    bridged = components == sizes.argmax()

    skeleton = skeletonize_3d(bridged) > 0
    if not skeleton.any():
        raise ValueError("Empty skeleton")

    dist = distance_transform_edt(bridged, sampling=spacing)
    skel_obj = Skeleton(skeleton, spacing=spacing)
    branch_data = summarize(skel_obj, separator='-')

    G = nx.Graph()
    for i, row in branch_data.iterrows():
        path_voxels = skel_obj.path_coordinates(i)
        radii = dist[tuple(path_voxels.T)]
        src, dst = int(row['node-id-src']), int(row['node-id-dst'])
        if src == dst:
            continue
        G.add_edge(src, dst,
                   length_mm=float(row['branch-distance']),
                   radius_mean_mm=float(radii.mean()),
                   radius_min_mm=float(radii.min()),
                   radius_max_mm=float(radii.max()),
                   num_points=len(path_voxels),
                   path_voxels=path_voxels,
                   node_src=src, node_dst=dst)

    # Prune spurious short leaf branches near high-connectivity trunk
    # regions -- otherwise the root/first-bifurcation search latches onto
    # skeleton noise. Iterative: removing a leaf can expose a new short leaf.
    changed = True
    while changed:
        changed = False
        for node in list(G.nodes()):
            if G.degree(node) == 1:
                (neighbor,) = G.neighbors(node)
                if G.edges[node, neighbor]['length_mm'] < min_branch_length_mm and G.number_of_edges() > 1:
                    G.remove_node(node)
                    changed = True

    if G.number_of_edges() == 0:
        raise ValueError("Branch tree is empty after pruning")
    return G


def find_root_node(G: nx.Graph):
    """Root = degree-1 (endpoint) node whose incident edge has the highest
    radius_mean_mm over the whole tree (the pulmonic-valve end of the main
    PA trunk)."""
    endpoints = [n for n in G.nodes() if G.degree(n) == 1]
    if not endpoints:
        raise ValueError("No endpoint nodes found; branch tree has no leaves")

    def edge_radius(n):
        (neighbor,) = G.neighbors(n)
        return G.edges[n, neighbor]['radius_mean_mm']

    return max(endpoints, key=edge_radius)


def side_of_subtree(di: nx.DiGraph, u, v, lung_right: np.ndarray, lung_left: np.ndarray,
                    image_mid_x: float) -> str:
    """Side of edge (u, v), decided by AGGREGATE lung-mask overlap over that
    edge PLUS its entire descendant subtree (not just its own short proximal
    segment -- the RPA runs posterior to the ascending aorta before curving
    right, so a single edge right after the split can be ambiguous)."""
    right_total = left_total = 0
    xs = []
    stack = [(u, v)]
    while stack:
        a, b = stack.pop()
        pv = di.edges[a, b]['path_voxels']
        right_total += int(lung_right[tuple(pv.T)].sum())
        left_total += int(lung_left[tuple(pv.T)].sum())
        xs.append(pv[:, 0])
        for child in di.successors(b):
            stack.append((b, child))

    if right_total == 0 and left_total == 0:
        mean_x = np.concatenate(xs).mean() if xs else image_mid_x
        return "right" if mean_x < image_mid_x else "left"
    return "right" if right_total >= left_total else "left"


def lobe_overlap_of_subtree(di: nx.DiGraph, u, v, lobe_masks: dict[str, np.ndarray],
                            lobe_names: list[str]) -> dict[str, float]:
    """Per-lobe overlap fraction aggregated over edge (u, v) plus its entire
    descendant subtree -- catches a "sole continuation" segment that sits at
    ~0% overlap for several generations (still in the interlobar fissure)
    even while everything downstream of it is clearly inside one lobe."""
    counts = {lobe: 0 for lobe in lobe_names}
    total = 0
    stack = [(u, v)]
    while stack:
        a, b = stack.pop()
        pv = di.edges[a, b]['path_voxels']
        total += len(pv)
        for lobe in lobe_names:
            counts[lobe] += int(lobe_masks[lobe][tuple(pv.T)].sum())
        for child in di.successors(b):
            stack.append((b, child))
    return {lobe: (c / total if total else 0.0) for lobe, c in counts.items()}


# --------------------------------------------------------------------------
# PERADS.net F: named-structure classification (trunk -> RPA/LPA ->
# lobar/segmental/subsegmental per lobe). Unchanged from this session's
# final, validated F.
# --------------------------------------------------------------------------

def classify_named_structures(G: nx.Graph, root, lung_right: np.ndarray, lung_left: np.ndarray,
                              lobe_masks: dict[str, np.ndarray], image_mid_x: float) -> dict:
    di = nx.bfs_tree(G, root)
    for u, v in di.edges():
        di.edges[u, v].update(G.edges[u, v])
        if di.edges[u, v]['node_src'] != u:
            di.edges[u, v]['path_voxels'] = di.edges[u, v]['path_voxels'][::-1].copy()

    root_children = list(di.successors(root))
    if len(root_children) != 1:
        raise ValueError(f"Root node {root} does not have exactly one child edge (has {len(root_children)})")
    root_child = root_children[0]

    edge_id_of = {edge: i for i, edge in enumerate(nx.bfs_edges(di, root))}
    incoming_edge_id: dict = {}
    node_generation = {root: 0, root_child: 1}
    records: dict[int, dict] = {}

    trunk_edge_id = edge_id_of[(root, root_child)]
    incoming_edge_id[root_child] = trunk_edge_id
    trunk_path_voxels = di.edges[root, root_child]['path_voxels']
    records[trunk_edge_id] = {
        "edge_id": trunk_edge_id, "nodo_padre": -1, "generazione": 0,
        "structure_name": "main_trunk",
        "lunghezza_mm": di.edges[root, root_child]['length_mm'],
        "raggio_medio_mm": di.edges[root, root_child]['radius_mean_mm'],
        "raggio_min_mm": di.edges[root, root_child]['radius_min_mm'],
        "raggio_max_mm": di.edges[root, root_child]['radius_max_mm'],
        "numero_punti": di.edges[root, root_child]['num_points'],
        "path_voxels": trunk_path_voxels,
    }
    # Side (R/L) locks once past the first real bifurcation and never
    # changes again. Lobe assignment never locks: every edge is
    # reclassified fresh from its own local evidence (own overlap vs
    # siblings' calibers), so a branch mistaken for one lobe near a
    # boundary can correct once its own descendants show clearer overlap.
    # Depth within a lobe resets to 0 on fresh entry: 0="<lobe>_artery",
    # 1="<lobe>_segmental", >=2="<lobe>_subsegmental".
    node_state = {root_child: {"side": None, "lobe": None, "depth_in_lobe": -1}}
    CONTINUATION_RATIO = 0.75
    HIGH_CONFIDENCE = 0.50

    for u, v in nx.bfs_edges(di, root_child):
        path_voxels = di.edges[u, v]['path_voxels']
        parent = node_state[u]
        side = parent["side"]

        if side is None:
            side = side_of_subtree(di, u, v, lung_right, lung_left, image_mid_x)
            name = "RPA" if side == "right" else "LPA"
            lobe, depth_in_lobe = None, -1
        else:
            lobe_names = LOBES_RIGHT if side == "right" else LOBES_LEFT
            total = len(path_voxels)
            overlaps = {ln: float(lobe_masks[ln][tuple(path_voxels.T)].sum()) / total if total else 0.0
                       for ln in lobe_names}
            best_lobe_mask = max(overlaps, key=overlaps.get)
            best_frac = overlaps[best_lobe_mask]

            is_sole_continuation = False
            if di.out_degree(u) >= 2:
                siblings = list(di.successors(u))
                radii = {c: di.edges[u, c]['radius_mean_mm'] for c in siblings}
                max_radius = max(radii.values())
                others_max = max((r for c, r in radii.items() if c != v), default=0.0)
                is_sole_continuation = (radii[v] == max_radius) and (others_max < CONTINUATION_RATIO * max_radius)

            if (best_frac >= HIGH_CONFIDENCE or not is_sole_continuation) and best_frac > 0:
                lobe = LOBE_TO_SHORT[best_lobe_mask]
                depth_in_lobe = 0 if lobe != parent["lobe"] else parent["depth_in_lobe"] + 1
                suffix = "artery" if depth_in_lobe == 0 else ("segmental" if depth_in_lobe == 1 else "subsegmental")
                name = f"{lobe}_{suffix}"
            else:
                subtree_overlaps = lobe_overlap_of_subtree(di, u, v, lobe_masks, lobe_names)
                subtree_best = max(subtree_overlaps, key=subtree_overlaps.get)
                if subtree_overlaps[subtree_best] >= HIGH_CONFIDENCE:
                    lobe = LOBE_TO_SHORT[subtree_best]
                    depth_in_lobe = 0 if lobe != parent["lobe"] else parent["depth_in_lobe"] + 1
                    suffix = "artery" if depth_in_lobe == 0 else ("segmental" if depth_in_lobe == 1 else "subsegmental")
                    name = f"{lobe}_{suffix}"
                else:
                    name = "RPA" if side == "right" else "LPA"
                    lobe, depth_in_lobe = None, -1

        node_state[v] = {"side": side, "lobe": lobe, "depth_in_lobe": depth_in_lobe}

        this_edge_id = edge_id_of[(u, v)]
        incoming_edge_id[v] = this_edge_id
        node_generation[v] = node_generation[u] + 1

        records[this_edge_id] = {
            "edge_id": this_edge_id,
            "nodo_padre": incoming_edge_id[u],
            "generazione": node_generation[u],
            "structure_name": name,
            "lunghezza_mm": di.edges[u, v]['length_mm'],
            "raggio_medio_mm": di.edges[u, v]['radius_mean_mm'],
            "raggio_min_mm": di.edges[u, v]['radius_min_mm'],
            "raggio_max_mm": di.edges[u, v]['radius_max_mm'],
            "numero_punti": di.edges[u, v]['num_points'],
            "path_voxels": path_voxels,
        }

    return records


def create_labelmap(arteries: np.ndarray, records: dict) -> np.ndarray:
    """Nearest-edge-voxel assignment via a single pooled cKDTree."""
    labelmap = np.zeros(arteries.shape, dtype=np.uint8)
    arterial_voxels = np.argwhere(arteries > 0)
    if len(arterial_voxels) == 0:
        return labelmap

    all_points, all_codes = [], []
    for r in records.values():
        all_points.append(r["path_voxels"])
        code = STRUCTURE_CODES[r["structure_name"]]
        all_codes.append(np.full(len(r["path_voxels"]), code, dtype=np.uint8))
    pooled_points = np.concatenate(all_points, axis=0)
    pooled_codes = np.concatenate(all_codes, axis=0)

    tree = cKDTree(pooled_points)
    _, nearest_idx = tree.query(arterial_voxels)
    labelmap[tuple(arterial_voxels.T)] = pooled_codes[nearest_idx]
    return labelmap


def compute_perads_grade(classified_embolus: np.ndarray, perads_labelmap: np.ndarray,
                         voxel_volume_mm3: float) -> dict:
    """Grade = most central PE-RADS class (4=main/right/left ... 1=subsegmental)
    holding at least OVERLAP fraction of the embolus, once total volume
    clears MIN_EMBOLUS_VOLUME_MM3."""
    total = int(classified_embolus.sum())
    bcv_mm3 = total * voxel_volume_mm3

    fractions = {c: (int((classified_embolus & (perads_labelmap == c)).sum()) / total if total else 0.0)
                 for c in (4, 3, 2, 1)}

    if bcv_mm3 < MIN_EMBOLUS_VOLUME_MM3:
        grade, reason = 0, "below_minimum_embolus_volume"
    else:
        grade = next((c for c in (4, 3, 2, 1) if fractions[c] >= OVERLAP), 0)
        reason = "overlap_hierarchy"

    return {
        'perads_grade': grade,
        'anatomic_level': PE_CLASS_NAMES.get(grade, "none"),
        'classification_reason': reason,
        'embolic_voxels_hardmask': total,
        'bcv_mm3': round(bcv_mm3, 3),
        'voxel_volume_mm3': round(voxel_volume_mm3, 6),
        'minimum_embolus_volume_mm3': MIN_EMBOLUS_VOLUME_MM3,
        'minimum_embolus_overlap_fraction': OVERLAP,
        'class_fractions': {str(c): round(f, 5) for c, f in fractions.items()},
    }


def compute_grade_with_named_site(labelmap: np.ndarray, classified_embolus: np.ndarray,
                                  voxel_volume_mm3: float) -> dict:
    """PE-RADS grade via the hard-mask/threshold/overlap rule above, applied
    to F's 19 named structures collapsed down to the 4 standard PE-RADS
    proximity classes. Also reports WHICH specific named structure(s) at the
    winning class actually carry the embolus."""
    code_to_class = {STRUCTURE_CODES[name]: cls for name, cls in PERADS_CLASS.items()}
    perads_labelmap = np.zeros(labelmap.shape, dtype=np.uint8)
    for code, cls in code_to_class.items():
        perads_labelmap[labelmap == code] = cls

    grade_info = compute_perads_grade(classified_embolus, perads_labelmap, voxel_volume_mm3)
    grade = grade_info["perads_grade"]

    named_sites = []
    if grade > 0:
        for name, cls in PERADS_CLASS.items():
            if cls != grade:
                continue
            code = STRUCTURE_CODES[name]
            count = int((classified_embolus & (labelmap == code)).sum())
            if count > 0:
                named_sites.append((name, count))
        named_sites.sort(key=lambda x: -x[1])

    grade_info["most_proximal_structures"] = [name for name, _ in named_sites]
    return grade_info


# --------------------------------------------------------------------------
# End-to-end per-case pipeline
# --------------------------------------------------------------------------

def process_case(ct_path: Path, output_dir: Path, case_id: str, *, model: Path = DEFAULT_MODEL,
                 device: str = "cuda", folds: list[int] | None = None,
                 totalsegmentator: str = "TotalSegmentator", nnunet: str = "nnUNetv2_predict_from_modelfolder",
                 overwrite: bool = False) -> dict:
    folds = folds if folds is not None else [0, 1, 2, 3, 4]
    t0 = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    ts_device = "gpu" if device == "cuda" else "cpu"
    ts_dir = output_dir / "totalsegmentator"

    # Step 1 -- TotalSegmentator x3, all on the SAME raw CT (natively
    # co-registered outputs, no resampling needed between them).
    ts_lung, ts_heart, ts_lobes = ts_dir / "lung_vessels", ts_dir / "heartchambers", ts_dir / "lobes"
    lung_masks = [ts_lung / f"{n}.nii.gz" for n in ("lung_airways", "lung_arteries", "lung_veins")]
    if not all(p.exists() for p in lung_masks):
        cmd = [totalsegmentator, "-i", str(ct_path), "-o", str(ts_lung), "-ta", "lung_vessels"]
        if os.environ.get("PERADS_TOTALSEG_NO_DEVICE") != "1": cmd += ["-d", ts_device]
        run(cmd)

    heart_masks = [ts_heart / f"heart_{n}.nii.gz" for n in
                   ("atrium_right", "atrium_left", "ventricle_right", "ventricle_left")]
    if not all(p.exists() for p in heart_masks):
        cmd = [totalsegmentator, "-i", str(ct_path), "-o", str(ts_heart), "-ta", "heartchambers_highres"]
        if os.environ.get("PERADS_TOTALSEG_NO_DEVICE") != "1": cmd += ["-d", ts_device]
        run(cmd)

    lobe_names = LOBES_RIGHT + LOBES_LEFT
    lobe_files = [ts_lobes / f"{n}.nii.gz" for n in lobe_names]
    if not all(p.exists() for p in lobe_files):
        cmd = [totalsegmentator, "-i", str(ct_path), "-o", str(ts_lobes), "-ta", "total", "-rs", *lobe_names]
        if os.environ.get("PERADS_TOTALSEG_NO_DEVICE") != "1": cmd += ["-d", ts_device]
        run(cmd)

    # Step 2 -- crop once, for nnU-Net input only. Everything else (arteries,
    # lobes, heart chambers) stays at the raw CT's native full resolution.
    raw = nib.load(str(ct_path)); raw_shape = raw.shape
    slices = crop_slices(lung_masks, raw_shape)
    pre = output_dir / "preprocessed"; images = pre / "imagesTs"; images.mkdir(parents=True, exist_ok=True)
    cropped_ct = images / f"{case_id}_0000.nii.gz"
    cropped_prior = images / f"{case_id}_0001.nii.gz"
    save_crop(raw, slices, cropped_ct)
    arteries_img = nib.load(str(ts_lung / "lung_arteries.nii.gz"))
    prior = binary_dilation(np.asanyarray(arteries_img.dataobj) > 0, iterations=5).astype(np.uint8)
    save_crop(nib.Nifti1Image(prior, arteries_img.affine, arteries_img.header.copy()), slices, cropped_prior)

    # Step 3 -- Dataset547 5-fold inference on the small crop.
    inference = output_dir / "embolus"; inference.mkdir(parents=True, exist_ok=True)
    prediction = inference / f"{case_id}.nii.gz"
    if not prediction.exists() or overwrite:
        run([nnunet, "-i", str(images), "-o", str(inference), "-m", str(model),
             "-f", *[str(f) for f in folds], "-device", device, "-npp", "1", "-nps", "1"])

    # Step 4 -- load full-res arteries + lobe masks (all natively co-registered
    # with the raw CT already, no resampling), align the small embolus
    # prediction back into that same full-res grid.
    arteries_raw, affine, spacing = load_binary_mask(ts_lung / "lung_arteries.nii.gz")
    arteries = largest_connected_component(arteries_raw)
    image_mid_x = arteries.shape[0] / 2.0

    lobe_masks = {}
    for name in lobe_names:
        mask, _, _ = load_binary_mask(ts_lobes / f"{name}.nii.gz")
        lobe_masks[name] = binary_dilation(mask, iterations=LOBE_DILATION_ITERATIONS)
    lung_right = np.zeros(arteries.shape, dtype=bool)
    for name in LOBES_RIGHT: lung_right |= lobe_masks[name]
    lung_left = np.zeros(arteries.shape, dtype=bool)
    for name in LOBES_LEFT: lung_left |= lobe_masks[name]

    # Step 5 -- PERADS.net F classification + grade.
    G = build_branch_tree(arteries, spacing)
    root = find_root_node(G)
    records = classify_named_structures(G, root, lung_right, lung_left, lobe_masks, image_mid_x)
    labelmap = create_labelmap(arteries, records)

    embolus = align_embolus_to_arteries(prediction, arteries.shape, affine)
    classified_embolus = embolus & (arteries_raw > 0)
    voxel_volume_mm3 = float(np.prod(spacing))
    grade_info = compute_grade_with_named_site(labelmap, classified_embolus, voxel_volume_mm3)

    structure_counts: dict[str, int] = {}
    for r in records.values():
        structure_counts[r["structure_name"]] = structure_counts.get(r["structure_name"], 0) + 1

    # Step 6 -- RV/LV v3, on the full-res CT + full-res heart chamber masks
    # (its native expected input -- unchanged from run_perads_case.py).
    rvlv_dir = output_dir / "rv_lv"
    ra, la, rv, lv = heart_masks
    rvlv = calculate_rvlv(ct_path, ra, la, rv, lv, case_id, rvlv_dir)

    # Step 7 -- write outputs.
    final = output_dir / "results"; final.mkdir(parents=True, exist_ok=True)
    labelmap_path = final / "structure_labelmap.nii.gz"
    nib.save(nib.Nifti1Image(labelmap, affine), str(labelmap_path))

    duration_sec = round(time.time() - t0, 1)
    result = {
        "case_id": case_id,
        "num_branches": G.number_of_edges(),
        "structure_branch_counts": structure_counts,
        **grade_info,
        "rv_lv_ratio": rvlv["ratio_four_chamber"],
        "rv_diameter_mm": rvlv["rv_chord"]["length_mm"],
        "lv_diameter_mm": rvlv["lv_chord"]["length_mm"],
        "duration_sec": duration_sec,
    }
    (final / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    csv_row = {
        "case_id": case_id,
        "perads_grade": result["perads_grade"],
        "anatomic_level": result["anatomic_level"],
        "most_proximal_structures": "+".join(result["most_proximal_structures"]),
        "embolic_voxels_hardmask": result["embolic_voxels_hardmask"],
        "bcv_mm3": result["bcv_mm3"],
        "classification_reason": result["classification_reason"],
        "rv_lv_ratio": round(result["rv_lv_ratio"], 4) if result["rv_lv_ratio"] is not None else None,
        "rv_diameter_mm": result["rv_diameter_mm"],
        "lv_diameter_mm": result["lv_diameter_mm"],
        "num_branches": result["num_branches"],
        "duration_sec": duration_sec,
    }
    print(f"✓ {case_id}: grade={result['perads_grade']} ({result['anatomic_level']}) "
          f"site={csv_row['most_proximal_structures']} rv_lv={csv_row['rv_lv_ratio']} "
          f"[{duration_sec}s]", flush=True)
    return csv_row


CSV_FIELDNAMES = ["case_id", "perads_grade", "anatomic_level", "most_proximal_structures",
                  "embolic_voxels_hardmask", "bcv_mm3", "classification_reason",
                  "rv_lv_ratio", "rv_diameter_mm", "lv_diameter_mm", "num_branches", "duration_sec"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Raw, uncropped chest CT NIfTI (.nii or .nii.gz)")
    parser.add_argument("--output", required=True, type=Path, help="Output directory")
    parser.add_argument("--case-id", help="Optional case identifier; defaults to input filename")
    parser.add_argument("--csv", type=Path, help="CSV file to append the result row to (default: <output>/result.csv)")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--totalsegmentator", default=os.environ.get("PERADS_TOTALSEGMENTATOR", "TotalSegmentator"))
    parser.add_argument("--nnunet", default=os.environ.get("PERADS_NNUNET", "nnUNetv2_predict_from_modelfolder"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    ct_path = args.input.resolve(); output = args.output.resolve()
    case_id = args.case_id or ct_path.name.removesuffix(".nii.gz").removesuffix(".nii")
    if not ct_path.is_file(): parser.error(f"Input not found: {ct_path}")
    if not args.model.is_dir(): parser.error(f"nnU-Net model not found: {args.model}")

    row = process_case(ct_path, output, case_id, model=args.model, device=args.device, folds=args.folds,
                       totalsegmentator=args.totalsegmentator, nnunet=args.nnunet, overwrite=args.overwrite)

    csv_path = args.csv or (output / "result.csv")
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if write_header: writer.writeheader()
        writer.writerow(row)
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
