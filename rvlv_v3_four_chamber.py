#!/usr/bin/env python3
"""PERADS.net RV/LV v3: AV-base triangle four-chamber diameters.

The triangle apex is the LV apex; its unconstrained straight base is fitted to
the RV-RA and LV-LA atrioventricular junctions. The measurement line is
parallel to the base at 25% of the base-to-apex height, on the proximal/basal
side. RV and LV are measured as separate contiguous mask intersections,
excluding the interventricular gap.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, cKDTree


def unit(vector: np.ndarray, name: str = "vector") -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-6:
        raise ValueError(f"Degenerate {name}")
    return np.asarray(vector, dtype=float) / norm


def largest_component(mask: np.ndarray) -> np.ndarray:
    labels, count = ndimage.label(mask)
    if not count:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(labels.ravel()); sizes[0] = 0
    return labels == sizes.argmax()


def voxel_to_world(indices: np.ndarray, affine: np.ndarray) -> np.ndarray:
    return nib.affines.apply_affine(affine, np.asarray(indices, dtype=float))


def centroid_world(mask: np.ndarray, affine: np.ndarray, name: str) -> np.ndarray:
    component = largest_component(mask)
    points = np.argwhere(component)
    if len(points) < 20:
        raise ValueError(f"Insufficient {name} mask voxels")
    return voxel_to_world(points.mean(axis=0), affine)


def boundary_world(mask: np.ndarray, affine: np.ndarray, max_points: int = 30000) -> np.ndarray:
    component = largest_component(mask)
    boundary = component & ~ndimage.binary_erosion(component, iterations=1)
    points = np.argwhere(boundary)
    if len(points) > max_points:
        points = points[:: int(np.ceil(len(points) / max_points))]
    return voxel_to_world(points, affine)


def four_chamber_plane(centroids: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    points = np.stack([centroids[key] for key in ("RA", "LA", "RV", "LV")])
    origin = points.mean(axis=0)
    centered = points - origin
    _, singular, vectors = np.linalg.svd(centered, full_matrices=False)
    measure_hint = unit(centroids["LV"] - centroids["RV"], "RV-LV axis")
    long_hint = ((centroids["RA"] + centroids["LA"]) - (centroids["RV"] + centroids["LV"])) / 2
    degeneracy = float(singular[1] / singular[0]) if singular[0] else 0.0
    if degeneracy < 0.05:
        measure = measure_hint
        long_axis = long_hint - np.dot(long_hint, measure) * measure
        normal = unit(np.cross(measure, unit(long_axis, "atrioventricular axis")), "fallback plane normal")
    else:
        normal = unit(vectors[-1], "four-chamber plane normal")
    measure = measure_hint - np.dot(measure_hint, normal) * normal
    measure = unit(measure, "projected RV-LV axis")
    long_axis = unit(np.cross(normal, measure), "four-chamber long axis")
    if np.dot(long_axis, long_hint) < 0:
        long_axis = -long_axis
        normal = -normal
    residuals = np.abs(centered @ normal)
    quality = {
        "singular_values": singular.tolist(),
        "degeneracy_ratio_s1_s0": degeneracy,
        "centroid_plane_residual_mm_max": float(residuals.max()),
        "centroid_plane_residual_mm_mean": float(residuals.mean()),
        "fallback_plane": bool(degeneracy < 0.05),
    }
    return origin, normal, measure, long_axis, quality


def septal_measure_axis(rv: np.ndarray, lv: np.ndarray, affine: np.ndarray, plane_normal: np.ndarray,
                        fallback: np.ndarray) -> tuple[np.ndarray, str, dict]:
    rv_points, lv_points = boundary_world(rv, affine), boundary_world(lv, affine)
    if len(rv_points) < 20 or len(lv_points) < 20:
        return fallback, "centroid_fallback", {"interface_points": 0}
    distances, nearest = cKDTree(lv_points).query(rv_points, k=1)
    threshold = min(12.0, float(distances.min()) + 5.0)
    selected = distances <= threshold
    interface = (rv_points[selected] + lv_points[nearest[selected]]) / 2
    if len(interface) < 20:
        return fallback, "centroid_fallback", {"interface_points": int(len(interface)), "interface_threshold_mm": threshold}
    centered = interface - interface.mean(axis=0)
    _, singular, vectors = np.linalg.svd(centered, full_matrices=False)
    septal_normal = vectors[-1]
    projected = septal_normal - np.dot(septal_normal, plane_normal) * plane_normal
    if np.linalg.norm(projected) < 1e-5:
        return fallback, "centroid_fallback", {"interface_points": int(len(interface)), "interface_threshold_mm": threshold}
    projected = unit(projected)
    if np.dot(projected, fallback) < 0: projected = -projected
    return projected, "3d_interface_plane_normal", {
        "interface_points": int(len(interface)),
        "interface_threshold_mm": threshold,
        "interface_singular_values": singular.tolist(),
    }


def plane_bounds(masks: list[np.ndarray], affine: np.ndarray, origin: np.ndarray,
                 axis_u: np.ndarray, axis_v: np.ndarray, margin_mm: float = 12.0) -> tuple[float, float, float, float]:
    projected = []
    for mask in masks:
        points = np.argwhere(largest_component(mask))
        if len(points) > 50000: points = points[:: int(np.ceil(len(points) / 50000))]
        world = voxel_to_world(points, affine) - origin
        projected.append(np.column_stack((world @ axis_u, world @ axis_v)))
    all_points = np.concatenate(projected)
    return (float(all_points[:, 0].min() - margin_mm), float(all_points[:, 0].max() + margin_mm),
            float(all_points[:, 1].min() - margin_mm), float(all_points[:, 1].max() + margin_mm))


def sample_plane(data: np.ndarray, inv_affine: np.ndarray, origin: np.ndarray, axis_u: np.ndarray,
                 axis_v: np.ndarray, bounds: tuple[float, float, float, float], spacing_mm: float,
                 order: int, offset_normal: np.ndarray | None = None, offset_mm: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u0, u1, v0, v1 = bounds
    u = np.arange(u0, u1 + spacing_mm * .5, spacing_mm)
    v = np.arange(v0, v1 + spacing_mm * .5, spacing_mm)
    uu, vv = np.meshgrid(u, v)
    plane_origin = origin if offset_normal is None else origin + offset_mm * offset_normal
    world = plane_origin[None, None, :] + uu[..., None] * axis_u + vv[..., None] * axis_v
    vox = nib.affines.apply_affine(inv_affine, world.reshape(-1, 3)).reshape(world.shape)
    sampled = ndimage.map_coordinates(data, [vox[..., 0], vox[..., 1], vox[..., 2]], order=order, mode="constant", cval=0)
    return sampled, u, v


def select_parallel_plane(masks: dict[str, np.ndarray], inv_affine: np.ndarray, origin: np.ndarray,
                          normal: np.ndarray, axis_u: np.ndarray, axis_v: np.ndarray,
                          bounds: tuple[float, float, float, float], spacing_mm: float) -> tuple[float, dict]:
    best = None
    for offset in np.arange(-8.0, 8.01, 1.0):
        areas = {}
        for key in ("RV", "LV", "RA", "LA"):
            sampled, _, _ = sample_plane(masks[key].astype(np.uint8), inv_affine, origin, axis_u, axis_v,
                                         bounds, spacing_mm, 0, normal, float(offset))
            areas[key] = int((sampled > 0).sum())
        score = np.sqrt(areas["RV"] * areas["LV"]) + .15 * np.sqrt(areas["RA"] * areas["LA"])
        candidate = (float(score), -abs(float(offset)), float(offset), areas)
        if best is None or candidate[:2] > best[:2]: best = candidate
    if best is None or best[0] <= 0: raise ValueError("Unable to find a plane intersecting both ventricles")
    return best[2], {"selection_score": best[0], "areas_px": best[3]}


def line_runs(mask: np.ndarray, point_xy: np.ndarray, direction_xy: np.ndarray,
              spacing_mm: float, sample_step_px: float = .25) -> list[dict]:
    direction_xy = unit(direction_xy, "2D line direction")
    extent = float(np.hypot(*mask.shape) * 1.2)
    samples = np.arange(-extent, extent + sample_step_px * .5, sample_step_px)
    xy = point_xy[None, :] + samples[:, None] * direction_xy[None, :]
    values = ndimage.map_coordinates(mask.astype(np.uint8), [xy[:, 1], xy[:, 0]], order=0, mode="constant") > 0
    changes = np.diff(np.r_[False, values, False].astype(np.int8))
    runs=[]
    for start,end in zip(np.flatnonzero(changes == 1),np.flatnonzero(changes == -1)):
        s0,s1=float(samples[start]),float(samples[end-1])
        p0=point_xy+s0*direction_xy; p1=point_xy+s1*direction_xy
        runs.append({"length_mm":float((end-start)*sample_step_px*spacing_mm),
                     "endpoints_px":[p0.tolist(),p1.tolist()],"s_range_px":[s0,s1],
                     "contains_origin":bool(s0 <= 0 <= s1)})
    return runs


def longest_internal_chord_2d(mask: np.ndarray, spacing_mm: float) -> dict:
    component=largest_component(mask)
    boundary=component & ~ndimage.binary_erosion(component)
    yx=np.argwhere(boundary)
    if len(yx)<3: raise ValueError("Insufficient LV boundary on four-chamber plane")
    xy=yx[:,[1,0]].astype(float)
    hull=xy[ConvexHull(xy).vertices]
    distances=np.linalg.norm(hull[:,None]-hull[None,:],axis=2)
    a,b=np.unravel_index(np.argmax(distances),distances.shape)
    direction=unit(hull[b]-hull[a],"LV longitudinal direction")
    normal=np.array([-direction[1],direction[0]])
    center=xy.mean(axis=0); offsets=(xy-center)@normal
    best=None
    for offset in np.arange(offsets.min(),offsets.max()+.5,.5):
        for run in line_runs(component,center+offset*normal,direction,spacing_mm):
            if best is None or run["length_mm"]>best["length_mm"]:
                best={**run,"direction_xy":direction.tolist(),"offset_px":float(offset)}
    if best is None: raise ValueError("Unable to identify LV longitudinal chord")
    return best


def av_junction_points(ventricle: np.ndarray, atrium: np.ndarray) -> np.ndarray:
    """Return basal ventricular boundary points facing the matching atrium."""
    boundary=ventricle & ~ndimage.binary_erosion(ventricle)
    yx=np.argwhere(boundary)
    if len(yx)<5: raise ValueError("Insufficient ventricular boundary for AV junction")
    xy=yx[:,[1,0]].astype(float)
    vent_y,vent_x=ndimage.center_of_mass(ventricle)
    atrium_y,atrium_x=ndimage.center_of_mass(atrium)
    vent_center=np.array([vent_x,vent_y]); atrium_center=np.array([atrium_x,atrium_y])
    toward_atrium=unit(atrium_center-vent_center,"ventricle-to-atrium direction")
    basal_projection=(xy-vent_center)@toward_atrium
    basal=xy[basal_projection>=np.percentile(basal_projection,70)]
    if len(basal)<5: basal=xy
    distance_to_atrial_center=np.linalg.norm(basal-atrium_center,axis=1)
    count=min(30,max(8,int(np.ceil(.20*len(basal)))))
    return basal[np.argsort(distance_to_atrial_center)[:count]]


def biventricular_triangle_chords(rv: np.ndarray, lv: np.ndarray,
                                  ra: np.ndarray, la: np.ndarray,
                                  u: np.ndarray, v: np.ndarray,
                                  spacing_mm: float) -> tuple[dict, dict, dict, dict]:
    rv,lv=largest_component(rv),largest_component(lv)
    ra,la=largest_component(ra),largest_component(la)
    longitudinal=longest_internal_chord_2d(lv,spacing_mm)
    ends=np.asarray(longitudinal["endpoints_px"],dtype=float)
    la_y,la_x=ndimage.center_of_mass(la)
    la_center=np.array([la_x,la_y])
    base_index=int(np.argmin(np.linalg.norm(ends-la_center,axis=1)))
    lv_base=ends[base_index]; apex=ends[1-base_index]

    # Fit one straight base to the two actual atrioventricular junctions.
    # The nearest ventricular boundary points to RA and LA approximate the
    # tricuspid and mitral valve planes; total least squares gives a common,
    # unconstrained (not necessarily symmetric) basal tangent.
    rv_av=av_junction_points(rv,ra)
    lv_av=av_junction_points(lv,la)
    av_points=np.vstack([rv_av,lv_av])
    base_center=av_points.mean(axis=0)
    _,_,vh=np.linalg.svd(av_points-base_center,full_matrices=False)
    base_direction=unit(vh[0],"atrioventricular base direction")
    projections=(av_points-base_center)@base_direction
    low=float(np.percentile(projections,2)); high=float(np.percentile(projections,98))
    padding=.05*max(high-low,1.0)
    base_left=base_center+(low-padding)*base_direction
    base_right=base_center+(high+padding)*base_direction
    base_midpoint=(base_left+base_right)/2
    transverse=base_direction
    height=float(abs(np.cross(base_right-base_left,apex-base_left))/
                 max(np.linalg.norm(base_right-base_left),1e-6))
    if height<2: raise ValueError("Degenerate apex-to-AV-base triangle height")

    # Place the common diameter line in the proximal quarter: start at the AV
    # base and advance 25% of the base-to-apex height. It remains parallel to
    # the fitted base, and mask runs remain separate by construction.
    measurement_fraction_from_base=.25
    point=base_midpoint+measurement_fraction_from_base*(apex-base_midpoint)
    lv_runs=line_runs(lv,point,transverse,spacing_mm)
    if not lv_runs: raise ValueError("Triangle proximal 25% line does not intersect LV")
    lv_chord=max(lv_runs,key=lambda run:run["length_mm"])
    rv_runs=line_runs(rv,point,transverse,spacing_mm)
    if not rv_runs: raise ValueError("Triangle proximal 25% line does not intersect RV")
    rv_chord=max(rv_runs,key=lambda run:run["length_mm"])
    def finish(chord):
        plane=[]
        for x,y in chord["endpoints_px"]:
            plane.append([float(u[0]+x*spacing_mm),float(v[0]+y*spacing_mm)])
        return {**chord,"endpoints_plane_mm":plane,"line_point_px":point.tolist(),
                "direction_xy":transverse.tolist()}
    longitudinal={**longitudinal,"apex_px":apex.tolist(),"base_px":lv_base.tolist()}
    triangle={
        "apex_px":apex.tolist(),
        "base_left_px":base_left.tolist(),
        "base_right_px":base_right.tolist(),
        "base_midpoint_px":base_midpoint.tolist(),
        "midline_point_px":point.tolist(),  # Backward-compatible alias.
        "measurement_line_point_px":point.tolist(),
        "height_mm":float(height*spacing_mm),
        "base_width_mm":float(np.linalg.norm(base_right-base_left)*spacing_mm),
        "measurement_height_fraction":measurement_fraction_from_base,
        "measurement_fraction_reference":"atrioventricular_base_toward_lv_apex",
        "measurement_height_fraction_from_base":measurement_fraction_from_base,
        "measurement_height_fraction_from_apex":1.0-measurement_fraction_from_base,
        "base_fit_source":"nearest_RV_RA_and_LV_LA_boundary_points",
        "rv_av_points":int(len(rv_av)),
        "lv_av_points":int(len(lv_av)),
    }
    return finish(rv_chord),finish(lv_chord),longitudinal,triangle


def calculate(ct_path: Path, ra_path: Path, la_path: Path, rv_path: Path, lv_path: Path,
              case_id: str, output_dir: Path, spacing_mm: float = 0.75) -> dict:
    paths = {"CT": ct_path, "RA": ra_path, "LA": la_path, "RV": rv_path, "LV": lv_path}
    images = {key: nib.load(str(path)) for key, path in paths.items()}
    reference = images["CT"]
    for key, image in images.items():
        if image.shape != reference.shape or not np.allclose(image.affine, reference.affine, atol=1e-4):
            raise ValueError(f"Geometry mismatch for {key}")
    ct = np.asanyarray(reference.dataobj)
    masks = {key: np.asanyarray(images[key].dataobj) > 0 for key in ("RA", "LA", "RV", "LV")}
    centroids = {key: centroid_world(masks[key], reference.affine, key) for key in masks}
    origin, normal, centroid_measure, long_axis, plane_quality = four_chamber_plane(centroids)
    measure, measure_source, septal_quality = septal_measure_axis(masks["RV"], masks["LV"], reference.affine,
                                                                  normal, centroid_measure)
    # Rebuild the orthogonal in-plane long axis after refining the trans-septal direction.
    long_axis = unit(np.cross(normal, measure), "refined four-chamber long axis")
    long_hint = ((centroids["RA"] + centroids["LA"]) - (centroids["RV"] + centroids["LV"])) / 2
    if np.dot(long_axis, long_hint) < 0: long_axis = -long_axis
    bounds = plane_bounds(list(masks.values()), reference.affine, origin, measure, long_axis)
    inv_affine = np.linalg.inv(reference.affine)
    offset, offset_quality = select_parallel_plane(masks, inv_affine, origin, normal, measure, long_axis, bounds, spacing_mm)
    selected_origin = origin + offset * normal
    ct_4ch, u, v = sample_plane(ct, inv_affine, origin, measure, long_axis, bounds, spacing_mm, 1, normal, offset)
    sampled = {}
    for key in masks:
        arr, _, _ = sample_plane(masks[key].astype(np.uint8), inv_affine, origin, measure, long_axis,
                                 bounds, spacing_mm, 0, normal, offset)
        sampled[key] = arr > 0
    rv_chord, lv_chord, lv_longitudinal, triangle = biventricular_triangle_chords(
        sampled["RV"], sampled["LV"], sampled["RA"], sampled["LA"], u, v, spacing_mm)
    for chord in (rv_chord, lv_chord):
        chord["endpoints_world_mm"] = [
            (selected_origin + point[0] * measure + point[1] * long_axis).tolist()
            for point in chord["endpoints_plane_mm"]
        ]
    result = {
        "case_id": case_id,
        "method": "four_chamber_av_tangent_triangle_proximal25_shared_line",
        "plane_origin_world_mm": selected_origin.tolist(),
        "plane_normal_world": normal.tolist(),
        "measurement_axis_world": measure.tolist(),
        "long_axis_world": long_axis.tolist(),
        "parallel_plane_offset_mm": offset,
        "reformat_spacing_mm": spacing_mm,
        "centroids_world_mm": {key: value.tolist() for key, value in centroids.items()},
        "measurement_axis_source": measure_source,
        "lv_longitudinal_chord": lv_longitudinal,
        "biventricular_triangle": triangle,
        "rv_chord": rv_chord,
        "lv_chord": lv_chord,
        "ratio_four_chamber": round(rv_chord["length_mm"] / lv_chord["length_mm"], 4),
        "quality": {"plane": plane_quality, "septum": septal_quality, "parallel_plane": offset_quality},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rv_lv_v3_four_chamber.json").write_text(json.dumps(result, indent=2) + "\n")
    create_preview(ct_4ch, sampled["RV"], sampled["LV"], result, output_dir / "rv_lv_v3_four_chamber_preview.png")
    return result


def create_preview(ct_4ch: np.ndarray, rv_4ch: np.ndarray, lv_4ch: np.ndarray,
                   metrics: dict, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 7), facecolor="black")
    ax.imshow(ct_4ch, cmap="gray", vmin=-160, vmax=240, origin="lower")
    ax.contour(rv_4ch, [0.5], colors=["#3b82f6"], linewidths=2)
    ax.contour(lv_4ch, [0.5], colors=["#ef4444"], linewidths=2)
    for chord, color in ((metrics["rv_chord"], "#3b82f6"), (metrics["lv_chord"], "#ef4444")):
        points = np.asarray(chord["endpoints_px"])
        ax.plot(points[:, 0], points[:, 1], color=color, linewidth=5)
    longitudinal=np.asarray(metrics["lv_longitudinal_chord"]["endpoints_px"])
    ax.plot(longitudinal[:,0],longitudinal[:,1],color="#facc15",linewidth=2,linestyle="--")
    triangle=metrics["biventricular_triangle"]
    triangle_points=np.asarray([triangle["apex_px"],triangle["base_left_px"],
                                triangle["base_right_px"],triangle["apex_px"]])
    ax.plot(triangle_points[:,0],triangle_points[:,1],color="#a78bfa",linewidth=3,
            linestyle=":",alpha=.9)
    ax.set_title(
        f"RV {metrics['rv_chord']['length_mm']:.1f} mm | "
        f"LV {metrics['lv_chord']['length_mm']:.1f} mm | RV/LV {metrics['ratio_four_chamber']:.2f}",
        color="white", fontsize=11,
    )
    ax.axis("off"); fig.tight_layout(pad=.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, facecolor="black"); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ct", required=True, type=Path)
    parser.add_argument("--ra", required=True, type=Path)
    parser.add_argument("--la", required=True, type=Path)
    parser.add_argument("--rv", required=True, type=Path)
    parser.add_argument("--lv", required=True, type=Path)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--spacing-mm", type=float, default=.75)
    args = parser.parse_args()
    result = calculate(args.ct, args.ra, args.la, args.rv, args.lv, args.case_id, args.output, args.spacing_mm)
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
