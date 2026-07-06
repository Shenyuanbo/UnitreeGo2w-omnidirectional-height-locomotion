#!/usr/bin/env python3
"""Estimate Unitree Go2W commanded base-height range with per-leg IK.

The script models the height command as a vertical motion of the floating base
while the wheel centers keep their default support projection in the base x-y
plane.  It solves analytic IK for the hip/thigh/calf chain of all four legs and
reports hard, soft, and margin-based dynamic ranges.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


LEG_NAMES = ("FL", "FR", "RL", "RR")
DEFAULT_JOINTS = {"hip": 0.0, "thigh": 0.8, "calf": -1.5}
DEFAULT_URDF = Path("source/robot_lab/data/Robots/unitree/go2w_description/urdf/go2w_description.urdf")
DEFAULT_HEIGHT_CFG = Path(
    "source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/wheeled/unitree_go2w/height_env_cfg.py"
)


@dataclass(frozen=True)
class JointInfo:
    name: str
    lower: float
    upper: float


@dataclass(frozen=True)
class LegModel:
    name: str
    hip_origin: np.ndarray
    abduction_offset: float
    thigh_len: float
    calf_len: float
    hip_limit: JointInfo
    thigh_limit: JointInfo
    calf_limit: JointInfo


@dataclass
class LegSolution:
    q: np.ndarray
    fk_error: float
    hard_margin: float
    soft_margin: float
    default_deviation: float


@dataclass
class HeightResult:
    height: float
    hard_ok: bool
    soft_ok: bool
    dynamic_ok: bool
    min_hard_margin: float
    min_soft_margin: float
    max_default_deviation: float
    max_fk_error: float
    base_ground_clearance: float
    limiting_leg: str
    q_by_leg: dict[str, np.ndarray]


def parse_xyz(text: str | None) -> np.ndarray:
    if not text:
        return np.zeros(3)
    return np.array([float(v) for v in text.split()], dtype=float)


def rot_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def fk_wheel_center(leg: LegModel, q: np.ndarray) -> np.ndarray:
    hip, thigh, calf = q
    offset = np.array([0.0, leg.abduction_offset, 0.0])
    thigh_vec = np.array([0.0, 0.0, -leg.thigh_len])
    calf_vec = np.array([0.0, 0.0, -leg.calf_len])
    return leg.hip_origin + rot_x(hip) @ (offset + rot_y(thigh) @ (thigh_vec + rot_y(calf) @ calf_vec))


def joint_margin(q: float, joint: JointInfo) -> float:
    return min(q - joint.lower, joint.upper - q)


def soft_limits(joint: JointInfo, factor: float) -> tuple[float, float]:
    center = 0.5 * (joint.lower + joint.upper)
    half_width = 0.5 * (joint.upper - joint.lower) * factor
    return center - half_width, center + half_width


def soft_joint_margin(q: float, joint: JointInfo, factor: float) -> float:
    lower, upper = soft_limits(joint, factor)
    return min(q - lower, upper - q)


def within(q: float, lower: float, upper: float, eps: float = 1.0e-9) -> bool:
    return lower - eps <= q <= upper + eps


def parse_urdf(urdf_path: Path) -> tuple[dict[str, LegModel], float]:
    root = ET.parse(urdf_path).getroot()
    joints = {j.get("name"): j for j in root.findall("joint")}

    def origin(joint_name: str) -> np.ndarray:
        joint = joints[joint_name]
        origin_node = joint.find("origin")
        return parse_xyz(origin_node.get("xyz") if origin_node is not None else None)

    def limit(joint_name: str) -> JointInfo:
        joint = joints[joint_name]
        limit_node = joint.find("limit")
        if limit_node is None:
            raise ValueError(f"Joint {joint_name} has no limit.")
        return JointInfo(
            name=joint_name,
            lower=float(limit_node.get("lower")),
            upper=float(limit_node.get("upper")),
        )

    base_half_z = 0.0
    for link in root.findall("link"):
        if link.get("name") != "base":
            continue
        for collision in link.findall("collision"):
            box = collision.find("geometry/box")
            if box is not None:
                base_half_z = max(base_half_z, float(box.get("size").split()[2]) * 0.5)

    legs: dict[str, LegModel] = {}
    for leg_name in LEG_NAMES:
        hip_origin = origin(f"{leg_name}_hip_joint")
        abduction_offset = float(origin(f"{leg_name}_thigh_joint")[1])
        thigh_len = abs(float(origin(f"{leg_name}_calf_joint")[2]))
        calf_len = abs(float(origin(f"{leg_name}_foot_joint")[2]))
        legs[leg_name] = LegModel(
            name=leg_name,
            hip_origin=hip_origin,
            abduction_offset=abduction_offset,
            thigh_len=thigh_len,
            calf_len=calf_len,
            hip_limit=limit(f"{leg_name}_hip_joint"),
            thigh_limit=limit(f"{leg_name}_thigh_joint"),
            calf_limit=limit(f"{leg_name}_calf_joint"),
        )
    return legs, base_half_z


def resolve_package_mesh(urdf_path: Path, filename: str) -> Path | None:
    if not filename.startswith("package://"):
        path = Path(filename)
        return path if path.exists() else None
    rel = filename.removeprefix("package://")
    package, _, rest = rel.partition("/")
    for parent in [urdf_path.parent, *urdf_path.parents]:
        candidate = parent / rest
        if candidate.exists():
            return candidate
        candidate = parent / package / rest
        if candidate.exists():
            return candidate
    return None


def mesh_bounds_from_dae(dae_path: Path) -> tuple[np.ndarray, np.ndarray]:
    root = ET.parse(dae_path).getroot()
    ns = {"c": root.tag.split("}")[0].strip("{")} if root.tag.startswith("{") else {}
    arrays = root.findall(".//c:float_array", ns) if ns else root.findall(".//float_array")
    candidates = []
    for array in arrays:
        text = array.text or ""
        values = np.fromstring(text, sep=" ")
        if values.size < 9 or values.size % 3 != 0:
            continue
        points = values.reshape(-1, 3)
        extent = points.max(axis=0) - points.min(axis=0)
        array_id = array.get("id", "").lower()
        score = float(np.linalg.norm(extent))
        if "position" in array_id or "positions" in array_id:
            score += 1000.0
        candidates.append((score, points))
    if not candidates:
        raise ValueError(f"No vertex float arrays found in {dae_path}")
    points = max(candidates, key=lambda item: item[0])[1]
    return points.min(axis=0), points.max(axis=0)


def estimate_wheel_radius(urdf_path: Path) -> float:
    root = ET.parse(urdf_path).getroot()
    radii = []
    for link in root.findall("link"):
        if not link.get("name", "").endswith("_foot"):
            continue
        for mesh in link.findall("collision/geometry/mesh"):
            filename = mesh.get("filename")
            if not filename:
                continue
            mesh_path = resolve_package_mesh(urdf_path, filename)
            if mesh_path is None:
                continue
            mn, mx = mesh_bounds_from_dae(mesh_path)
            extent = mx - mn
            # Wheel axis is the URDF y-axis; radius is in the x-z cross-section.
            radii.append(0.5 * max(float(extent[0]), float(extent[2])))
    if not radii:
        raise ValueError("Could not estimate wheel radius from foot collision meshes.")
    return float(np.mean(radii))


def hip_candidates_for_target(leg: LegModel, target: np.ndarray) -> Iterable[float]:
    r = target - leg.hip_origin
    ry, rz = float(r[1]), float(r[2])
    radius = math.hypot(ry, rz)
    if radius < 1.0e-12:
        return []
    ratio = leg.abduction_offset / radius
    if abs(ratio) > 1.0 + 1.0e-9:
        return []
    ratio = max(-1.0, min(1.0, ratio))
    phi = math.atan2(rz, ry)
    alpha = math.acos(ratio)
    return [phi + alpha, phi - alpha]


def pitch_ik_candidates(leg: LegModel, x: float, z: float) -> Iterable[tuple[float, float]]:
    l1, l2 = leg.thigh_len, leg.calf_len
    d2 = x * x + z * z
    cos_calf = (d2 - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    if cos_calf < -1.0 - 1.0e-9 or cos_calf > 1.0 + 1.0e-9:
        return []
    cos_calf = max(-1.0, min(1.0, cos_calf))
    calf_abs = math.acos(cos_calf)
    candidates = []
    for calf in (-calf_abs, calf_abs):
        k1 = l1 + l2 * math.cos(calf)
        k2 = l2 * math.sin(calf)
        thigh = math.atan2(-x, -z) - math.atan2(k2, k1)
        thigh = math.atan2(math.sin(thigh), math.cos(thigh))
        candidates.append((thigh, calf))
    return candidates


def solve_leg_ik(
    leg: LegModel,
    target: np.ndarray,
    soft_factor: float,
    default_q: np.ndarray,
) -> LegSolution | None:
    solutions: list[LegSolution] = []
    for hip in hip_candidates_for_target(leg, target):
        hip = math.atan2(math.sin(hip), math.cos(hip))
        if not within(hip, leg.hip_limit.lower, leg.hip_limit.upper):
            continue
        r_hip = rot_x(-hip) @ (target - leg.hip_origin)
        pitch_target = r_hip - np.array([0.0, leg.abduction_offset, 0.0])
        if abs(float(pitch_target[1])) > 1.0e-6:
            continue
        for thigh, calf in pitch_ik_candidates(leg, float(pitch_target[0]), float(pitch_target[2])):
            if not within(thigh, leg.thigh_limit.lower, leg.thigh_limit.upper):
                continue
            if not within(calf, leg.calf_limit.lower, leg.calf_limit.upper):
                continue
            q = np.array([hip, thigh, calf], dtype=float)
            err = float(np.linalg.norm(fk_wheel_center(leg, q) - target))
            hard_margin = min(
                joint_margin(hip, leg.hip_limit),
                joint_margin(thigh, leg.thigh_limit),
                joint_margin(calf, leg.calf_limit),
            )
            soft_margin = min(
                soft_joint_margin(hip, leg.hip_limit, soft_factor),
                soft_joint_margin(thigh, leg.thigh_limit, soft_factor),
                soft_joint_margin(calf, leg.calf_limit, soft_factor),
            )
            default_deviation = float(np.max(np.abs(q - default_q)))
            solutions.append(
                LegSolution(
                    q=q,
                    fk_error=err,
                    hard_margin=hard_margin,
                    soft_margin=soft_margin,
                    default_deviation=default_deviation,
                )
            )
    if not solutions:
        return None
    return min(solutions, key=lambda s: (s.default_deviation, s.fk_error))


def contiguous_ranges(values: list[float], flags: list[bool]) -> list[tuple[float, float]]:
    ranges: list[tuple[float, float]] = []
    start: float | None = None
    prev: float | None = None
    for value, ok in zip(values, flags):
        if ok and start is None:
            start = value
        if not ok and start is not None and prev is not None:
            ranges.append((start, prev))
            start = None
        prev = value
    if start is not None and prev is not None:
        ranges.append((start, prev))
    return ranges


def parse_height_cfg_ranges(path: Path) -> dict[str, tuple[float, float]]:
    if not path.exists():
        return {}
    text = path.read_text()
    constants: dict[str, float] = {}
    for match in re.finditer(r"^([A-Z_]+)\s*=\s*([-+]?\d+(?:\.\d+)?)\s*$", text, re.MULTILINE):
        constants[match.group(1)] = float(match.group(2))

    def number_or_constant(raw: str) -> float:
        value = raw.strip()
        if value in constants:
            return constants[value]
        return float(value)

    out = {}
    for name in ("STAGE_A_HEIGHT_RANGE", "INITIAL_HEIGHT_RANGE", "FINAL_HEIGHT_RANGE"):
        match = re.search(rf"{name}\s*=\s*\(([^,]+),\s*([^)]+)\)", text)
        if match:
            out[name] = (number_or_constant(match.group(1)), number_or_constant(match.group(2)))
    return out


def evaluate_heights(
    legs: dict[str, LegModel],
    wheel_radius: float,
    base_half_z: float,
    height_values: list[float],
    soft_factor: float,
    joint_margin_required: float,
    soft_margin_required: float,
    base_ground_margin: float,
) -> tuple[list[HeightResult], float, dict[str, np.ndarray]]:
    default_q = np.array([DEFAULT_JOINTS["hip"], DEFAULT_JOINTS["thigh"], DEFAULT_JOINTS["calf"]])
    default_support: dict[str, np.ndarray] = {}
    default_heights = []
    for name, leg in legs.items():
        default_foot = fk_wheel_center(leg, default_q)
        default_support[name] = default_foot.copy()
        default_heights.append(wheel_radius - float(default_foot[2]))
    default_base_height = float(np.mean(default_heights))

    results = []
    for height in height_values:
        q_by_leg: dict[str, np.ndarray] = {}
        max_fk_error = 0.0
        min_hard_margin = float("inf")
        min_soft_margin = float("inf")
        max_default_deviation = 0.0
        limiting_leg = ""
        all_hard_ok = True
        for name, leg in legs.items():
            target = default_support[name].copy()
            target[2] = wheel_radius - height
            sol = solve_leg_ik(leg, target, soft_factor, default_q)
            if sol is None:
                all_hard_ok = False
                limiting_leg = name
                break
            q_by_leg[name] = sol.q
            max_fk_error = max(max_fk_error, sol.fk_error)
            if sol.hard_margin < min_hard_margin:
                min_hard_margin = sol.hard_margin
                limiting_leg = name
            min_soft_margin = min(min_soft_margin, sol.soft_margin)
            max_default_deviation = max(max_default_deviation, sol.default_deviation)

        base_ground_clearance = height - base_half_z
        hard_ok = all_hard_ok and max_fk_error < 1.0e-5 and base_ground_clearance >= -1.0e-9
        soft_ok = hard_ok and min_soft_margin >= -1.0e-9
        dynamic_ok = (
            soft_ok
            and min_hard_margin >= joint_margin_required
            and min_soft_margin >= soft_margin_required
            and base_ground_clearance >= base_ground_margin
        )
        results.append(
            HeightResult(
                height=height,
                hard_ok=hard_ok,
                soft_ok=soft_ok,
                dynamic_ok=dynamic_ok,
                min_hard_margin=min_hard_margin if math.isfinite(min_hard_margin) else float("nan"),
                min_soft_margin=min_soft_margin if math.isfinite(min_soft_margin) else float("nan"),
                max_default_deviation=max_default_deviation,
                max_fk_error=max_fk_error,
                base_ground_clearance=base_ground_clearance,
                limiting_leg=limiting_leg,
                q_by_leg=q_by_leg,
            )
        )
    return results, default_base_height, default_support


def fmt_range(ranges: list[tuple[float, float]]) -> str:
    if not ranges:
        return "none"
    return ", ".join(f"{lo:.3f} - {hi:.3f} m ({lo*100:.1f} - {hi*100:.1f} cm)" for lo, hi in ranges)


def write_csv(path: Path, results: list[HeightResult]) -> None:
    fieldnames = [
        "height_m",
        "hard_ok",
        "soft_ok",
        "dynamic_ok",
        "min_hard_margin_rad",
        "min_soft_margin_rad",
        "max_default_deviation_rad",
        "max_fk_error_m",
        "base_ground_clearance_m",
        "limiting_leg",
        "FL_q",
        "FR_q",
        "RL_q",
        "RR_q",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            row = {
                "height_m": item.height,
                "hard_ok": item.hard_ok,
                "soft_ok": item.soft_ok,
                "dynamic_ok": item.dynamic_ok,
                "min_hard_margin_rad": item.min_hard_margin,
                "min_soft_margin_rad": item.min_soft_margin,
                "max_default_deviation_rad": item.max_default_deviation,
                "max_fk_error_m": item.max_fk_error,
                "base_ground_clearance_m": item.base_ground_clearance,
                "limiting_leg": item.limiting_leg,
            }
            for leg_name in LEG_NAMES:
                q = item.q_by_leg.get(leg_name)
                row[f"{leg_name}_q"] = "" if q is None else " ".join(f"{v:.8f}" for v in q)
            writer.writerow(row)


def print_boundary_solution(label: str, results: list[HeightResult], predicate: str) -> None:
    ok_items = [r for r in results if getattr(r, predicate)]
    if not ok_items:
        return
    for item in (ok_items[0], ok_items[-1]):
        print(f"\n{label} boundary at {item.height:.3f} m:")
        print(
            f"  min_hard_margin={item.min_hard_margin:.3f} rad, "
            f"min_soft_margin={item.min_soft_margin:.3f} rad, "
            f"base_clearance={item.base_ground_clearance:.3f} m, limiting_leg={item.limiting_leg}"
        )
        for leg_name in LEG_NAMES:
            q = item.q_by_leg.get(leg_name)
            if q is not None:
                print(f"  {leg_name}: hip={q[0]: .3f}, thigh={q[1]: .3f}, calf={q[2]: .3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--height-cfg", type=Path, default=DEFAULT_HEIGHT_CFG)
    parser.add_argument("--height-min", type=float, default=0.05)
    parser.add_argument("--height-max", type=float, default=0.55)
    parser.add_argument("--height-step", type=float, default=0.001)
    parser.add_argument("--soft-joint-factor", type=float, default=0.9)
    parser.add_argument("--joint-margin", type=float, default=0.10)
    parser.add_argument("--soft-margin", type=float, default=0.03)
    parser.add_argument("--base-ground-margin", type=float, default=0.03)
    parser.add_argument("--csv", type=Path, default=Path("/tmp/go2w_height_ik_range.csv"))
    args = parser.parse_args()

    if args.height_step <= 0.0:
        raise ValueError("--height-step must be positive.")

    legs, base_half_z = parse_urdf(args.urdf)
    wheel_radius = estimate_wheel_radius(args.urdf)
    count = int(math.floor((args.height_max - args.height_min) / args.height_step)) + 1
    heights = [args.height_min + i * args.height_step for i in range(count + 1)]
    heights = [h for h in heights if h <= args.height_max + 1.0e-12]
    results, default_height, default_support = evaluate_heights(
        legs=legs,
        wheel_radius=wheel_radius,
        base_half_z=base_half_z,
        height_values=heights,
        soft_factor=args.soft_joint_factor,
        joint_margin_required=args.joint_margin,
        soft_margin_required=args.soft_margin,
        base_ground_margin=args.base_ground_margin,
    )
    write_csv(args.csv, results)

    hard_ranges = contiguous_ranges([r.height for r in results], [r.hard_ok for r in results])
    soft_ranges = contiguous_ranges([r.height for r in results], [r.soft_ok for r in results])
    dynamic_ranges = contiguous_ranges([r.height for r in results], [r.dynamic_ok for r in results])
    current_ranges = parse_height_cfg_ranges(args.height_cfg)

    print("Go2W base-height IK range analysis")
    print("=" * 40)
    print(f"URDF: {args.urdf}")
    print(f"Wheel radius from mesh: {wheel_radius:.6f} m")
    print(f"Base collision half height: {base_half_z:.6f} m")
    print(f"Default IK base height from default joints: {default_height:.6f} m")
    print(f"Soft joint factor: {args.soft_joint_factor:.3f}")
    print(f"Dynamic joint margin: {args.joint_margin:.3f} rad")
    print(f"Dynamic soft-limit margin: {args.soft_margin:.3f} rad")
    print(f"Dynamic base-ground margin: {args.base_ground_margin:.3f} m")
    print("\nDefault wheel-center support points in base frame:")
    for name in LEG_NAMES:
        p = default_support[name]
        print(f"  {name}: x={p[0]: .4f}, y={p[1]: .4f}, z={p[2]: .4f}")

    print("\nRanges")
    print(f"  hard_ik:             {fmt_range(hard_ranges)}")
    print(f"  soft_ik:             {fmt_range(soft_ranges)}")
    print(f"  dynamic_recommended: {fmt_range(dynamic_ranges)}")
    print(f"  detailed CSV:        {args.csv}")

    if current_ranges:
        print("\nCurrent height_env_cfg ranges")
        for name, value in current_ranges.items():
            print(f"  {name}: {value[0]:.3f} - {value[1]:.3f} m ({value[0]*100:.1f} - {value[1]*100:.1f} cm)")

    print_boundary_solution("hard_ik", results, "hard_ok")
    print_boundary_solution("soft_ik", results, "soft_ok")
    print_boundary_solution("dynamic_recommended", results, "dynamic_ok")

    print("\nInterpretation")
    print("  hard_ik only checks URDF joint limits and base-ground contact clearance.")
    print("  soft_ik uses IsaacLab's soft_joint_pos_limit_factor and is closer to training constraints.")
    print("  dynamic_recommended adds hard/soft joint and base-ground margins; use this as the safer command range seed.")


if __name__ == "__main__":
    main()
