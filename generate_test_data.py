"""Build deterministic descent sequences from the three supplied seed photos."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
FRAMES = 10
INTERVAL_S = 0.5

# End drift is measured as a fraction of the seed dimensions. Every authored
# path stays inside its final crop window, so generated frames never need padding.
SPECS = {
    "mars_base_retarget": {
        "title": "Mars Base Emergency Retarget", "mode": "RECOVERY",
        "description": "Emergency descent near a Mars research base with lateral drift.",
        "gravity": 3.71, "end_scale": .56, "drift": (.155, -.115), "curve": .035,
    },
    "lunar_crater_safe": {
        "title": "Lunar Crater Safe Descent", "mode": "SAFETY",
        "description": "A stable descent toward a flat lunar plateau.",
        "gravity": 1.62, "end_scale": .60, "drift": (.018, -.012), "curve": .004,
    },
    "emergency_recovery_zone": {
        "title": "Emergency Recovery Zone", "mode": "RECOVERY",
        "description": "Facility-aware recovery over an arid operations area.",
        "gravity": 9.81, "end_scale": .58, "drift": (-.095, .072), "curve": -.016,
    },
}


def output_size(source: Image.Image) -> tuple[int, int]:
    """Use 960² for square seeds; otherwise preserve aspect ratio at width 1280."""
    width, height = source.size
    if width == height:
        return 960, 960
    return 1280, max(1, round(1280 * height / width))


def crop_trajectory(spec: dict, index: int) -> tuple[float, float, float]:
    """Return normalized center x/y and crop scale for a smooth descent."""
    t = index / (FRAMES - 1)
    smooth = t * t * (3 - 2 * t)
    # The Mars path accelerates more strongly; the others remain near-linear.
    progress = smooth ** (1.35 if spec["mode"] == "RECOVERY" and spec["gravity"] < 5 else 1.0)
    dx, dy = spec["drift"]
    center_x = .5 + dx * progress
    center_y = .5 + dy * progress + spec["curve"] * math.sin(math.pi * smooth)
    scale = 1.0 + (spec["end_scale"] - 1.0) * smooth
    return center_x, center_y, scale


def generate(name: str, spec: dict) -> None:
    seed_path = ROOT / "seed_photos" / f"{name}.png"
    if not seed_path.is_file():
        raise FileNotFoundError(f"Missing seed image:\n{seed_path}")
    with Image.open(seed_path) as loaded:
        seed = loaded.convert("RGB")
    src_w, src_h = seed.size
    dst_w, dst_h = output_size(seed)
    out = ROOT / "scenarios" / name
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("*.png"):
        old.unlink()

    rows, previews, prior_center = [], [], None
    for i in range(FRAMES):
        center_x_norm, center_y_norm, scale = crop_trajectory(spec, i)
        crop_w, crop_h = max(2, round(src_w * scale)), max(2, round(src_h * scale))
        center_x, center_y = center_x_norm * src_w, center_y_norm * src_h
        center_x = min(max(center_x, crop_w / 2), src_w - crop_w / 2)
        center_y = min(max(center_y, crop_h / 2), src_h - crop_h / 2)
        left, top = round(center_x - crop_w / 2), round(center_y - crop_h / 2)
        frame = seed.crop((left, top, left + crop_w, top + crop_h))
        if frame.size != (dst_w, dst_h):
            frame = frame.resize((dst_w, dst_h), Image.Resampling.LANCZOS)
        filename = f"{i:06d}.png"
        frame.save(frames_dir / filename, optimize=True)
        previews.append(frame.resize((480, round(480 * dst_h / dst_w)), Image.Resampling.LANCZOS))

        altitude = 120.0 + (30.0 - 120.0) * (i / (FRAMES - 1))
        if prior_center is None:
            velocity_x = velocity_y = 0.0
        else:
            ground_width = 2 * altitude * math.tan(math.radians(78) / 2)
            velocity_x = (center_x_norm - prior_center[0]) * ground_width / INTERVAL_S
            velocity_y = (center_y_norm - prior_center[1]) * ground_width / INTERVAL_S
        prior_center = (center_x_norm, center_y_norm)
        rows.append({
            "frame_id": i, "filename": filename, "timestamp_s": round(i * INTERVAL_S, 2),
            "altitude_m": round(altitude, 1), "velocity_x_mps": round(velocity_x, 2),
            "velocity_y_mps": round(velocity_y, 2), "velocity_z_mps": -5.0,
            "roll_deg": round(.08 * math.sin(math.pi * i / 9), 2),
            "pitch_deg": round(.08 * math.sin(math.pi * i / 9 + .4), 2),
            "yaw_deg": round(15 + .18 * i, 2), "camera_hfov_deg": 78.0,
            "camera_vfov_deg": round(math.degrees(2 * math.atan(math.tan(math.radians(78) / 2) * src_h / src_w)), 2),
            "crop_center_x_norm": round(center_x / src_w, 6),
            "crop_center_y_norm": round(center_y / src_h, 6), "crop_scale": round(scale, 6),
        })

    with (out / "metadata.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    scenario_path = out / "scenario.json"
    scenario = json.loads(scenario_path.read_text()) if scenario_path.exists() else {}
    scenario.update({"scenario_name": name, "title": spec["title"], "mission_mode": spec["mode"],
                     "description": spec["description"], "gravity_mps2": spec["gravity"]})
    scenario.setdefault("vehicle_radius_m", 2.5)
    scenario_path.write_text(json.dumps(scenario, indent=2) + "\n")

    preview_dir = ROOT / "outputs" / "test_generation_preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(preview_dir / f"{name}_preview.gif",
                    [np.asarray(frame) for frame in previews], duration=.55, loop=0)
    thumb_w, thumb_h = previews[0].size
    sheet = Image.new("RGB", (thumb_w * 5, thumb_h * 2))
    for index, frame in enumerate(previews):
        sheet.paste(frame, ((index % 5) * thumb_w, (index // 5) * thumb_h))
    sheet.save(preview_dir / f"{name}_contact_sheet.jpg", quality=91, optimize=True)
    print(f"Generated {name}: {FRAMES} frames at {dst_w}x{dst_h} from {seed_path.name}")


if __name__ == "__main__":
    for scenario_name, scenario_spec in SPECS.items():
        generate(scenario_name, scenario_spec)
