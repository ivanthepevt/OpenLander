"""Generate deterministic 150-frame evaluation scenarios from every real seed."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
SEED_DIR = ROOT / "seed_photos"
SCENARIO_DIR = ROOT / "scenarios"
PREVIEW_DIR = ROOT / "outputs" / "test_generation_preview"
EVALUATION_DIR = ROOT / "outputs" / "evaluation"
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
FRAMES = 150
FPS = 30
INTERVAL_S = 1 / FPS


def discover_seeds() -> list[Path]:
    """Return all supported, non-hidden seed images in stable filename order."""
    if not SEED_DIR.is_dir():
        raise FileNotFoundError(f"Missing seed directory:\n{SEED_DIR}")
    return sorted(
        (p for p in SEED_DIR.iterdir()
         if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in SUPPORTED_EXTENSIONS),
        key=lambda p: p.name.casefold(),
    )


def clean_stem(filename: str) -> str:
    """Remove repeated image extensions and normalize to a safe scenario id."""
    value = filename
    while Path(value).suffix.lower() in SUPPORTED_EXTENSIONS:
        value = value[: -len(Path(value).suffix)]
    value = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return value or "scenario"


def seed_manifest(seeds: list[Path]) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in seeds}


def scenario_ids(seeds: list[Path]) -> dict[Path, str]:
    """Assign deterministic unique ids even if two filenames sanitize alike."""
    assigned: dict[Path, str] = {}
    used: set[str] = set()
    for seed in seeds:
        base = clean_stem(seed.name)
        candidate = base
        number = 2
        while candidate in used:
            candidate = f"{base}_{number}"
            number += 1
        used.add(candidate)
        assigned[seed] = candidate
    return assigned


def output_size(source: Image.Image) -> tuple[int, int]:
    """Preserve aspect ratio, using 960 square or a 1280-pixel width."""
    width, height = source.size
    if width == height:
        return 960, 960
    return 1280, max(2, round(1280 * height / width))


def inferred_mode(name: str) -> str:
    recovery = ("parking", "warehouse", "suburban", "dockyard", "harbor", "road", "urban", "industrial")
    return "RECOVERY" if any(hint in name for hint in recovery) else "SAFETY"


def starting_altitude(name: str) -> float:
    match = re.search(r"(?:^|_)(\d{2,4})m(?:_|$)", name)
    return float(match.group(1)) if match else 400.0


def generation_spec(seed: Path, scenario_name: str, source_size: tuple[int, int]) -> dict:
    """Derive a reproducible bounded motion profile from the scenario identity."""
    digest = hashlib.sha256(scenario_name.encode()).digest()
    side = "side" in scenario_name
    straight = "straight" in scenario_name
    size_guard = .02 if min(source_size) < 1000 else 0.0
    end_scale = min(.69, .61 + (digest[0] / 255) * .06 + size_guard)
    margin = (1 - end_scale) / 2
    strength_range = (.52, .72) if side else ((.25, .42) if straight else (.38, .58))
    strength = strength_range[0] + (digest[1] / 255) * (strength_range[1] - strength_range[0])
    angle = (digest[2] / 255) * math.tau
    drift_radius = margin * strength
    dx, dy = drift_radius * math.cos(angle), drift_radius * math.sin(angle)
    curve = margin * (.035 + .04 * digest[3] / 255) * (-1 if digest[4] & 1 else 1)
    altitude = starting_altitude(scenario_name)
    return {
        "source_seed": seed.name,
        "title": scenario_name.replace("_", " ").title(),
        "mode": inferred_mode(scenario_name),
        "start_altitude": altitude,
        "final_altitude": altitude * (.22 + .10 * digest[5] / 255),
        "end_scale": end_scale,
        "drift": (dx, dy),
        "curve": curve,
        "side": side,
        "straight": straight,
        "yaw_start": float((int.from_bytes(digest[6:8], "big") / 65535) * 360),
    }


def crop_trajectory(spec: dict, index: int) -> tuple[float, float, float]:
    """Return a continuous normalized crop center and scale."""
    t = index / (FRAMES - 1)
    smooth = .5 - .5 * math.cos(math.pi * t)
    zoom_progress = smooth ** 1.22
    scale = 1 + (spec["end_scale"] - 1) * zoom_progress
    dx, dy = spec["drift"]
    drift_progress = zoom_progress * (.88 + .12 * smooth)
    curve_progress = math.sin(math.pi * t) * zoom_progress
    center_x = .5 + dx * drift_progress - spec["curve"] * curve_progress * math.sin(math.pi / 4)
    center_y = .5 + dy * drift_progress + spec["curve"] * curve_progress * math.cos(math.pi / 4)
    margin = (1 - scale) / 2
    return min(max(center_x, .5 - margin), .5 + margin), min(max(center_y, .5 - margin), .5 + margin), scale


def export_preview(frames_dir: Path, destination: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to create raw evaluation preview MP4s.")
    command = [
        ffmpeg, "-y", "-loglevel", "error", "-framerate", str(FPS),
        "-i", str(frames_dir / "%06d.png"), "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(destination),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode or not destination.is_file():
        raise RuntimeError(f"Preview export failed: {completed.stderr.strip()}")


def generate(seed_path: Path, name: str) -> Path:
    with Image.open(seed_path) as loaded:
        seed = loaded.convert("RGB")
    src_w, src_h = seed.size
    dst_w, dst_h = output_size(seed)
    spec = generation_spec(seed_path, name, seed.size)
    out = SCENARIO_DIR / name
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("*.png"):
        old.unlink()

    rows: list[dict] = []
    centers: list[tuple[float, float]] = []
    altitudes: list[float] = []
    vertical_fov = math.degrees(2 * math.atan(math.tan(math.radians(78) / 2) * src_h / src_w))
    for index in range(FRAMES):
        center_x_norm, center_y_norm, scale = crop_trajectory(spec, index)
        crop_w, crop_h = max(2.0, src_w * scale), max(2.0, src_h * scale)
        center_x, center_y = center_x_norm * src_w, center_y_norm * src_h
        left, top = center_x - crop_w / 2, center_y - crop_h / 2
        frame = seed.resize((dst_w, dst_h), Image.Resampling.LANCZOS,
                            box=(left, top, left + crop_w, top + crop_h))
        filename = f"{index:06d}.png"
        frame.save(frames_dir / filename, compress_level=6)

        t = index / (FRAMES - 1)
        descent = .5 - .5 * math.cos(math.pi * t)
        altitude = spec["start_altitude"] + (spec["final_altitude"] - spec["start_altitude"]) * descent
        centers.append((center_x_norm, center_y_norm))
        altitudes.append(altitude)
        if index == 0:
            next_center = crop_trajectory(spec, 1)[:2]
            next_descent = .5 - .5 * math.cos(math.pi / (FRAMES - 1))
            next_altitude = spec["start_altitude"] + (spec["final_altitude"] - spec["start_altitude"]) * next_descent
            delta_x, delta_y = next_center[0] - center_x_norm, next_center[1] - center_y_norm
            velocity_z = (next_altitude - altitude) / INTERVAL_S
        else:
            delta_x, delta_y = center_x_norm - centers[index - 1][0], center_y_norm - centers[index - 1][1]
            velocity_z = (altitude - altitudes[index - 1]) / INTERVAL_S
        ground_width = 2 * altitude * math.tan(math.radians(78) / 2)
        base_pitch = -12.0 if spec["side"] else 0.0
        base_roll = 1.5 if spec["side"] else 0.0
        attitude_amplitude = .65 if spec["side"] else (.12 if spec["straight"] else .25)
        rows.append({
            "frame_id": index, "filename": filename, "timestamp_s": f"{index / FPS:.3f}",
            "altitude_m": round(altitude, 2),
            "velocity_x_mps": round(delta_x * ground_width / INTERVAL_S, 2),
            "velocity_y_mps": round(delta_y * ground_width / INTERVAL_S, 2),
            "velocity_z_mps": round(velocity_z, 2),
            "roll_deg": round(base_roll + attitude_amplitude * math.sin(math.pi * t), 2),
            "pitch_deg": round(base_pitch + attitude_amplitude * math.sin(math.pi * t + .35), 2),
            "yaw_deg": round((spec["yaw_start"] + 2.0 * descent) % 360, 2),
            "camera_hfov_deg": 78.0, "camera_vfov_deg": round(vertical_fov, 2),
            "crop_center_x_norm": round(center_x_norm, 7),
            "crop_center_y_norm": round(center_y_norm, 7), "crop_scale": round(scale, 7),
        })

    with (out / "metadata.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    scenario = {
        "scenario_name": name, "title": spec["title"], "source_seed": seed_path.name,
        "mission_mode": spec["mode"],
        "description": "Real evaluation seed converted into a simulated 5-second landing descent.",
        "starting_altitude_m": spec["start_altitude"], "frame_count": FRAMES, "fps": FPS,
        "source_type": "real_seed_photo", "source_width": src_w, "source_height": src_h,
        "final_crop_scale": round(spec["end_scale"], 7), "vehicle_radius_m": 2.5,
    }
    (out / "scenario.json").write_text(json.dumps(scenario, indent=2) + "\n")
    preview = PREVIEW_DIR / f"{name}_preview.mp4"
    export_preview(frames_dir, preview)
    print(f"Generated {name}: {FRAMES} frames at {dst_w}x{dst_h} from {seed_path.name}", flush=True)
    return out


def main() -> None:
    seeds = discover_seeds()
    if not seeds:
        raise RuntimeError(f"No supported seed images found in {SEED_DIR}")
    mapping = scenario_ids(seeds)
    before = seed_manifest(seeds)
    print(f"Seeds discovered: {len(seeds)}", flush=True)
    SCENARIO_DIR.mkdir(exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    generated = [generate(seed, mapping[seed]) for seed in seeds]
    after = seed_manifest(seeds)
    unchanged = before == after
    integrity = {
        "verified_at": datetime.now().isoformat(timespec="seconds"), "seed_count": len(seeds),
        "unchanged": unchanged, "before": before, "after": after,
    }
    (EVALUATION_DIR / "seed_integrity.json").write_text(json.dumps(integrity, indent=2) + "\n")
    if not unchanged:
        raise RuntimeError("Seed integrity check failed: one or more source images changed.")
    print(f"Scenarios generated: {len(generated)}", flush=True)
    print("Seed integrity: VERIFIED UNCHANGED", flush=True)


if __name__ == "__main__":
    main()
