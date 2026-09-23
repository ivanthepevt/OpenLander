"""Run the real-seed OpenLander evaluation sequentially with shared AI models."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import engine

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "evaluation"
INTEGRITY_PATH = OUTPUT_DIR / "seed_integrity.json"
BATCH_ID = "real_seed_batch_2026_09"
COLUMNS = [
    "scenario_name", "source_seed", "mission_mode", "frame_count", "resolution",
    "starting_altitude_m", "device", "total_analysis_time_s",
    "average_analysis_ms_per_frame", "final_state", "final_selected_lz", "final_score",
    "retarget_count", "target_lock_timestamp_s", "max_visual_drift_px_s",
    "average_visual_drift_px_s", "detected_object_count_total", "run_folder",
    "landing_video", "dashboard_video", "success", "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OpenLander on every generated real-seed scenario.")
    parser.add_argument("--scenario", help="Analyze one normalized scenario name only.")
    parser.add_argument("--force", action="store_true", help="Rerun scenarios already completed in this batch report.")
    return parser.parse_args()


def current_seed_hashes() -> dict[str, str]:
    extensions = {".png", ".jpg", ".jpeg", ".webp"}
    seeds = sorted(
        p for p in (ROOT / "seed_photos").iterdir()
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in extensions
    )
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in seeds}


def discover_scenarios(selected: str | None) -> list[Path]:
    current_seeds = set(current_seed_hashes())
    found = []
    for folder in sorted((ROOT / "scenarios").iterdir()):
        config = folder / "scenario.json"
        if not folder.is_dir() or not config.is_file():
            continue
        scenario = json.loads(config.read_text())
        if scenario.get("source_type") != "real_seed_photo" or scenario.get("source_seed") not in current_seeds:
            continue
        if selected and scenario.get("scenario_name") != selected:
            continue
        found.append(folder)
    if selected and not found:
        raise ValueError(f"Unknown generated scenario: {selected}")
    return found


def environment_label(scenario_name: str) -> str:
    tokens = [
        token for token in scenario_name.split("_")
        if not token.isdigit() and token not in {"side", "straight"}
        and not (token.endswith("m") and token[:-1].isdigit())
    ]
    return " ".join(tokens).title() or scenario_name


def video_metadata(path: Path) -> dict:
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,r_frame_rate,nb_frames:format=duration",
        "-of", "json", str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(f"ffprobe failed for {path.name}: {completed.stderr.strip()}")
    payload = json.loads(completed.stdout)
    stream = payload["streams"][0]
    numerator, denominator = map(float, stream["r_frame_rate"].split("/"))
    return {
        "codec": stream.get("codec_name"), "width": int(stream["width"]),
        "height": int(stream["height"]), "fps": numerator / denominator,
        "frame_count": int(stream["nb_frames"]), "duration_s": float(payload["format"]["duration"]),
    }


def verify_video(path: Path, width: int, height: int) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Missing or empty export: {path}")
    info = video_metadata(path)
    expected = (width + width % 2, height + height % 2)
    valid = (
        info["codec"] == "h264" and info["frame_count"] == 150
        and abs(info["fps"] - 30) < .01 and abs(info["duration_s"] - 5) < .15
        and (info["width"], info["height"]) == expected
    )
    if not valid:
        raise RuntimeError(f"Invalid video metadata for {path.name}: {info}")
    return info


def success_row(scenario: dict, run_folder: Path, landing: Path, dashboard: Path) -> tuple[dict, dict]:
    run = json.loads((run_folder / "run.json").read_text())
    result = json.loads((run_folder / "result.json").read_text())
    frames = result["frames"]
    summary = result["mission_summary"]
    drifts = [float(frame["drift"].get("visual_drift_speed", 0)) for frame in frames]
    row = {
        "scenario_name": scenario["scenario_name"], "source_seed": scenario["source_seed"],
        "mission_mode": scenario["mission_mode"], "frame_count": len(frames),
        "resolution": f"{run['image_width']}x{run['image_height']}",
        "starting_altitude_m": scenario.get("starting_altitude_m", ""),
        "device": run.get("device_label", run.get("device", "")),
        "total_analysis_time_s": summary.get("total_analysis_time_s", run.get("analysis_time_s", "")),
        "average_analysis_ms_per_frame": summary.get("average_analysis_time_ms", ""),
        "final_state": frames[-1].get("state", ""),
        "final_selected_lz": summary.get("final_landing_zone", ""),
        "final_score": summary.get("final_safety_score", ""),
        "retarget_count": summary.get("retarget_events", ""),
        "target_lock_timestamp_s": summary.get("target_locked_at_s", ""),
        "max_visual_drift_px_s": round(max(drifts), 2) if drifts else "",
        "average_visual_drift_px_s": round(sum(drifts) / len(drifts), 2) if drifts else "",
        "detected_object_count_total": sum(len(frame.get("detections", [])) for frame in frames),
        "run_folder": str(run_folder), "landing_video": str(landing),
        "dashboard_video": str(dashboard), "success": True, "error": "",
    }
    videos = {
        "landing_replay": verify_video(landing, run["image_width"], run["image_height"]),
        "dashboard_replay": verify_video(dashboard, run["image_width"], run["image_height"]),
    }
    return row, videos


def failure_row(scenario: dict, error: Exception) -> dict:
    row = {column: "" for column in COLUMNS}
    row.update({
        "scenario_name": scenario.get("scenario_name", ""), "source_seed": scenario.get("source_seed", ""),
        "mission_mode": scenario.get("mission_mode", ""), "frame_count": scenario.get("frame_count", ""),
        "starting_altitude_m": scenario.get("starting_altitude_m", ""), "success": False,
        "error": f"{type(error).__name__}: {error}",
    })
    return row


def write_reports(rows: list[dict], videos: dict, started: float, final: bool) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    elapsed = time.perf_counter() - started
    successful = sum(bool(row["success"]) for row in rows)
    metadata = {
        "evaluation_set": BATCH_ID, "generated_at": datetime.now().isoformat(timespec="seconds"),
        "complete": final, "total_scenarios": len(rows), "successful_scenarios": successful,
        "failed_scenarios": len(rows) - successful,
        "total_processed_frames": sum(int(row["frame_count"] or 0) for row in rows if row["success"]),
        "compute_device": next((row["device"] for row in rows if row["device"]), "Apple MPS"),
        "total_batch_processing_time_s": round(elapsed, 2), "scenarios": rows,
        "video_validation": videos,
    }
    with (OUTPUT_DIR / "batch_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in COLUMNS} for row in rows)
    (OUTPUT_DIR / "batch_summary.json").write_text(json.dumps(metadata, indent=2) + "\n")

    table = [
        "| Scenario | Environment | Frames | Analysis Time | ms/frame | Retargets | Final Score | Target Lock | Result |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        score = f"{float(row['final_score']):.1%}" if row["final_score"] != "" else "—"
        lock = f"{float(row['target_lock_timestamp_s']):.3f}s" if row["target_lock_timestamp_s"] != "" else "—"
        result = row["final_state"] if row["success"] else "FAILED"
        table.append(
            f"| {row['scenario_name']} | {environment_label(row['scenario_name'])} | {row['frame_count'] or '—'} | "
            f"{row['total_analysis_time_s'] or '—'}s | {row['average_analysis_ms_per_frame'] or '—'} | "
            f"{row['retarget_count'] or 0} | {score} | {lock} | {result} |"
        )
    sections = []
    for row in rows:
        section_score = f"{float(row['final_score']):.1%}" if row["final_score"] != "" else "Unavailable"
        sections.append(
            f"## {row['scenario_name']}\n\n"
            f"Source: `{row['source_seed']}`  \nMission mode: {row['mission_mode']}  \n"
            f"Result: {row['final_state'] if row['success'] else 'FAILED'}  \n"
            f"Final selected zone: {row['final_selected_lz'] or 'Unavailable'}  \n"
            f"Final score: {section_score}  \n"
            f"Retargets: {row['retarget_count'] if row['retarget_count'] != '' else 'Unavailable'}  \n"
            f"Analysis time: {str(row['total_analysis_time_s']) + ' s' if row['total_analysis_time_s'] != '' else 'Unavailable'}  \n"
            f"Replay: `{row['landing_video'] or 'Unavailable'}`  \n"
            f"Dashboard: `{row['dashboard_video'] or 'Unavailable'}`  \n"
            f"Error: {row['error'] or 'None'}\n"
        )
    report = (
        "# OPENLANDER REAL-SEED EVALUATION\n\n"
        "> This report describes system behavior, inference performance, decision stability, retarget behavior, runtime, and perception outputs. The seed set has no human-validated landing ground truth, so it does not measure landing accuracy.\n\n"
        f"- Total scenarios: {len(rows)}\n- Successful scenarios: {successful}\n"
        f"- Failed scenarios: {len(rows) - successful}\n"
        f"- Total processed frames: {metadata['total_processed_frames']}\n"
        f"- Compute device: {metadata['compute_device']}\n"
        f"- Total batch processing time: {elapsed / 60:.2f} min\n\n"
        + "\n".join(table) + "\n\n" + "\n".join(sections)
    )
    (OUTPUT_DIR / "batch_report.md").write_text(report)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    scenarios = discover_scenarios(args.scenario)
    integrity = json.loads(INTEGRITY_PATH.read_text()) if INTEGRITY_PATH.is_file() else {}
    hashes_before = integrity.get("before", current_seed_hashes())
    current = current_seed_hashes()
    if current != hashes_before:
        raise RuntimeError("Seed integrity differs from the pre-generation SHA256 manifest.")
    print(f"Seeds discovered: {len(current)}", flush=True)
    print(f"Scenarios queued: {len(scenarios)}", flush=True)

    prior_rows: list[dict] = []
    prior_videos: dict = {}
    summary_path = OUTPUT_DIR / "batch_summary.json"
    if summary_path.is_file():
        prior = json.loads(summary_path.read_text())
        if prior.get("evaluation_set") == BATCH_ID:
            prior_rows = prior.get("scenarios", [])
            prior_videos = prior.get("video_validation", {})
    completed = {row["scenario_name"] for row in prior_rows if row.get("success")}
    rows = list(prior_rows)
    videos = dict(prior_videos)

    device, label = engine.compute_device()
    if device != "mps":
        raise RuntimeError(f"This evaluation requires Apple MPS; detected {label}.")
    print(f"Device: {label}", flush=True)
    queued_names = {json.loads((folder / "scenario.json").read_text())["scenario_name"] for folder in scenarios}
    needs_analysis = args.force or bool(queued_names - completed)
    models = engine.load_models(device, lambda message: print(message, flush=True)) if needs_analysis else None
    try:
        for index, folder in enumerate(scenarios, 1):
            scenario = json.loads((folder / "scenario.json").read_text())
            name = scenario["scenario_name"]
            if name in completed and not args.force:
                print(f"[{index}/{len(scenarios)}] {name}: already complete, skipping", flush=True)
                continue
            print(f"[{index}/{len(scenarios)}] {name}: analysis started", flush=True)
            try:
                def scenario_progress(message: str, scenario_name: str = name) -> None:
                    if not message.endswith("— saved"):
                        return
                    frame_number = int(message.split()[1])
                    if frame_number == 1 or frame_number % 25 == 0 or frame_number == 150:
                        print(f"  {scenario_name}: {message}", flush=True)

                run_path = Path(engine.analyze_scenario(
                    folder, scenario_progress,
                    models=models, device=device, evaluation_set=BATCH_ID,
                ))
                landing = Path(engine.export_video(run_path, "Final Decision", 30))
                dashboard = Path(engine.export_video(run_path, "Debug Dashboard", 30))
                row, video_info = success_row(scenario, run_path, landing, dashboard)
                videos[name] = video_info
                print(f"[{index}/{len(scenarios)}] {name}: complete", flush=True)
            except Exception as error:
                row = failure_row(scenario, error)
                print(f"[{index}/{len(scenarios)}] {name}: FAILED — {row['error']}", flush=True)
            rows = [existing for existing in rows if existing["scenario_name"] != name]
            rows.append(row)
            rows.sort(key=lambda item: item["scenario_name"])
            write_reports(rows, videos, started, final=False)
            gc.collect()
            models["torch"].mps.empty_cache()
    finally:
        if models is not None: del models
        gc.collect()

    after = current_seed_hashes()
    integrity.update({
        "verified_after_batch_at": datetime.now().isoformat(timespec="seconds"),
        "seed_count": len(after), "before": hashes_before,
        "after_generation": integrity.get("after", current), "after_batch": after,
        "unchanged": hashes_before == after,
    })
    INTEGRITY_PATH.write_text(json.dumps(integrity, indent=2) + "\n")
    if not integrity["unchanged"]:
        raise RuntimeError("Seed integrity check failed after batch processing.")
    write_reports(rows, videos, started, final=True)
    successful = [row for row in rows if row["success"]]
    failed = [row for row in rows if not row["success"]]
    print("\nOPENLANDER REAL-SEED EVALUATION COMPLETE", flush=True)
    print(f"Seeds discovered: {len(after)}", flush=True)
    print(f"Scenarios generated: {len(scenarios)}", flush=True)
    print(f"Scenarios analyzed: {len(rows)}", flush=True)
    print(f"Successful: {len(successful)}", flush=True)
    print(f"Failed: {len(failed)}", flush=True)
    print(f"Frames analyzed: {sum(int(row['frame_count']) for row in successful)}", flush=True)
    print(f"Device: {label}", flush=True)
    print(f"Total batch processing time: {(time.perf_counter() - started) / 60:.2f} min", flush=True)
    print(f"Batch report: {OUTPUT_DIR / 'batch_report.md'}", flush=True)
    print(f"Batch CSV: {OUTPUT_DIR / 'batch_summary.csv'}", flush=True)
    for row in successful:
        print(f"Run: {row['run_folder']}", flush=True)
        print(f"  Landing: {row['landing_video']}", flush=True)
        print(f"  Dashboard: {row['dashboard_video']}", flush=True)


if __name__ == "__main__":
    main()
