import os
from typing import Dict, List, Tuple

import pandas as pd

from eyes_detection import analyze_tadpole_microscope
from logs import log_error, log_parameters, log_status, setup_logger, timestamp_now
from utils import ensure_directory, list_images, write_machine_friendly_csv, write_metadata


PipelineResult = Tuple[pd.DataFrame, Dict]


def _extract_condition_replicate(image_path: str) -> Tuple[str, str]:
    parts = os.path.normpath(image_path).split(os.sep)
    try:
        return parts[-3], parts[-2]
    except Exception:
        return "unknown", "unknown"


def _build_row(image_path: str, body_px: float, eye_px: float, status: str, pixel_mm_ratio: float, tail_factor: float, orientation: str) -> Dict:
    condition, replicate = _extract_condition_replicate(image_path)
    body_mm = body_px * pixel_mm_ratio
    total_mm = body_mm * tail_factor
    eye_mm = eye_px * pixel_mm_ratio
    ratio = (eye_mm / total_mm) if total_mm else 0.0

    return {
        "frog_id": os.path.splitext(os.path.basename(image_path))[0],
        "condition": condition,
        "replicate": replicate,
        "body_mm": round(body_mm, 3),
        "total_mm": round(total_mm, 3),
        "eye_distance_mm": round(eye_mm, 3),
        "ratio": round(ratio, 4),
        "status": status,
        "orientation": orientation,
        "full_path": image_path,
    }


def run_tadpole_batch(
    images_dir: str,
    pixel_mm_ratio: float = 0.0053,
    tail_factor: float = 2.6,
    output_dir: str = None,
    debug: bool = False,
) -> PipelineResult:
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Input directory not found: {images_dir}")

    target_dir = output_dir or os.path.join(images_dir, "results")
    ensure_directory(target_dir)
    logger = setup_logger(target_dir)
    log_parameters(logger, pixel_mm_ratio, tail_factor)

    image_paths = list_images(images_dir)
    results: List[Dict] = []
    error_count = 0

    for idx, image_path in enumerate(image_paths, start=1):
        log_status(logger, f"Processing {idx}/{len(image_paths)} - {image_path}")
        try:
            _, len_px_corps, eyes_px, snout_px, status, orientation = analyze_tadpole_microscope(image_path, debug=debug)
            if not status.lower().startswith("succès"):
                error_count += 1
            results.append(_build_row(image_path, body_px, eyes_px, snout_px, status, pixel_mm_ratio, tail_factor, orientation))
        except Exception as exc:  
            error_count += 1
            log_error(logger, f"Crash on {image_path}: {exc}")
            results.append(
                _build_row(
                    image_path,
                    body_px=0.0,
                    eye_px=0.0,
                    status=f"Erreur: {exc}",
                    pixel_mm_ratio=pixel_mm_ratio,
                    tail_factor=tail_factor,
                    orientation="error"
                )
            )

    df = pd.DataFrame(results, columns=[
        "frog_id",
        "condition",
        "replicate",
        "body_mm",
        "total_mm",
        "eye_distance_mm",
        "ratio",
        "status",
        "orientation",
        "full_path",
    ])

    metadata = {
        "pixel_mm_ratio": pixel_mm_ratio,
        "tail_factor": tail_factor,
        "n_images": len(image_paths),
        "n_errors": error_count,
        "date": timestamp_now(),
    }

    write_machine_friendly_csv(df, os.path.join(target_dir, "resultats_codex.csv"))
    write_metadata(metadata, os.path.join(target_dir, "metadata.json"))

    return df, metadata
