import os
from typing import Dict, List, Tuple

import pandas as pd

from .egg_counting import analyze_eggs
from .logs import log_error, log_status, setup_logger, timestamp_now
from .utils import ensure_directory, list_images, write_machine_friendly_csv, write_metadata


EggPipelineResult = Tuple[pd.DataFrame, Dict]


def _extract_condition_replicate(image_path: str) -> Tuple[str, str]:
    parts = os.path.normpath(image_path).split(os.sep)
    try:
        return parts[-3], parts[-2]
    except Exception:
        return "unknown", "unknown"


def _build_row(image_path: str, fertilized: int, unfertilized: int, status: str) -> Dict:
    condition, replicate = _extract_condition_replicate(image_path)
    return {
        "egg_id": os.path.splitext(os.path.basename(image_path))[0],
        "condition": condition,
        "replicate": replicate,
        "fertilized": fertilized,
        "unfertilized": unfertilized,
        "status": status,
        "full_path": image_path,
    }


def run_egg_batch(images_dir: str, output_dir: str = None, debug: bool = False) -> EggPipelineResult:
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Input directory not found: {images_dir}")

    target_dir = output_dir or os.path.join(images_dir, "results")
    ensure_directory(target_dir)
    logger = setup_logger(target_dir, name="egg_pipeline")

    image_paths = list_images(images_dir)
    results: List[Dict] = []
    error_count = 0

    for idx, image_path in enumerate(image_paths, start=1):
        log_status(logger, f"Processing {idx}/{len(image_paths)} - {image_path}")
        try:
            _, fertilized, unfertilized, status = analyze_eggs(image_path, debug=debug)
            if "Aucun" in status:
                error_count += 1
            results.append(_build_row(image_path, fertilized, unfertilized, status))
        except Exception as exc:  # pragma: no cover
            error_count += 1
            log_error(logger, f"Crash on {image_path}: {exc}")
            results.append(_build_row(image_path, 0, 0, f"Erreur: {exc}"))

    df = pd.DataFrame(results, columns=[
        "egg_id",
        "condition",
        "replicate",
        "fertilized",
        "unfertilized",
        "status",
        "full_path",
    ])

    metadata = {
        "n_images": len(image_paths),
        "n_errors": error_count,
        "date": timestamp_now(),
    }

    write_machine_friendly_csv(df, os.path.join(target_dir, "resultats_codex_oeufs.csv"))
    write_metadata(metadata, os.path.join(target_dir, "metadata_oeufs.json"))

    return df, metadata

