import json
import os
from typing import Dict, Iterable, List

import pandas as pd


SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def list_images(root_dir: str) -> List[str]:
    images: List[str] = []
    for base, _, files in os.walk(root_dir):
        for fname in files:
            if fname.lower().endswith(SUPPORTED_EXTENSIONS):
                images.append(os.path.join(base, fname))
    return sorted(images)


def ensure_directory(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def write_machine_friendly_csv(df: pd.DataFrame, path: str) -> None:
    """Write a predictable CSV even with the lightweight pandas stub."""
    if hasattr(df, "to_csv"):
        df.to_csv(path, index=False)
    else:
        headers = list(df.columns.keys())
        with open(path, "w") as f:
            f.write(",".join(headers) + "\n")
            for i in range(len(df)):
                row = [str(df.columns[h][i]) if i < len(df.columns[h]) else "" for h in headers]
                f.write(",".join(row) + "\n")


def write_metadata(metadata: Dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)

