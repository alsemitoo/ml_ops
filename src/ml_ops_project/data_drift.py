"""Data drift detection utilities using Evidently.

Run locally to compare reference vs current image distributions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import hydra
import numpy as np
import pandas as pd
from evidently.legacy.metric_preset import DataDriftPreset  # type: ignore
from evidently.legacy.report import Report  # type: ignore
from omegaconf import DictConfig
from PIL import Image

CONFIG_DIR = Path(__file__).parent.parent.parent / "configs"


def iter_image_paths(root: Path, patterns: Iterable[str], start_idx: int = 0, end_idx: int | None = None) -> list[Path]:
    """Collect image file paths under root matching patterns within index range.

    Args:
        root: Root directory to search
        patterns: File patterns to match (e.g., ["*.png"])
        start_idx: Starting index (inclusive)
        end_idx: Ending index (exclusive), None for no limit
    """

    all_paths: list[Path] = []
    for pattern in patterns:
        all_paths.extend(sorted(root.rglob(pattern)))

    all_paths = sorted(all_paths)

    if end_idx is None:
        return all_paths[start_idx:]
    return all_paths[start_idx:end_idx]


def image_features(path: Path) -> dict[str, float]:
    """Compute simple image features for drift detection."""

    with Image.open(path) as img:
        gray = img.convert("L")
        arr = np.asarray(gray, dtype=np.float32)
        brightness = float(arr.mean())
        contrast = float(arr.std())
        gy, gx = np.gradient(arr)
        sharpness = float(np.abs(gx).mean() + np.abs(gy).mean())
        width, height = gray.size
        aspect_ratio = float(width / height) if height else 0.0
    return {
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": sharpness,
        "width": float(width),
        "height": float(height),
        "aspect_ratio": aspect_ratio,
    }


def build_dataframe(paths: list[Path]) -> pd.DataFrame:
    """Turn image files into a feature DataFrame."""

    records = [image_features(p) for p in paths]
    return pd.DataFrame.from_records(records)


def run_drift_report(reference: pd.DataFrame, current: pd.DataFrame, html_path: Path, json_path: Path) -> None:
    """Run Evidently drift report and save outputs."""

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(html_path))
    report.save_json(str(json_path))


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="drift")
def main(cfg: DictConfig) -> None:
    """Entry point for local drift detection."""

    ref_dir = Path(cfg.paths.reference_dir)
    cur_dir = Path(cfg.paths.current_dir)
    image_globs = cfg.paths.image_globs

    ref_start = cfg.paths.get("reference_start_idx", 0)
    ref_end = cfg.paths.get("reference_end_idx", None)
    cur_start = cfg.paths.get("current_start_idx", 0)
    cur_end = cfg.paths.get("current_end_idx", None)

    ref_paths = iter_image_paths(ref_dir, image_globs, ref_start, ref_end)
    cur_paths = iter_image_paths(cur_dir, image_globs, cur_start, cur_end)

    if not ref_paths:
        msg = f"No reference images found in {ref_dir} (indices {ref_start}:{ref_end})"
        raise FileNotFoundError(msg)
    if not cur_paths:
        msg = f"No current images found in {cur_dir} (indices {cur_start}:{cur_end})"
        raise FileNotFoundError(msg)

    reference_df = build_dataframe(ref_paths)
    current_df = build_dataframe(cur_paths)

    html_path = Path(cfg.output.html_report)
    json_path = Path(cfg.output.json_report)

    run_drift_report(reference_df, current_df, html_path, json_path)


if __name__ == "__main__":
    main()
