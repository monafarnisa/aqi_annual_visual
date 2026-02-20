"""
04_animate_inset_loop.py
Create a looping animation from yearly raster inset PNGs (2020-2025).
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

YEARS = list(range(2020, 2026))
INPUT_DIR = Path(__file__).resolve().parent / "outputs" / "idw_raster_inset"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "animation"
OUTPUT_GIF = OUTPUT_DIR / "aqi_days_above_100_inset_2020_2025_loop.gif"

# Animation pacing: 900ms per frame, loop forever.
FRAME_DURATION_MS = 900
LOOP_FOREVER = 0


def frame_paths(years: list[int]) -> list[Path]:
    paths = [INPUT_DIR / f"us_days_above_100_inset_{year}.png" for year in years]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing inset frames. Re-run script 2 to generate:\n"
            + "\n".join(str(p) for p in missing)
        )
    return paths


def frame_paths_with_suffix(years: list[int], suffix: str) -> list[Path]:
    suffix = suffix.strip("_")
    paths = [INPUT_DIR / f"us_days_above_100_inset_{year}_{suffix}.png" for year in years]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing inset frames for suffix. Re-run generation:\n"
            + "\n".join(str(p) for p in missing)
        )
    return paths


def build_loop_gif(
    years: list[int] = YEARS,
    frame_duration_ms: int = FRAME_DURATION_MS,
    output_path: Path = OUTPUT_GIF,
) -> Path:
    paths = frame_paths(years)
    frames = [Image.open(p).convert("RGB") for p in paths]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=LOOP_FOREVER,
        optimize=True,
    )
    print(f"Saved {output_path}")
    return output_path


def main() -> None:
    build_loop_gif()


if __name__ == "__main__":
    main()
