"""
Phase 2: Combine all per-race parquet files into three unified
datasets — one for laps, one for results, one for weather.

Run locally: python src/combine_data.py
"""

import glob
import pandas as pd
import os


def combine_folder(raw_dir, output_path):
    """Read every parquet file in raw_dir, concatenate into one DataFrame,
    save as a single combined parquet file."""
    files = sorted(glob.glob(f"{raw_dir}/*.parquet"))
    print(f"Combining {len(files)} files from {raw_dir}...")

    if not files:
        print(f"  WARNING: no files found in {raw_dir} — did the pull actually run?")
        return None

    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            dfs.append(df)
        except Exception as e:
            print(f"  Skipping {f} — failed to read: {e}")

    combined = pd.concat(dfs, ignore_index=True)
    combined.to_parquet(output_path)
    print(f"  Saved combined file: {output_path} ({len(combined)} rows)")
    return combined


if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)

    print("=== Combining laps ===")
    laps = combine_folder("data/raw/laps", "data/processed/all_laps.parquet")

    print("\n=== Combining results ===")
    results = combine_folder("data/raw/results", "data/processed/all_results.parquet")

    print("\n=== Combining weather ===")
    weather = combine_folder("data/raw/weather", "data/processed/all_weather.parquet")

    print("\n=== Combining qualifying ===")
    qualifying = combine_folder("data/raw/qualifying", "data/processed/all_qualifying.parquet")

    print("\n=== Summary ===")
    if laps is not None:
        print(f"Laps: {len(laps)} rows, {laps['season'].nunique()} seasons, "
              f"{laps.groupby('season')['round'].nunique().sum()} races")
    if results is not None:
        print(f"Results: {len(results)} rows")
    if weather is not None:
        print(f"Weather: {len(weather)} rows")
    if qualifying is not None:
        print(f"Qualifying: {len(qualifying)} rows")

    print("\nDone. Combined files saved in data/processed/.")