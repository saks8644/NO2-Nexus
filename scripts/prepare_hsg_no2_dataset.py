from __future__ import annotations

import argparse
import tarfile
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd


DATASET_URL = "https://github.com/HSG-AIML/NO2-dataset/raw/main/no2_dataset.tar.gz"
DEFAULT_ARCHIVE = Path("data/raw/hsg_no2/no2_dataset.tar.gz")
DEFAULT_EXTRACT_DIR = Path("data/raw/hsg_no2/extracted")
DEFAULT_OUTPUT = Path("data/processed/hsg_no2_training.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and prepare the public HSG-AIML Sentinel-5P/ground NO2 dataset."
    )
    parser.add_argument("--url", default=DATASET_URL, help="Dataset archive URL.")
    parser.add_argument("--archive", default=DEFAULT_ARCHIVE, type=Path, help="Local archive path.")
    parser.add_argument("--extract-dir", default=DEFAULT_EXTRACT_DIR, type=Path, help="Extraction directory.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, type=Path, help="Processed CSV output path.")
    return parser.parse_args()


def normalise_column_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def download_archive(url: str, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        print(f"Using existing archive: {archive_path}")
        return
    print(f"Downloading {url}")
    urlretrieve(url, archive_path)


def extract_archive(archive_path: Path, extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    if any(extract_dir.iterdir()):
        print(f"Using existing extracted directory: {extract_dir}")
        return
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(extract_dir)


def find_tabular_files(extract_dir: Path) -> list[Path]:
    return sorted(
        [
            *extract_dir.rglob("*.csv"),
            *extract_dir.rglob("*.tsv"),
            *extract_dir.rglob("*.txt"),
        ]
    )


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def choose_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    for column in columns:
        if any(candidate in column for candidate in candidates):
            return column
    return None


def standardise_training_frame(frame: pd.DataFrame) -> pd.DataFrame | None:
    data = frame.copy()
    data.columns = [normalise_column_name(column) for column in data.columns]
    columns = list(data.columns)

    latitude = choose_column(columns, ("latitude", "lat"))
    longitude = choose_column(columns, ("longitude", "lon", "lng"))
    target = choose_column(columns, ("target_no2", "ground_truth_no2", "observed_no2", "station_no2", "no2"))
    coarse = choose_column(
        columns,
        (
            "coarse_no2",
            "tropospheric_no2_column_number_density",
            "no2_column_number_density",
            "s5p_no2",
            "tropomi_no2",
        ),
    )

    required = {"latitude": latitude, "longitude": longitude, "target_no2": target}
    if any(value is None for value in required.values()):
        return None

    rename_map = {
        latitude: "latitude",
        longitude: "longitude",
        target: "target_no2",
    }
    if coarse is not None:
        rename_map[coarse] = "coarse_no2"

    data = data.rename(columns=rename_map)
    numeric_columns = [
        column
        for column in data.columns
        if column in {"latitude", "longitude", "target_no2"} or pd.api.types.is_numeric_dtype(data[column])
    ]
    selected = data[numeric_columns].replace([float("inf"), float("-inf")], pd.NA).dropna()
    if selected.empty:
        return None
    return selected


def prepare_dataset(extract_dir: Path, output_path: Path) -> Path:
    candidates = find_tabular_files(extract_dir)
    if not candidates:
        raise FileNotFoundError(f"No CSV/TSV/TXT files found under {extract_dir}")

    errors: list[str] = []
    for candidate in candidates:
        try:
            frame = read_table(candidate)
            prepared = standardise_training_frame(frame)
        except Exception as exc:  # pragma: no cover - diagnostic path for external data
            errors.append(f"{candidate}: {exc}")
            continue
        if prepared is None:
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)
        prepared.to_csv(output_path, index=False)
        print(f"Prepared {len(prepared):,} rows from {candidate}")
        print(f"Wrote {output_path}")
        return output_path

    raise ValueError(
        "Could not infer latitude, longitude, and target NO2 columns from downloaded files. "
        "Inspect the extracted files and pass a cleaned CSV to no2-nexus manually.\n"
        + "\n".join(errors[:5])
    )


def main() -> None:
    args = parse_args()
    download_archive(args.url, args.archive)
    extract_archive(args.archive, args.extract_dir)
    prepare_dataset(args.extract_dir, args.output)


if __name__ == "__main__":
    main()
