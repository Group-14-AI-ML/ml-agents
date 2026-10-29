import csv
import json
import logging
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
CONFIG_PATH = SCRIPT_DIR / "runner_config.json"
BASE_YAML = REPO_ROOT / "config/ppo/PushBlock.yaml"
TEMP_YAML = REPO_ROOT / "config/ppo/PushBlock_temp.yaml"
RESULTS_DIR = REPO_ROOT / "results"
ENV_PATH = REPO_ROOT / "Project/PushBlock.x86_64"
OUTPUT_DIR = REPO_ROOT / "personal-src"
DEFAULT_MASTER_CSV = OUTPUT_DIR / "local_results_master.csv"
MASTER_CSV = DEFAULT_MASTER_CSV
LOG_FILE = SCRIPT_DIR / "nightly_runner.log"

TELEMETRY_COLUMNS = [
    "run_id",
    "environment_name",
    "algorithm",
    "learning_rate",
    "batch_size",
    "buffer_size",
    "max_steps",
    "random_seed",
    "cpu_model",
    "ram_gb",
    "has_gpu",
    "eval_success_rate",
    "elapsed_seconds",
    "steps_to_converge",
    "policy_stability_std",
]

sys.path.insert(0, str(SCRIPT_DIR))
from extract_telemetry import extract_run_data

logger = logging.getLogger("nightly_runner")


def setup_logging():
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return
    formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


def load_runner_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Runner config not found: {CONFIG_PATH}")
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Runner config must be a JSON object: {CONFIG_PATH}")
    for key in ("user_initials", "config_csv"):
        value = config.get(key)
        if value is None or not str(value).strip():
            raise ValueError(f"Missing or empty '{key}' in {CONFIG_PATH}")
    return config


def resolve_path(value):
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def resolve_master_csv(config):
    master_name = Path(str(config.get("master_csv") or DEFAULT_MASTER_CSV.name)).name
    return OUTPUT_DIR / master_name


def load_config_rows(csv_path):
    if not csv_path.exists():
        raise FileNotFoundError(f"Configuration CSV not found: {csv_path}")
    dataframe = pd.read_csv(csv_path)
    required = ("environment_name", "learning_rate", "batch_size", "buffer_size", "max_steps")
    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {', '.join(missing)}")
    environment = dataframe["environment_name"]
    normalized = environment.astype(str).str.strip()
    active = environment.notna() & ~normalized.str.lower().isin({"", "nan", "none"})
    return dataframe.loc[active].reset_index(drop=True)


def resolve_mlagents_command():
    command = shutil.which("mlagents-learn")
    if command:
        return command
    candidate = Path(sys.executable).parent / "mlagents-learn"
    if candidate.exists():
        return str(candidate)
    raise FileNotFoundError("mlagents-learn not found in PATH or the active Python environment")


def generate_run_id(initials, used_run_ids):
    prefix = re.sub(r"[^A-Za-z0-9_-]+", "_", initials.strip()).strip("_-") or "RUN"
    while True:
        run_id = f"{prefix}-{datetime.now():%Y%m%d_%H%M%S}"
        if run_id not in used_run_ids and not (RESULTS_DIR / run_id).exists():
            return run_id
        time.sleep(1)


def as_float(value, name):
    if pd.isna(value):
        raise ValueError(f"Missing value for '{name}'")
    return float(value)


def as_int(value, name):
    if pd.isna(value):
        raise ValueError(f"Missing value for '{name}'")
    return int(float(value))


def write_temp_yaml(row):
    behavior_name = str(row["environment_name"]).strip()
    with BASE_YAML.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    behaviors = config.get("behaviors")
    if not isinstance(behaviors, dict) or behavior_name not in behaviors:
        raise ValueError(f"Behavior '{behavior_name}' not found in {BASE_YAML}")

    behavior = behaviors[behavior_name]
    hyperparameters = behavior.setdefault("hyperparameters", {})
    if not isinstance(hyperparameters, dict):
        raise ValueError(f"Invalid hyperparameters for behavior '{behavior_name}'")

    hyperparameters["learning_rate"] = as_float(row["learning_rate"], "learning_rate")
    hyperparameters["batch_size"] = as_int(row["batch_size"], "batch_size")
    hyperparameters["buffer_size"] = as_int(row["buffer_size"], "buffer_size")
    behavior["max_steps"] = as_int(row["max_steps"], "max_steps")

    if "summary_freq" in row.index and not pd.isna(row["summary_freq"]):
        behavior["summary_freq"] = as_int(row["summary_freq"], "summary_freq")

    if "random_seed" in row.index and not pd.isna(row["random_seed"]):
        env_settings = config.setdefault("env_settings", {})
        if not isinstance(env_settings, dict):
            raise ValueError("Invalid env_settings in base YAML")
        env_settings["seed"] = as_int(row["random_seed"], "random_seed")

    TEMP_YAML.parent.mkdir(parents=True, exist_ok=True)
    with TEMP_YAML.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    return behavior_name


def run_training(run_id, mlagents_command):
    command = [
        mlagents_command,
        str(TEMP_YAML),
        f"--env={ENV_PATH}",
        f"--run-id={run_id}",
        "--force",
        "--no-graphics",
    ]
    logger.info("Starting training: %s", " ".join(command))
    started = time.monotonic()
    result = subprocess.run(command, cwd=str(REPO_ROOT))
    elapsed = time.monotonic() - started
    if result.returncode != 0:
        raise RuntimeError(f"Training failed with exit code {result.returncode}")
    logger.info("Training finished in %.1f seconds", elapsed)


def extract_telemetry(run_id, behavior_name):
    row = extract_run_data(
        run_id,
        results_base_dir=str(RESULTS_DIR),
        behavior_name=behavior_name,
    )
    if row is None:
        raise RuntimeError(f"No telemetry extracted for run '{run_id}'")
    missing = [column for column in TELEMETRY_COLUMNS if column not in row]
    if missing:
        raise RuntimeError(f"Telemetry row missing columns: {', '.join(missing)}")
    return row


def append_to_master(row, master_csv=MASTER_CSV):
    master_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not master_csv.exists() or master_csv.stat().st_size == 0
    with master_csv.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TELEMETRY_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({column: row[column] for column in TELEMETRY_COLUMNS})


def cleanup_run(run_id):
    run_dir = RESULTS_DIR / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)
        logger.warning("Removed failed run directory: %s", run_dir)


def main():
    setup_logging()

    try:
        config = load_runner_config()
        csv_path = resolve_path(config["config_csv"])
        master_csv = resolve_master_csv(config)
        mlagents_command = resolve_mlagents_command()
        if not ENV_PATH.exists():
            raise FileNotFoundError(f"Unity environment not found: {ENV_PATH}")
        rows = load_config_rows(csv_path)
    except Exception:
        logger.exception("Startup failed")
        return 1

    if rows.empty:
        logger.warning("No active training rows found in %s", csv_path)
        return 0

    logger.info("Found %d active training run(s) in %s", len(rows), csv_path)
    used_run_ids = set()
    succeeded = 0
    failed = 0

    for index, row in rows.iterrows():
        run_id = generate_run_id(str(config["user_initials"]), used_run_ids)
        used_run_ids.add(run_id)
        logger.info("[%d/%d] Generated run ID: %s", index + 1, len(rows), run_id)

        try:
            behavior_name = write_temp_yaml(row)
            run_training(run_id, mlagents_command)
            telemetry = extract_telemetry(run_id, behavior_name)
            append_to_master(telemetry, master_csv=master_csv)
            succeeded += 1
            logger.info("Run %s appended to %s", run_id, master_csv)
        except KeyboardInterrupt:
            logger.warning("Interrupted while processing run %s", run_id)
            raise
        except Exception:
            failed += 1
            logger.exception("Run %s failed; no data appended to master CSV", run_id)
            cleanup_run(run_id)
        finally:
            TEMP_YAML.unlink(missing_ok=True)

    logger.info("Finished: %d succeeded, %d failed", succeeded, failed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
