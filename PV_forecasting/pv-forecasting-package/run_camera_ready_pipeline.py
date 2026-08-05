"""Regenerate all camera-ready PV-forecasting results with one command.

The command runs the data audit, both main evaluation protocols, multi-seed
stability runs, both-protocol ablations, traceable result summarisation, and
all paper/appendix figures.  Every subprocess is logged and a checksum-backed
``run_manifest.json`` is updated after every step.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

PACKAGE_ROOT = Path(__file__).resolve().parent


def _find_repo_root(start: Path = PACKAGE_ROOT) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError(f"Could not find a Git repository above {start}")


REPO_ROOT = _find_repo_root()
MANIFEST_NAME = "run_manifest.json"
SOFTWARE_PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "matplotlib",
    "torch",
    "pvlib",
    "requests",
)

MAIN_OUTPUTS = (
    "pv_v4_run_config.json",
    "pv_v4_metrics.csv",
    "pv_v4_per_fold_metrics.csv",
    "pv_v4_predictions.csv",
    "pv_v4_per_month.csv",
    "pv_v4_residual_corr_temporal.csv",
    "pv_v4_residual_corr_kfold.csv",
    "pv_v4_significance.csv",
    "pv_v4_summary.png",
)
SUMMARY_OUTPUTS = (
    "camera_ready_headline_metrics.csv",
    "camera_ready_claims.json",
    "camera_ready_results_table.tex",
)
PAPER_FIGURES = tuple(
    f"{stem}.{extension}"
    for stem in (
        "pv_v4_fig_model_comparison",
        "pv_v4_fig_residual_corr",
        "pv_v4_fig_representative_week",
    )
    for extension in ("pdf", "png")
)
RESULTS_FIGURES = tuple(
    f"pv_v4_fig_results.{extension}" for extension in ("pdf", "png")
)
APPENDIX_FIGURES = tuple(
    f"{stem}.{extension}"
    for stem in (
        "pv_v4_fig_app_monthly_skill",
        "pv_v4_fig_app_missingness",
        "pv_v4_fig_app_error_by_hour",
        "pv_v4_fig_app_error_by_clearness",
        "pv_v4_fig_app_skill_baselines",
        "pv_v4_fig_app_ablation",
        "pv_v4_fig_app_alignment",
    )
    for extension in ("pdf", "png")
)


@dataclass(frozen=True)
class Step:
    name: str
    command: tuple[str, ...]
    expected_outputs: tuple[str, ...]
    environment: tuple[tuple[str, str], ...] = ()


def build_steps(python: str, seeds: Iterable[int]) -> tuple[Step, ...]:
    """Return the complete, ordered and testable camera-ready execution plan."""
    seed_args = tuple(str(seed) for seed in seeds)
    return (
        Step(
            "data_audit",
            (python, "audit data/audit_data.py"),
            ("data_audit_missing_runs.csv", "data_audit_summary.json"),
        ),
        Step(
            "alignment_scan",
            (python, "tests/scan_time_shift.py"),
            ("pv_time_shift_scan.csv",),
        ),
        Step("main_experiment", (python, "analysis/analyze_pv_v4.py"), MAIN_OUTPUTS),
        Step(
            "multiseed_kfold",
            (
                python,
                "tests/run_multiseed.py",
                "--mode",
                "KFOLD",
                "--seeds",
                *seed_args,
            ),
            (
                "pv_v4_multiseed_raw__multiseed_kfold.csv",
                "pv_v4_multiseed_summary__multiseed_kfold.csv",
            ),
            (("PV_RUN_TAG", "multiseed_kfold"),),
        ),
        Step(
            "multiseed_temporal",
            (
                python,
                "tests/run_multiseed.py",
                "--mode",
                "TEMPORAL",
                "--seeds",
                *seed_args,
            ),
            (
                "pv_v4_multiseed_raw__multiseed_temporal.csv",
                "pv_v4_multiseed_summary__multiseed_temporal.csv",
            ),
            (("PV_RUN_TAG", "multiseed_temporal"),),
        ),
        Step(
            "ablations",
            (python, "tests/abliations/run_ablations.py", "--both"),
            ("pv_v4_ablation.csv",),
        ),
        Step(
            "result_summary",
            (python, "summarise_camera_ready_results.py"),
            SUMMARY_OUTPUTS,
        ),
        Step(
            "paper_figures",
            (python, "figures/make_paper_figures.py"),
            PAPER_FIGURES,
        ),
        Step(
            "results_figure",
            (python, "figures/make_results_figure.py"),
            RESULTS_FIGURES,
        ),
        Step(
            "appendix_figures",
            (python, "figures/make_appendix_figures.py"),
            APPENDIX_FIGURES,
        ),
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git", *args),
        cwd=REPO_ROOT,
        check=check,
        capture_output=True,
        text=True,
    )


def git_info() -> dict[str, object]:
    status = _git("status", "--porcelain", "--untracked-files=all").stdout.splitlines()
    return {
        "commit": _git("rev-parse", "HEAD").stdout.strip(),
        "branch": _git("branch", "--show-current").stdout.strip(),
        "dirty": bool(status),
        "dirty_paths": status,
    }


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def _is_git_ignored(path: Path) -> bool:
    candidates = [path.resolve()]
    # Directory-only patterns do not match the bare name of a directory that
    # does not exist yet. A sentinel child verifies the rule before a run
    # creates that private/generated directory.
    if not path.exists() or path.is_dir():
        candidates.append(path.resolve() / ".codex-ignore-check")
    return any(
        _git(
            "check-ignore", "--no-index", "--quiet", "--", str(candidate), check=False
        ).returncode
        == 0
        for candidate in candidates
    )


def enforce_private_path(path: Path, description: str) -> None:
    """Refuse a repository-local private/generated path unless Git ignores it."""
    if _is_within(path, REPO_ROOT) and not _is_git_ignored(path):
        raise RuntimeError(
            f"{description} is inside the repository but is not Git-ignored: {path}. "
            "Move it outside the repository or add a narrow ignore rule first."
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(path: Path) -> dict[str, object]:
    """Describe an input without recording its confidential absolute path."""
    return {
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _software_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in SOFTWARE_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _weather_metadata(path: Path) -> dict[str, object]:
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Invalid weather provenance metadata: {path}: {exc}"
        ) from exc
    return {
        "source": metadata.get("weather_model", "unknown"),
        "source_url": metadata.get("source_url", "unknown"),
        "metadata": metadata,
    }


def _python_paths() -> list[Path]:
    return [
        PACKAGE_ROOT,
        PACKAGE_ROOT / "analysis",
        PACKAGE_ROOT / "analysis" / "Previous versions",
        PACKAGE_ROOT / "baselines",
        PACKAGE_ROOT / "config",
        PACKAGE_ROOT / "tests",
        PACKAGE_ROOT / "tests" / "abliations",
        PACKAGE_ROOT / "figures",
        PACKAGE_ROOT / "audit data",
    ]


def build_environment(data_dir: Path, output_dir: Path) -> dict[str, str]:
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    pythonpath = [str(path) for path in _python_paths()]
    if existing_pythonpath:
        pythonpath.append(existing_pythonpath)
    environment.update(
        {
            "PV_DATA_DIR": str(data_dir),
            "PV_OUTPUT_DIR": str(output_dir),
            "PV_DATA_PATH": str(data_dir / "PV_data.csv"),
            "PV_WEATHER_CACHE_PATH": str(data_dir / "weather_cache.csv"),
            "PV_WEATHER_META_PATH": str(data_dir / "weather_cache.meta.json"),
            "PV_RUN_TAG": "",
            "PYTHONPATH": os.pathsep.join(pythonpath),
            "PYTHONHASHSEED": environment.get("PV_SEED", "0"),
            "MPLBACKEND": "Agg",
        }
    )
    return environment


def _config_record(environment: dict[str, str]) -> dict[str, object]:
    old_environment = os.environ.copy()
    try:
        os.environ.update(environment)
        sys.path.insert(0, str(PACKAGE_ROOT / "config"))
        from config import load_config

        return load_config().describe()
    finally:
        if sys.path and sys.path[0] == str(PACKAGE_ROOT / "config"):
            sys.path.pop(0)
        os.environ.clear()
        os.environ.update(old_environment)


def atomic_write_manifest(path: Path, manifest: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _output_inventory(output_dir: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name in {MANIFEST_NAME, MANIFEST_NAME + ".tmp"}:
            continue
        rows.append(
            {
                "path": path.relative_to(output_dir).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def _step_outputs(step: Step, output_dir: Path) -> list[dict[str, object]]:
    return [
        {
            "path": filename,
            "size_bytes": (output_dir / filename).stat().st_size,
            "sha256": sha256_file(output_dir / filename),
        }
        for filename in step.expected_outputs
    ]


def _missing_outputs(step: Step, output_dir: Path) -> list[str]:
    return [name for name in step.expected_outputs if not (output_dir / name).is_file()]


def run_step(
    step: Step,
    index: int,
    base_environment: dict[str, str],
    output_dir: Path,
) -> dict[str, object]:
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{index:02d}_{step.name}.log"
    environment = base_environment.copy()
    environment.update(dict(step.environment))
    record: dict[str, object] = {
        "name": step.name,
        "command": list(step.command),
        "environment_overrides": dict(step.environment),
        "expected_outputs": list(step.expected_outputs),
        "log": log_path.relative_to(output_dir).as_posix(),
        "started_at_utc": _utc_now(),
        "status": "running",
    }
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8", newline="") as log:
        process = subprocess.Popen(
            step.command,
            cwd=PACKAGE_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = process.wait()
    record.update(
        {
            "finished_at_utc": _utc_now(),
            "duration_seconds": round(time.monotonic() - started, 3),
            "return_code": return_code,
        }
    )
    if return_code:
        record["status"] = "failed"
        raise StepFailure(
            record, f"Step {step.name!r} failed with exit code {return_code}"
        )
    missing = _missing_outputs(step, output_dir)
    if missing:
        record["status"] = "failed"
        record["missing_outputs"] = missing
        raise StepFailure(record, f"Step {step.name!r} did not create: {missing}")
    record["status"] = "succeeded"
    record["outputs"] = _step_outputs(step, output_dir)
    return record


class StepFailure(RuntimeError):
    def __init__(self, record: dict[str, object], message: str):
        super().__init__(message)
        self.record = record


def _sanitised_command(arguments: list[str]) -> list[str]:
    redacted = list(arguments)
    for option, replacement in (
        ("--data-dir", "<private-data-dir>"),
        ("--output-dir", "<run-output-dir>"),
    ):
        if option in redacted:
            position = redacted.index(option) + 1
            if position < len(redacted):
                redacted[position] = replacement
    return redacted


def create_manifest(
    command: list[str],
    git: dict[str, object],
    config: dict[str, object],
    seeds: list[int],
    pv_data: Path,
    weather_cache: Path,
    weather_meta: Path,
) -> dict[str, object]:
    """Create the initial provenance record before any scientific step runs."""
    return {
        "schema_version": 1,
        "pipeline": "physics-aware-day-ahead-pv-camera-ready",
        "status": "running",
        "started_at_utc": _utc_now(),
        "command": _sanitised_command(command),
        "git": git,
        "random_seeds": {
            "main_seed": config["seed"],
            "cnn_seeds": config["cnn_seeds"],
            "multiseed": seeds,
            "python_hash_seed": str(config["seed"]),
        },
        "configuration": config,
        "data": {
            "pv_input": fingerprint(pv_data),
            "weather_input": fingerprint(weather_cache),
            "date_range": None,
            "reporting_capacity_kw": None,
        },
        "weather": {
            **_weather_metadata(weather_meta),
            "metadata_input": fingerprint(weather_meta),
        },
        "software_versions": _software_versions(),
        "runtime": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "steps": [],
        "output_inventory": [],
    }


def _update_from_audit(manifest: dict[str, object], output_dir: Path) -> None:
    audit = json.loads((output_dir / "data_audit_summary.json").read_text("utf-8"))
    data = manifest["data"]
    assert isinstance(data, dict)
    data["date_range"] = {
        "pv_start": audit["pv_raw_start"],
        "pv_end": audit["pv_raw_end"],
        "weather_start": audit["weather_start"],
        "weather_end": audit["weather_end"],
    }
    data["reporting_capacity_kw"] = audit["empirical_capacity_q999_kw"]
    data["audit"] = audit


def _validate_resume(
    manifest_path: Path,
    current_manifest: dict[str, object],
) -> None:
    if not manifest_path.is_file():
        return
    previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = (
        ("Git commit", previous["git"]["commit"], current_manifest["git"]["commit"]),
        (
            "PV input checksum",
            previous["data"]["pv_input"]["sha256"],
            current_manifest["data"]["pv_input"]["sha256"],
        ),
        (
            "weather checksum",
            previous["data"]["weather_input"]["sha256"],
            current_manifest["data"]["weather_input"]["sha256"],
        ),
        (
            "configuration",
            previous["configuration"],
            current_manifest["configuration"],
        ),
        (
            "multi-seed list",
            previous["random_seeds"]["multiseed"],
            current_manifest["random_seeds"]["multiseed"],
        ),
    )
    changed = [name for name, old, new in checks if old != new]
    if changed:
        raise RuntimeError(
            "Cannot resume because provenance changed: " + ", ".join(changed)
        )


def _default_output_dir(git_commit: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return PACKAGE_ROOT / "camera_ready_outputs" / f"run-{stamp}-{git_commit[:8]}"


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PACKAGE_ROOT / "private_data",
        help=(
            "Local directory containing PV_data.csv, weather_cache.csv, and "
            "weather_cache.meta.json."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Run directory (default: unique directory under camera_ready_outputs).",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse completed steps only if commit, config, seeds, and input hashes "
            "match."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without reading data or running models.",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow an uncommitted working tree (development only; not camera-ready).",
    )
    return parser.parse_args(arguments)


def _print_plan(steps: tuple[Step, ...], data_dir: Path, output_dir: Path) -> None:
    print(f"Private input directory: {data_dir}")
    print(f"Generated output directory: {output_dir}")
    for index, step in enumerate(steps, 1):
        command = (
            subprocess.list2cmdline(step.command)
            if os.name == "nt"
            else shlex.join(step.command)
        )
        print(f"{index:02d}. {step.name}: {command}")
        print("    outputs: " + ", ".join(step.expected_outputs))


def main(arguments: list[str] | None = None) -> int:
    args = parse_args(arguments)
    if len(set(args.seeds)) != len(args.seeds):
        raise SystemExit("--seeds must not contain duplicates")
    git = git_info()
    output_dir = (args.output_dir or _default_output_dir(str(git["commit"]))).resolve()
    data_dir = args.data_dir.resolve()
    steps = build_steps(sys.executable, args.seeds)
    _print_plan(steps, data_dir, output_dir)
    if args.dry_run:
        print("Dry run only: no inputs were read and no files were created.")
        return 0

    if git["dirty"] and not args.allow_dirty:
        raise RuntimeError(
            "Camera-ready runs require a clean Git worktree. Commit the pipeline "
            "changes first, or use --allow-dirty only for development runs."
        )

    input_paths = (
        data_dir / "PV_data.csv",
        data_dir / "weather_cache.csv",
        data_dir / "weather_cache.meta.json",
    )
    missing_inputs = [path.name for path in input_paths if not path.is_file()]
    if missing_inputs:
        raise FileNotFoundError(
            f"Private input directory {data_dir} is missing: "
            f"{', '.join(missing_inputs)}"
        )
    enforce_private_path(data_dir, "Private data directory")
    enforce_private_path(output_dir, "Generated output directory")

    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise RuntimeError(
            f"Output directory is not empty: {output_dir}. "
            "Use a new directory or --resume."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_NAME
    if args.resume and any(output_dir.iterdir()) and not manifest_path.is_file():
        raise RuntimeError(
            "Cannot resume a non-empty output directory without run_manifest.json."
        )
    environment = build_environment(data_dir, output_dir)
    config = _config_record(environment)
    manifest = create_manifest(
        [sys.executable, str(Path(__file__).name), *(arguments or sys.argv[1:])],
        git,
        config,
        list(args.seeds),
        *input_paths,
    )
    if args.resume:
        _validate_resume(manifest_path, manifest)
    atomic_write_manifest(manifest_path, manifest)

    try:
        for index, step in enumerate(steps, 1):
            if args.resume and not _missing_outputs(step, output_dir):
                record = {
                    "name": step.name,
                    "command": list(step.command),
                    "environment_overrides": dict(step.environment),
                    "expected_outputs": list(step.expected_outputs),
                    "status": "reused",
                    "outputs": _step_outputs(step, output_dir),
                }
                print(f"Reusing complete step: {step.name}")
            else:
                print(f"\n=== Step {index}/{len(steps)}: {step.name} ===")
                record = run_step(step, index, environment, output_dir)
            manifest["steps"].append(record)
            if step.name == "data_audit":
                _update_from_audit(manifest, output_dir)
            atomic_write_manifest(manifest_path, manifest)
    except StepFailure as exc:
        manifest["steps"].append(exc.record)
        manifest["status"] = "failed"
        manifest["finished_at_utc"] = _utc_now()
        manifest["failure"] = str(exc)
        manifest["output_inventory"] = _output_inventory(output_dir)
        atomic_write_manifest(manifest_path, manifest)
        raise
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["finished_at_utc"] = _utc_now()
        manifest["failure"] = f"{type(exc).__name__}: {exc}"
        manifest["output_inventory"] = _output_inventory(output_dir)
        atomic_write_manifest(manifest_path, manifest)
        raise

    manifest["status"] = "succeeded"
    manifest["finished_at_utc"] = _utc_now()
    manifest["output_inventory"] = _output_inventory(output_dir)
    atomic_write_manifest(manifest_path, manifest)
    print(f"\nCamera-ready pipeline succeeded. Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
