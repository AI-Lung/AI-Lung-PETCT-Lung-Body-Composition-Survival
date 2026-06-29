from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.metadata
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"
BOOTSTRAP_PACKAGES = ["pip", "setuptools", "wheel"]


@dataclass(frozen=True)
class RequirementSpec:
    raw: str
    name: str
    specifiers: tuple[tuple[str, str], ...]


def main() -> int:
    args = _build_parser().parse_args()
    os.chdir(PROJECT_ROOT)

    print("=" * 14 + " PETCT BodyComp Windows Bootstrap " + "=" * 14)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version.split()[0]}")

    if sys.version_info < (3, 10):
        print("[ERROR] Python 3.10 or newer is required.")
        return 1

    _ensure_pip_available()
    try:
        requirements = _load_requirements(REQUIREMENTS_FILE)
    except ValueError as exc:
        print(f"[ERROR] Invalid requirement entry: {exc}")
        return 1

    unsatisfied = _collect_unsatisfied_requirements(requirements)
    if unsatisfied:
        print("[INFO] Missing or outdated packages detected:")
        for item in unsatisfied:
            print(f"  - {item}")
        try:
            _install_requirements(REQUIREMENTS_FILE)
        except subprocess.CalledProcessError as exc:
            print(f"[ERROR] Package installation failed with exit code {exc.returncode}.")
            print("Please check your internet connection and try again.")
            return int(exc.returncode or 1)
        unsatisfied = _collect_unsatisfied_requirements(requirements)
        if unsatisfied:
            print("[ERROR] Some requirements are still not satisfied after installation:")
            for item in unsatisfied:
                print(f"  - {item}")
            return 1
        print("[INFO] All required packages are now available.")
    else:
        print("[INFO] All required packages are already installed.")

    if args.check_only:
        print("[INFO] Check-only mode finished successfully.")
        return 0

    try:
        import tkinter  # noqa: F401
    except Exception as exc:
        print(f"[ERROR] tkinter is unavailable in this Python environment: {exc}")
        print("Please use a standard Python or Anaconda/Miniconda environment with Tk support.")
        return 1

    try:
        from petct_bodycomp.gui import main as app_main

        return int(app_main() or 0)
    except KeyboardInterrupt:
        print("\n[INFO] Application stopped by user.")
        return 0
    except Exception:
        print("[ERROR] The application exited unexpectedly:")
        traceback.print_exc()
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bootstrap launcher for PET/CT BodyComp Extractor on Windows.")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Check the environment and required packages without launching the app.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser


def _ensure_pip_available() -> None:
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        print("[INFO] pip is missing. Running ensurepip...")
        import ensurepip

        ensurepip.bootstrap(upgrade=True)


def _load_requirements(path: Path) -> list[RequirementSpec]:
    requirements: list[RequirementSpec] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        requirements.append(_parse_requirement(raw))
    return requirements


def _parse_requirement(raw: str) -> RequirementSpec:
    match = re.fullmatch(r"([A-Za-z0-9_.-]+)\s*(.*)", raw)
    if not match:
        raise ValueError(raw)

    name = match.group(1)
    remainder = match.group(2).strip()
    specifiers: list[tuple[str, str]] = []

    if remainder:
        for chunk in remainder.split(","):
            spec = chunk.strip()
            spec_match = re.fullmatch(r"(<=|>=|==|!=|<|>|~=)\s*([A-Za-z0-9_.!+-]+)", spec)
            if not spec_match:
                raise ValueError(raw)
            specifiers.append((spec_match.group(1), spec_match.group(2)))

    return RequirementSpec(raw=raw, name=name, specifiers=tuple(specifiers))


def _collect_unsatisfied_requirements(requirements: list[RequirementSpec]) -> list[str]:
    unsatisfied: list[str] = []
    for requirement in requirements:
        normalized_name = _normalize_dist_name(requirement.name)
        try:
            installed_version = importlib.metadata.version(normalized_name)
        except importlib.metadata.PackageNotFoundError:
            unsatisfied.append(f"{requirement.name} (not installed)")
            continue

        if not _specifiers_satisfied(installed_version, requirement.specifiers):
            unsatisfied.append(f"{requirement.name} ({installed_version} does not satisfy {requirement.raw})")
    return unsatisfied


def _install_requirements(requirements_path: Path) -> None:
    print("[INFO] Installing required packages. This may take several minutes on first run...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", *BOOTSTRAP_PACKAGES],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(requirements_path)],
        check=True,
    )


def _normalize_dist_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _specifiers_satisfied(installed_version: str, specifiers: tuple[tuple[str, str], ...]) -> bool:
    for operator_text, expected_version in specifiers:
        if not _compare_versions(installed_version, operator_text, expected_version):
            return False
    return True


def _compare_versions(installed_version: str, operator_text: str, expected_version: str) -> bool:
    installed_key = _version_key(installed_version)
    expected_key = _version_key(expected_version)

    if operator_text == "==":
        return installed_key == expected_key
    if operator_text == "!=":
        return installed_key != expected_key
    if operator_text == ">":
        return installed_key > expected_key
    if operator_text == ">=":
        return installed_key >= expected_key
    if operator_text == "<":
        return installed_key < expected_key
    if operator_text == "<=":
        return installed_key <= expected_key
    if operator_text == "~=":
        return installed_key >= expected_key and installed_key < _compatible_release_upper_bound(expected_version)
    raise ValueError(f"Unsupported version operator: {operator_text}")


def _compatible_release_upper_bound(version_text: str) -> tuple[object, ...]:
    release_parts = [int(item) for item in re.findall(r"\d+", version_text)]
    if not release_parts:
        raise ValueError(f"Invalid compatible release version: {version_text}")

    if len(release_parts) == 1:
        return (release_parts[0] + 1,)

    upper_parts = release_parts[:-1]
    upper_parts[-1] += 1
    return tuple(upper_parts)


def _version_key(version_text: str) -> tuple[object, ...]:
    parts = re.findall(r"\d+|[A-Za-z]+", version_text)
    normalized: list[object] = []
    for item in parts:
        normalized.append(int(item) if item.isdigit() else item.lower())
    while len(normalized) > 1 and normalized[-1] == 0:
        normalized.pop()
    return tuple(normalized)


if __name__ == "__main__":
    raise SystemExit(main())
