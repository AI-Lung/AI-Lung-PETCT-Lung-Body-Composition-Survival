# -*- coding: utf-8 -*-
"""
Run TotalSegmentator for whole-body (bones, liver) and tissue_4_types (SM, SAT, IMAT, TAT).
Supports CLI (TotalSegmentator / totalsegmentator) and Python API fallback.
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from typing import Tuple, Optional

# Task names
TASK_TOTAL = "total"
TASK_TISSUE_4 = "tissue_4_types"
ENV_NAME = "petct_bodycomp"


def _get_conda_exe() -> Optional[str]:
    """If current Python is from Anaconda/Miniconda, return path to conda.exe so we don't rely on PATH."""
    exe = os.path.normpath(sys.executable)
    # ...\anaconda3\python.exe or ...\miniconda3\python.exe -> ...\Scripts\conda.exe
    for name in ("anaconda3", "miniconda3", "Anaconda3", "Miniconda3"):
        if name in exe:
            base = os.path.dirname(exe)
            if "envs" in exe.split(os.sep):
                # e.g. D:\anaconda3\envs\other\python.exe -> base = D:\anaconda3
                idx = exe.split(os.sep).index("envs")
                base = os.sep.join(exe.split(os.sep)[:idx])
            conda_exe = os.path.join(base, "Scripts", "conda.exe")
            if os.path.isfile(conda_exe):
                return conda_exe
            conda_bat = os.path.join(base, "Scripts", "conda.bat")
            if os.path.isfile(conda_bat):
                return conda_bat
            break
    return None


def _find_project_env_python() -> Optional[str]:
    """Return path to the project environment's Python if it can be found."""
    # Already in the project environment
    if os.environ.get("CONDA_DEFAULT_ENV") == ENV_NAME:
        return sys.executable
    exe = os.path.normpath(sys.executable)
    if "envs" in exe.split(os.sep):
        parts = exe.split(os.sep)
        idx = parts.index("envs")
        parent = parts[idx + 1] if idx + 1 < len(parts) else ""
        if parent.lower() == ENV_NAME:
            return sys.executable

    # 1) Ask conda for dl_env's Python (works regardless of install path)
    try:
        r = subprocess.run(
            ["conda", "run", "-n", ENV_NAME, "python", "-c", "import sys; print(sys.executable)"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if r.returncode == 0 and r.stdout and os.path.isfile(r.stdout.strip()):
            return r.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass

    # 2) Guess path: conda base -> envs\dl_env\python
    if sys.platform == "win32":
        if "envs" in exe.split(os.sep):
            parts = exe.split(os.sep)
            idx = parts.index("envs")
            base = os.sep.join(parts[:idx])
        else:
            base = os.path.dirname(exe)
        py = os.path.join(base, "envs", ENV_NAME, "python.exe")
    else:
        if "envs" in exe.split(os.sep):
            parts = exe.split(os.sep)
            idx = parts.index("envs")
            base = os.sep.join(parts[:idx])
        else:
            base = os.path.dirname(exe)
        py = os.path.join(base, "envs", ENV_NAME, "bin", "python")
    if os.path.isfile(py):
        return py

    # 3) Venv in app dir or parent
    app_dir = os.path.dirname(os.path.abspath(__file__))
    for rel in [ENV_NAME, "..", os.path.join("..", ENV_NAME), os.path.join("..", "..", ENV_NAME)]:
        base = os.path.normpath(os.path.join(app_dir, rel))
        py = os.path.join(base, "Scripts", "python.exe") if sys.platform == "win32" else os.path.join(base, "bin", "python")
        if os.path.isfile(py):
            return py
    return None


def _device_to_cli(device: str) -> str:
    """TotalSegmentator CLI uses 'gpu' or 'cpu', not 'cuda'."""
    if device.lower() in ("cuda", "gpu"):
        return "gpu"
    return "cpu"


def _find_totalseg_exe(python_exe: str) -> Optional[str]:
    """Find TotalSegmentator CLI executable next to a given python.exe."""
    scripts_dir = os.path.join(os.path.dirname(python_exe), "Scripts")
    if not os.path.isdir(scripts_dir):
        scripts_dir = os.path.dirname(python_exe)
    for name in ("TotalSegmentator.exe", "TotalSegmentator", "totalsegmentator.exe", "totalsegmentator"):
        p = os.path.join(scripts_dir, name)
        if os.path.isfile(p):
            return p
    return None


def _filter_error(text: str) -> str:
    """Strip tqdm progress bars and blank lines, keep actual error messages."""
    lines = text.splitlines()
    filtered = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if "%" in stripped and "|" in stripped and ("it/s" in stripped or "s/it" in stripped):
            continue
        if any(ch in stripped for ch in ("\u2588", "\u258f", "\u258e", "\u258d", "\u258c", "\u258b", "\u258a", "\u2589", "it/s", "s/it")):
            continue
        if stripped.startswith("0%|") or stripped.endswith("?it/s]"):
            continue
        filtered.append(stripped)
    return "\n".join(filtered)


def _is_memory_error(err_text: str) -> bool:
    err_lower = (err_text or "").lower()
    return any(kw in err_lower for kw in ("memoryerror", "arraymemoryerror", "unable to allocate", "out of memory"))


def _find_ts_executable() -> Optional[str]:
    """Find TotalSegmentator CLI executable from the current or project environment."""
    ts = _find_totalseg_exe(sys.executable)
    if ts:
        return ts
    dl_py = _find_project_env_python()
    if dl_py and dl_py != sys.executable:
        ts = _find_totalseg_exe(dl_py)
        if ts:
            return ts
    return None


TASKS_NO_FAST = {"tissue_4_types", "tissue_types", "tissue_types_mr"}


def _low_mem_env() -> dict:
    """Environment variables that force single-process / single-thread execution."""
    env = os.environ.copy()
    env["nnUNet_def_n_proc"] = "1"
    env["nnUNet_n_proc_DA"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    return env


def _run_totalsegmentator_cli(
    input_nii: str,
    output_dir: str,
    task: str,
    device: str,
    timeout: int,
    fast: bool = False,
    force_split: bool = False,
    nr_thr_resamp: int = 1,
    nr_thr_saving: int = 1,
) -> Tuple[bool, str]:
    """
    Run TotalSegmentator via its CLI executable.
    On MemoryError, auto-retries with progressively more aggressive memory savings.
    """
    import shutil
    output_dir = os.path.abspath(output_dir)
    input_nii = os.path.abspath(input_nii)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    device_cli = _device_to_cli(device)

    def _read_tail(path: Path, limit: int = 12000) -> str:
        if not path.is_file():
            return ""
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        except Exception:
            return ""
        return text[-limit:]

    def _kill_process_tree(pid: int) -> None:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            time.sleep(1.0)
            return
        try:
            os.kill(pid, 9)
        except Exception:
            pass

    def _kill_spawn_children(parent_pid: int) -> None:
        """Clean up Windows multiprocessing children orphaned after a failed CLI run."""
        if sys.platform != "win32":
            return
        pattern = f"parent_pid={parent_pid}"
        ps_command = (
            "Get-CimInstance Win32_Process | "
            f"Where-Object {{ $_.CommandLine -like '*{pattern}*' }} | "
            "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
        )
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_command],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        time.sleep(0.5)

    def _run_cmd(cmd: list, env: dict = None) -> Tuple[bool, str]:
        stdout_log = Path(output_dir).parent / f"{Path(output_dir).name}_{task}_stdout.log"
        stderr_log = Path(output_dir).parent / f"{Path(output_dir).name}_{task}_stderr.log"
        try:
            with open(stdout_log, "w", encoding="utf-8", errors="replace") as out_f, open(
                stderr_log, "w", encoding="utf-8", errors="replace"
            ) as err_f:
                creationflags = 0
                if sys.platform == "win32":
                    creationflags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
                proc = subprocess.Popen(
                    cmd,
                    stdout=out_f,
                    stderr=err_f,
                    text=True,
                    cwd=os.path.dirname(input_nii) or ".",
                    env=env or os.environ.copy(),
                    close_fds=True,
                    creationflags=creationflags,
                )
                try:
                    return_code = proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    _kill_process_tree(proc.pid)
                    _kill_spawn_children(proc.pid)
                    return False, (
                        f"Timeout after {timeout}s. Try GPU or increase timeout in the Segmentation Options tab.\n"
                        f"stdout log: {stdout_log}\nstderr log: {stderr_log}"
                    )

            if return_code == 0 and any(Path(output_dir).glob("*.nii*")):
                return True, ""
            _kill_spawn_children(proc.pid)
            out = _read_tail(stdout_log)
            err = _read_tail(stderr_log)
            error_msg = _filter_error(err) or _filter_error(out) or f"Exit code {return_code}"
            return False, error_msg + f"\nstdout log: {stdout_log}\nstderr log: {stderr_log}"
        except subprocess.TimeoutExpired:
            return False, f"Timeout after {timeout}s. Try GPU or increase timeout in the Segmentation Options tab."
        except FileNotFoundError as e:
            return False, str(e)
        except Exception as e:
            return False, str(e)

    def _build_args(use_fast: bool = False, use_fastest: bool = False,
                    use_split: bool = False, use_body_seg: bool = False) -> list:
        args = ["-i", input_nii, "-o", output_dir, "-ta", task, "-d", device_cli,
                "-nr", str(nr_thr_resamp), "-ns", str(nr_thr_saving)]
        task_supports_fast = task not in TASKS_NO_FAST
        if use_fastest and task_supports_fast:
            args.append("--fastest")
        elif use_fast and task_supports_fast:
            args.append("--fast")
        if use_split:
            args.append("--force_split")
        if use_body_seg:
            args.append("--body_seg")
        return args

    def _clean_output():
        shutil.rmtree(output_dir, ignore_errors=True)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    ts_exe = _find_ts_executable()
    if not ts_exe:
        return False, "TotalSegmentator executable not found. Run the Windows launcher again or install with: pip install TotalSegmentator"

    low_env = _low_mem_env()

    # Attempt 1: user settings + low-memory env vars
    cli_args = _build_args(use_fast=fast, use_split=force_split)
    ok, err = _run_cmd([ts_exe] + cli_args, env=low_env)
    if ok:
        return True, ""
    if not _is_memory_error(err):
        return False, "TotalSegmentator failed. " + (err or "")[:2000]

    # Attempt 2: + force_split + body_seg
    _clean_output()
    cli_args = _build_args(use_fast=fast, use_split=True, use_body_seg=True)
    ok, err = _run_cmd([ts_exe] + cli_args, env=low_env)
    if ok:
        return True, "(low-RAM: --force_split --body_seg) "
    if not _is_memory_error(err):
        return False, "TotalSegmentator failed. " + (err or "")[:2000]

    # Attempt 3: + fast (3mm) + force_split + body_seg
    _clean_output()
    cli_args = _build_args(use_fast=True, use_split=True, use_body_seg=True)
    ok, err = _run_cmd([ts_exe] + cli_args, env=low_env)
    if ok:
        return True, "(low-RAM: --fast --force_split --body_seg) "
    if not _is_memory_error(err):
        return False, "TotalSegmentator failed. " + (err or "")[:2000]

    # Attempt 4: fastest (6mm) + force_split + body_seg (most aggressive)
    _clean_output()
    cli_args = _build_args(use_fastest=True, use_split=True, use_body_seg=True)
    ok, err = _run_cmd([ts_exe] + cli_args, env=low_env)
    if ok:
        return True, "(low-RAM: --fastest --force_split --body_seg) "

    msg = "TotalSegmentator failed (MemoryError after all retries). " + (err or "")[:2000]
    msg += "\nYour system does not have enough RAM. Try: close all other programs, or run on a machine with more RAM (>=16 GB recommended)."
    return False, msg


def run_totalsegmentator(
    input_nii: str,
    output_dir: str,
    task: str = TASK_TISSUE_4,
    device: str = "cuda",
    timeout: int = 1800,
    fast: bool = False,
    force_split: bool = False,
    nr_thr_resamp: int = 1,
    nr_thr_saving: int = 1,
) -> Tuple[bool, str]:
    """
    Run TotalSegmentator on input_nii, write masks to output_dir.
    Returns (success, error_message). error_message is empty on success.
    task: 'total' for whole-body (bones, liver), 'tissue_4_types' for 4 tissues.
    """
    return _run_totalsegmentator_cli(
        input_nii, output_dir, task, device, timeout,
        fast=fast, force_split=force_split,
        nr_thr_resamp=nr_thr_resamp, nr_thr_saving=nr_thr_saving,
    )


def run_segmentation_pipeline(
    ct_nii_path: str,
    work_dir: str,
    device: str = "cuda",
    timeout_total: int = 5400,
    timeout_tissue: int = 2400,
    fast: bool = False,
    force_split: bool = False,
    nr_thr_resamp: int = 1,
    nr_thr_saving: int = 1,
) -> Tuple[Optional[str], Optional[str], str]:
    """
    Run both total and tissue_4_types on CT. Returns:
    - mask_dir_total: path to whole-body segmentation (bones, liver)
    - mask_dir_4tissue: path to 4-tissue segmentation
    - message: success or error string (includes stderr on failure)
    """
    work_dir = os.path.abspath(work_dir)
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    out_total = os.path.join(work_dir, "seg_total")
    out_tissue = os.path.join(work_dir, "seg_4tissue")

    if not os.path.isfile(ct_nii_path):
        return None, None, f"CT file not found: {ct_nii_path}"

    seg_kwargs = dict(
        fast=fast, force_split=force_split,
        nr_thr_resamp=nr_thr_resamp, nr_thr_saving=nr_thr_saving,
    )

    notes = []

    ok_total, msg_total = run_totalsegmentator(
        ct_nii_path, out_total, task=TASK_TOTAL, device=device,
        timeout=timeout_total, **seg_kwargs,
    )
    if not ok_total:
        return None, None, "TotalSegmentator (total) failed or timed out. " + (msg_total or "Check CT path and install: pip install TotalSegmentator.")
    if msg_total:
        notes.append("total: " + msg_total)

    ok_tissue, msg_tissue = run_totalsegmentator(
        ct_nii_path, out_tissue, task=TASK_TISSUE_4, device=device,
        timeout=timeout_tissue, **seg_kwargs,
    )
    if not ok_tissue:
        return out_total, None, "TotalSegmentator (tissue_4_types) failed or timed out. " + (msg_tissue or "")
    if msg_tissue:
        notes.append("tissue: " + msg_tissue)

    result_msg = "Segmentation completed successfully."
    if notes:
        result_msg += " " + " | ".join(notes)
    return out_total, out_tissue, result_msg
