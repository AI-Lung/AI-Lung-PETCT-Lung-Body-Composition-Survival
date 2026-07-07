from __future__ import annotations

import queue
import re
import shutil
import subprocess
import tempfile
import threading
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import pandas as pd

from .feature_extraction import ALL_FEATURE_NAMES, BONE_KEYWORDS, TISSUE_MASK_MAP, extract_all_features
from .risk_prediction import build_clinical_inputs, load_risk_model_spec, predict_risk
from .segmentation import run_segmentation_pipeline


APP_TITLE = "PET/CT Body Composition Extractor"
APP_SUBTITLE = (
    "Local TotalSegmentator workflow for PET/CT tissue segmentation and "
    "body-composition extraction with optional research-only OS/PFS risk scoring."
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
CSV_FILE_NAME = "petct_body_composition_parameters.csv"
RISK_CSV_FILE_NAME = "petct_research_risk_scores.csv"
COMBINED_CSV_FILE_NAME = "petct_body_composition_and_research_risk_scores.csv"
DEFAULT_RISK_CUTOFF_REFERENCE = "development_median"
RISK_EXPORT_DROP_COLUMNS = [
    "Selected cutoff",
    "Selected cutoff reference",
    "Development median group",
    "External validation median group",
]


def _detect_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


@dataclass(frozen=True)
class PipelineConfig:
    ct_path: Path
    pet_path: Path
    output_root: Path
    weight_kg: float
    dose_mbq: float
    height_cm: float | None
    pet_is_suv: bool
    skip_segmentation: bool
    mask_dir_total: Path | None
    mask_dir_4tissue: Path | None
    device: str
    force_split: bool
    fast_mode: bool
    nr_thr_resamp: int
    nr_thr_saving: int
    timeout_total_min: int
    timeout_tissue_min: int
    enable_risk_prediction: bool
    age: float | None
    sex_label: str | None
    clinical_stage_label: str | None
    histology_label: str | None
    tumor_suvmax: float | None
    cutoff_reference: str


@dataclass(frozen=True)
class PipelineResult:
    features_df: pd.DataFrame
    feature_csv_path: Path
    run_dir: Path
    risk_df: pd.DataFrame | None = None
    risk_csv_path: Path | None = None
    combined_csv_path: Path | None = None


@dataclass(frozen=True)
class MaskPair:
    mask_dir_total: Path
    mask_dir_4tissue: Path
    source_dir: Path


class LauncherGUI:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title(APP_TITLE)
        self._configure_window_geometry()
        self.root.configure(bg="#EEF3F8")

        self.log_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.result_df: pd.DataFrame | None = None
        self.result_csv_path: Path | None = None
        self.result_dir: Path | None = None
        self.risk_df: pd.DataFrame | None = None
        self.risk_csv_path: Path | None = None
        self.combined_csv_path: Path | None = None
        self.cuda_available = _detect_cuda_available()
        self.risk_spec = load_risk_model_spec()

        self.ct_path_var = tk.StringVar()
        self.pet_path_var = tk.StringVar()
        self.output_dir_var = tk.StringVar(value=str(DEFAULT_OUTPUT_ROOT))
        self.weight_var = tk.StringVar(value="70.0")
        self.dose_var = tk.StringVar(value="300.0")
        self.height_var = tk.StringVar(value="0.0")
        self.pet_is_suv_var = tk.BooleanVar(value=False)

        self.skip_segmentation_var = tk.BooleanVar(value=False)
        self.mask_total_var = tk.StringVar()
        self.mask_4tissue_var = tk.StringVar()

        self.use_gpu_var = tk.BooleanVar(value=self.cuda_available)
        self.force_split_var = tk.BooleanVar(value=True)
        self.fast_mode_var = tk.BooleanVar(value=not self.cuda_available)
        self.nr_thr_resamp_var = tk.StringVar(value="1")
        self.nr_thr_saving_var = tk.StringVar(value="1")
        self.timeout_total_var = tk.StringVar(value="120")
        self.timeout_tissue_var = tk.StringVar(value="120")

        clinical_mapping = self.risk_spec["clinical_input_mapping"]
        self.enable_risk_prediction_var = tk.BooleanVar(value=False)
        self.age_var = tk.StringVar(value="62")
        self.sex_var = tk.StringVar(value=next(iter(clinical_mapping["Gender"])))
        self.stage_var = tk.StringVar(value=next(iter(clinical_mapping["Cli"])))
        self.histology_var = tk.StringVar(value=next(iter(clinical_mapping["Path"])))
        self.tumor_suvmax_var = tk.StringVar()

        self.status_var = tk.StringVar(value="Select CT/PET NIfTI files, configure options, then start the pipeline.")
        self.input_preview_var = tk.StringVar(value="Input preview: CT and PET not selected.")
        self.output_preview_var = tk.StringVar(value=f"Results will be saved under {DEFAULT_OUTPUT_ROOT}")
        self.device_hint_var = tk.StringVar(
            value=(
                "CUDA detected. GPU mode is available."
                if self.cuda_available
                else "CUDA is not available in this Python environment. CPU mode is selected."
            )
        )

        self._configure_style()
        self._build_layout()
        self._bind_events()
        self._update_mask_widgets()
        self._update_risk_widgets()
        self.root.after(150, self._drain_log_queue)

    def _configure_window_geometry(self) -> None:
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        target_width = 1280
        target_height = 920
        window_width = min(target_width, max(980, screen_width - 80))
        window_height = min(target_height, max(700, screen_height - 80))
        min_width = min(1120, max(900, screen_width - 120))
        min_height = min(760, max(660, screen_height - 120))

        offset_x = max((screen_width - window_width) // 2, 0)
        offset_y = max((screen_height - window_height) // 2, 0)
        self.root.geometry(f"{window_width}x{window_height}+{offset_x}+{offset_y}")
        self.root.minsize(min_width, min_height)

    def _configure_style(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(".", background="#EEF3F8", foreground="#17324D")
        style.configure("App.TFrame", background="#EEF3F8")
        style.configure("Hero.TFrame", background="#102A43")
        style.configure("Card.TLabelframe", background="#F9FBFD", bordercolor="#D7E2EC", relief="solid")
        style.configure("Card.TLabelframe.Label", font=("Helvetica", 11, "bold"), foreground="#16324F")
        style.configure("Muted.TLabel", background="#EEF3F8", foreground="#5C6B7A")
        style.configure("SmallMuted.TLabel", background="#EEF3F8", foreground="#6A7785", font=("Helvetica", 10))
        style.configure("Primary.TButton", padding=(16, 10), background="#0F6E8C", foreground="white")
        style.map("Primary.TButton", background=[("active", "#135C75")])
        style.configure("Secondary.TButton", padding=(12, 8))
        style.configure("TNotebook", background="#EEF3F8", borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            background="#DCE7F1",
            foreground="#24425C",
            padding=(14, 8),
            font=("Helvetica", 10, "bold"),
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", "#0F3D5E"), ("active", "#C8D7E5")],
            foreground=[("selected", "#F7FBFF")],
        )
        style.configure("Treeview", rowheight=28, background="#FFFFFF", fieldbackground="#FFFFFF")
        style.configure("Treeview.Heading", font=("Helvetica", 10, "bold"))

    def _build_layout(self) -> None:
        main_frame = ttk.Frame(self.root, padding=16, style="App.TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True)

        self._build_hero(main_frame)
        self._build_action_bar(main_frame)

        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=(14, 0))

        self.input_tab = ttk.Frame(notebook, padding=16, style="App.TFrame")
        self.options_tab = ttk.Frame(notebook, padding=16, style="App.TFrame")
        self.risk_tab = ttk.Frame(notebook, padding=16, style="App.TFrame")
        self.results_tab = ttk.Frame(notebook, padding=16, style="App.TFrame")
        self.log_tab = ttk.Frame(notebook, padding=16, style="App.TFrame")

        notebook.add(self.input_tab, text="Input Data")
        notebook.add(self.options_tab, text="Segmentation Options")
        notebook.add(self.risk_tab, text="Risk Prediction")
        notebook.add(self.results_tab, text="Results")
        notebook.add(self.log_tab, text="Run Log")

        self._build_input_tab()
        self._build_options_tab()
        self._build_risk_tab()
        self._build_results_tab()
        self._build_log_tab()

    def _build_action_bar(self, parent: ttk.Frame) -> None:
        action_bar = tk.Frame(parent, bg="#F7FBFF", padx=18, pady=12, highlightbackground="#D7E2EC", highlightthickness=1)
        action_bar.pack(fill=tk.X, pady=(12, 0))
        status_block = tk.Frame(action_bar, bg="#F7FBFF")
        status_block.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(
            status_block,
            text="Run Controls",
            bg="#F7FBFF",
            fg="#0F6E8C",
            font=("Helvetica", 10, "bold"),
        ).pack(anchor="w")
        tk.Label(
            status_block,
            textvariable=self.status_var,
            bg="#F7FBFF",
            fg="#4A5A6A",
            font=("Helvetica", 10),
        ).pack(anchor="w", pady=(2, 0))

        self.open_output_button = ttk.Button(
            action_bar,
            text="Open Output Folder",
            command=self._open_output_folder,
            style="Secondary.TButton",
            state=tk.DISABLED,
        )
        self.open_output_button.pack(side=tk.RIGHT, padx=(10, 0))

        self.start_button = ttk.Button(
            action_bar,
            text="Run PET/CT Pipeline",
            command=self._start_run,
            style="Primary.TButton",
        )
        self.start_button.pack(side=tk.RIGHT)

    def _build_hero(self, parent: ttk.Frame) -> None:
        hero = tk.Frame(parent, bg="#102A43", padx=22, pady=18)
        hero.pack(fill=tk.X)

        left = tk.Frame(hero, bg="#102A43")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Label(
            left,
            text=APP_TITLE,
            bg="#102A43",
            fg="#F8FBFF",
            font=("Helvetica", 24, "bold"),
        ).pack(anchor="w")
        tk.Label(
            left,
            text=APP_SUBTITLE,
            bg="#102A43",
            fg="#C7D4E2",
            font=("Helvetica", 11),
            wraplength=780,
            justify=tk.LEFT,
        ).pack(anchor="w", pady=(6, 10))

        chip_frame = tk.Frame(left, bg="#102A43")
        chip_frame.pack(anchor="w")
        self._add_chip(chip_frame, "Input", "CT + PET NIfTI")
        self._add_chip(chip_frame, "Segmentation", "Total + tissue_4_types")
        self._add_chip(chip_frame, "Features", f"{len(ALL_FEATURE_NAMES)} body-composition parameters")
        self._add_chip(chip_frame, "Risk", "Optional OS/PFS high-low scoring")

        right = tk.Frame(hero, bg="#102A43")
        right.pack(side=tk.RIGHT, anchor="n")
        tk.Label(
            right,
            text="Research-use local extractor",
            bg="#102A43",
            fg="#F4B942",
            font=("Helvetica", 12, "bold"),
        ).pack(anchor="e")
        tk.Label(
            right,
            text="Risk outputs are exploratory and not treatment recommendations",
            bg="#102A43",
            fg="#C7D4E2",
            font=("Helvetica", 10),
        ).pack(anchor="e", pady=(6, 0))

    def _add_chip(self, parent: tk.Frame, title: str, value: str) -> None:
        chip = tk.Frame(parent, bg="#173F5F", padx=10, pady=6)
        chip.pack(side=tk.LEFT, padx=(0, 8))
        tk.Label(chip, text=title, bg="#173F5F", fg="#F4B942", font=("Helvetica", 9, "bold")).pack(anchor="w")
        tk.Label(chip, text=value, bg="#173F5F", fg="#F8FBFF", font=("Helvetica", 9)).pack(anchor="w")

    def _build_input_tab(self) -> None:
        self.input_tab.columnconfigure(0, weight=3)
        self.input_tab.columnconfigure(1, weight=2)
        self.input_tab.rowconfigure(0, weight=1)

        input_frame = ttk.LabelFrame(self.input_tab, text="NIfTI Inputs", style="Card.TLabelframe", padding=14)
        input_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        input_frame.columnconfigure(1, weight=1)

        ttk.Label(input_frame, text="CT image").grid(row=0, column=0, sticky="w", pady=8)
        ttk.Entry(input_frame, textvariable=self.ct_path_var).grid(row=0, column=1, sticky="ew", pady=8)
        ttk.Button(input_frame, text="Browse", command=self._choose_ct_file, style="Secondary.TButton").grid(
            row=0, column=2, padx=6, pady=8
        )

        ttk.Label(input_frame, text="PET image").grid(row=1, column=0, sticky="w", pady=8)
        ttk.Entry(input_frame, textvariable=self.pet_path_var).grid(row=1, column=1, sticky="ew", pady=8)
        ttk.Button(input_frame, text="Browse", command=self._choose_pet_file, style="Secondary.TButton").grid(
            row=1, column=2, padx=6, pady=8
        )

        ttk.Label(input_frame, textvariable=self.input_preview_var, style="SmallMuted.TLabel", wraplength=740).grid(
            row=2, column=0, columnspan=3, sticky="w", pady=(0, 10)
        )

        ttk.Separator(input_frame).grid(row=3, column=0, columnspan=3, sticky="ew", pady=10)

        ttk.Label(input_frame, text="Existing masks").grid(row=4, column=0, sticky="w", pady=8)
        ttk.Checkbutton(
            input_frame,
            text="Skip segmentation and use existing TotalSegmentator mask folders",
            variable=self.skip_segmentation_var,
            command=self._update_mask_widgets,
        ).grid(row=4, column=1, columnspan=2, sticky="w", pady=8)

        ttk.Label(input_frame, text="Whole-body mask folder").grid(row=5, column=0, sticky="w", pady=8)
        self.mask_total_entry = ttk.Entry(input_frame, textvariable=self.mask_total_var)
        self.mask_total_entry.grid(row=5, column=1, sticky="ew", pady=8)
        self.mask_total_button = ttk.Button(
            input_frame,
            text="Browse",
            command=lambda: self._choose_directory(self.mask_total_var, "Select whole-body mask folder"),
            style="Secondary.TButton",
        )
        self.mask_total_button.grid(row=5, column=2, padx=6, pady=8)

        ttk.Label(input_frame, text="4-tissue mask folder").grid(row=6, column=0, sticky="w", pady=8)
        self.mask_4tissue_entry = ttk.Entry(input_frame, textvariable=self.mask_4tissue_var)
        self.mask_4tissue_entry.grid(row=6, column=1, sticky="ew", pady=8)
        self.mask_4tissue_button = ttk.Button(
            input_frame,
            text="Browse",
            command=lambda: self._choose_directory(self.mask_4tissue_var, "Select 4-tissue mask folder"),
            style="Secondary.TButton",
        )
        self.mask_4tissue_button.grid(row=6, column=2, padx=6, pady=8)

        acquisition_frame = ttk.LabelFrame(self.input_tab, text="Acquisition & Output", style="Card.TLabelframe", padding=14)
        acquisition_frame.grid(row=0, column=1, sticky="nsew")
        acquisition_frame.columnconfigure(1, weight=1)

        ttk.Label(acquisition_frame, text="Body weight (kg)").grid(row=0, column=0, sticky="w", pady=8)
        ttk.Entry(acquisition_frame, textvariable=self.weight_var, width=18).grid(row=0, column=1, sticky="w", pady=8)

        ttk.Label(acquisition_frame, text="Injected activity (MBq)").grid(row=1, column=0, sticky="w", pady=8)
        ttk.Entry(acquisition_frame, textvariable=self.dose_var, width=18).grid(row=1, column=1, sticky="w", pady=8)

        ttk.Label(acquisition_frame, text="Height (cm, optional)").grid(row=2, column=0, sticky="w", pady=8)
        ttk.Entry(acquisition_frame, textvariable=self.height_var, width=18).grid(row=2, column=1, sticky="w", pady=8)
        ttk.Label(
            acquisition_frame,
            text="Use 0 to skip height normalization for torso fat. Height is required when risk scoring is enabled.",
            style="SmallMuted.TLabel",
            wraplength=360,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(0, 8))

        ttk.Checkbutton(
            acquisition_frame,
            text="PET image is already in SUV",
            variable=self.pet_is_suv_var,
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=8)

        ttk.Separator(acquisition_frame).grid(row=5, column=0, columnspan=2, sticky="ew", pady=10)

        ttk.Label(acquisition_frame, text="Output directory").grid(row=6, column=0, sticky="w", pady=8)
        ttk.Entry(acquisition_frame, textvariable=self.output_dir_var).grid(row=6, column=1, sticky="ew", pady=8)
        ttk.Button(
            acquisition_frame,
            text="Browse",
            command=lambda: self._choose_directory(self.output_dir_var, "Select output directory"),
            style="Secondary.TButton",
        ).grid(row=7, column=1, sticky="w", pady=(2, 8))
        ttk.Label(acquisition_frame, textvariable=self.output_preview_var, style="SmallMuted.TLabel", wraplength=360).grid(
            row=8, column=0, columnspan=2, sticky="w", pady=(6, 0)
        )

    def _build_options_tab(self) -> None:
        self.options_tab.columnconfigure(0, weight=1)
        self.options_tab.columnconfigure(1, weight=1)

        engine_frame = ttk.LabelFrame(self.options_tab, text="Execution Device", style="Card.TLabelframe", padding=14)
        engine_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=(0, 10))
        ttk.Checkbutton(engine_frame, text="Use GPU for segmentation when available", variable=self.use_gpu_var).pack(anchor="w")
        ttk.Label(
            engine_frame,
            textvariable=self.device_hint_var,
            style="SmallMuted.TLabel",
            wraplength=520,
        ).pack(anchor="w", pady=(8, 0))
        ttk.Label(
            engine_frame,
            text="CPU mode can be much slower, especially for whole-body CT volumes.",
            style="SmallMuted.TLabel",
            wraplength=520,
        ).pack(anchor="w", pady=(4, 0))

        memory_frame = ttk.LabelFrame(self.options_tab, text="Memory Optimization", style="Card.TLabelframe", padding=14)
        memory_frame.grid(row=0, column=1, sticky="nsew", pady=(0, 10))
        ttk.Checkbutton(
            memory_frame,
            text="Force split (low RAM mode)",
            variable=self.force_split_var,
        ).pack(anchor="w", pady=(0, 8))
        ttk.Checkbutton(
            memory_frame,
            text="Fast mode (3 mm resolution)",
            variable=self.fast_mode_var,
        ).pack(anchor="w")
        ttk.Label(
            memory_frame,
            text="Force split is safer for machines with limited RAM. Fast mode trades some resolution for speed and lower memory use.",
            style="SmallMuted.TLabel",
            wraplength=520,
        ).pack(anchor="w", pady=(8, 0))

        threads_frame = ttk.LabelFrame(self.options_tab, text="Threads", style="Card.TLabelframe", padding=14)
        threads_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        threads_frame.columnconfigure(1, weight=1)
        ttk.Label(threads_frame, text="Resampling threads").grid(row=0, column=0, sticky="w", pady=8)
        ttk.Entry(threads_frame, textvariable=self.nr_thr_resamp_var, width=12).grid(row=0, column=1, sticky="w", pady=8)
        ttk.Label(threads_frame, text="Saving threads").grid(row=1, column=0, sticky="w", pady=8)
        ttk.Entry(threads_frame, textvariable=self.nr_thr_saving_var, width=12).grid(row=1, column=1, sticky="w", pady=8)
        ttk.Label(
            threads_frame,
            text="Use 1 for the most conservative memory footprint.",
            style="SmallMuted.TLabel",
            wraplength=520,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))

        timeout_frame = ttk.LabelFrame(self.options_tab, text="Timeouts", style="Card.TLabelframe", padding=14)
        timeout_frame.grid(row=1, column=1, sticky="nsew")
        timeout_frame.columnconfigure(1, weight=1)
        ttk.Label(timeout_frame, text="Whole-body segmentation (min)").grid(row=0, column=0, sticky="w", pady=8)
        ttk.Entry(timeout_frame, textvariable=self.timeout_total_var, width=12).grid(row=0, column=1, sticky="w", pady=8)
        ttk.Label(timeout_frame, text="4-tissue segmentation (min)").grid(row=1, column=0, sticky="w", pady=8)
        ttk.Entry(timeout_frame, textvariable=self.timeout_tissue_var, width=12).grid(row=1, column=1, sticky="w", pady=8)
        ttk.Label(
            timeout_frame,
            text="Increase these values for large CT volumes or CPU-only runs.",
            style="SmallMuted.TLabel",
            wraplength=520,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))

    def _build_risk_tab(self) -> None:
        self.risk_tab.columnconfigure(0, weight=3)
        self.risk_tab.columnconfigure(1, weight=2)

        input_frame = ttk.LabelFrame(self.risk_tab, text="Research-Only OS/PFS Risk Scoring", style="Card.TLabelframe", padding=14)
        input_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        input_frame.columnconfigure(1, weight=1)

        ttk.Checkbutton(
            input_frame,
            text="Calculate OS/PFS high-low risk groups after feature extraction",
            variable=self.enable_risk_prediction_var,
            command=self._update_risk_widgets,
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

        ttk.Label(input_frame, text="Age").grid(row=1, column=0, sticky="w", pady=8)
        self.risk_age_entry = ttk.Entry(input_frame, textvariable=self.age_var, width=18)
        self.risk_age_entry.grid(row=1, column=1, sticky="w", pady=8)

        mapping = self.risk_spec["clinical_input_mapping"]
        ttk.Label(input_frame, text="Sex").grid(row=2, column=0, sticky="w", pady=8)
        self.risk_sex_combo = ttk.Combobox(
            input_frame,
            textvariable=self.sex_var,
            values=list(mapping["Gender"].keys()),
            state="readonly",
            width=28,
        )
        self.risk_sex_combo.grid(row=2, column=1, sticky="w", pady=8)

        ttk.Label(input_frame, text="Clinical stage").grid(row=3, column=0, sticky="w", pady=8)
        self.risk_stage_combo = ttk.Combobox(
            input_frame,
            textvariable=self.stage_var,
            values=list(mapping["Cli"].keys()),
            state="readonly",
            width=28,
        )
        self.risk_stage_combo.grid(row=3, column=1, sticky="w", pady=8)

        ttk.Label(input_frame, text="Histology").grid(row=4, column=0, sticky="w", pady=8)
        self.risk_histology_combo = ttk.Combobox(
            input_frame,
            textvariable=self.histology_var,
            values=list(mapping["Path"].keys()),
            state="readonly",
            width=28,
        )
        self.risk_histology_combo.grid(row=4, column=1, sticky="w", pady=8)

        ttk.Label(input_frame, text="Tumor SUVmax").grid(row=5, column=0, sticky="w", pady=8)
        self.risk_tumor_suvmax_entry = ttk.Entry(input_frame, textvariable=self.tumor_suvmax_var, width=18)
        self.risk_tumor_suvmax_entry.grid(row=5, column=1, sticky="w", pady=8)
        ttk.Label(
            input_frame,
            text="Leave blank if unavailable; the model will use its frozen development-cohort imputation rule.",
            style="SmallMuted.TLabel",
            wraplength=680,
        ).grid(row=6, column=0, columnspan=2, sticky="w", pady=(0, 8))

        note_frame = ttk.LabelFrame(self.risk_tab, text="Model Notes", style="Card.TLabelframe", padding=14)
        note_frame.grid(row=0, column=1, sticky="nsew")
        ttk.Label(
            note_frame,
            text=(
                "This module uses the frozen reduced Clinical+CT+PET ridge-Cox model specification "
                "from the revised Figure 6 analysis. It combines clinical inputs with extracted "
                "body-composition features and reports OS/PFS risk scores, high-low groups, and "
                "model-estimated event risks at the stored time horizons. High-low groups use the "
                "locked development-cohort median reference."
            ),
            style="SmallMuted.TLabel",
            wraplength=420,
            justify=tk.LEFT,
        ).pack(anchor="w")
        ttk.Label(
            note_frame,
            text=(
                "The output is for research stratification only. It is not a diagnosis, survival "
                "guarantee, treatment recommendation, or substitute for clinical review."
            ),
            style="SmallMuted.TLabel",
            wraplength=420,
            justify=tk.LEFT,
        ).pack(anchor="w", pady=(12, 0))
        ttk.Label(
            note_frame,
            text="Required extracted model features include TAT volume index, so height must be entered when risk scoring is enabled.",
            style="SmallMuted.TLabel",
            wraplength=420,
            justify=tk.LEFT,
        ).pack(anchor="w", pady=(12, 0))

        self.risk_widgets = [
            self.risk_age_entry,
            self.risk_sex_combo,
            self.risk_stage_combo,
            self.risk_histology_combo,
            self.risk_tumor_suvmax_entry,
        ]

    def _build_results_tab(self) -> None:
        self.results_tab.columnconfigure(0, weight=1)
        self.results_tab.rowconfigure(1, weight=1)
        self.results_tab.rowconfigure(2, weight=1)

        header = ttk.LabelFrame(self.results_tab, text="Result Summary", style="Card.TLabelframe", padding=14)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self.result_summary_var = tk.StringVar(value="No completed run yet.")
        ttk.Label(header, textvariable=self.result_summary_var, style="SmallMuted.TLabel", wraplength=1120).pack(anchor="w")

        table_frame = ttk.LabelFrame(self.results_tab, text="Extracted Body-Composition Parameters", style="Card.TLabelframe", padding=10)
        table_frame.grid(row=1, column=0, sticky="nsew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        self.result_tree = ttk.Treeview(table_frame, columns=("feature", "value"), show="headings")
        self.result_tree.heading("feature", text="Feature")
        self.result_tree.heading("value", text="Value")
        self.result_tree.column("feature", width=420, anchor=tk.W)
        self.result_tree.column("value", width=220, anchor=tk.E)
        self.result_tree.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.result_tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.result_tree.configure(yscrollcommand=scrollbar.set)

        risk_frame = ttk.LabelFrame(self.results_tab, text="Research Risk Scores", style="Card.TLabelframe", padding=10)
        risk_frame.grid(row=2, column=0, sticky="nsew", pady=(10, 0))
        risk_frame.rowconfigure(1, weight=1)
        risk_frame.columnconfigure(0, weight=1)
        self.risk_summary_var = tk.StringVar(value="Risk scoring has not been run.")
        ttk.Label(risk_frame, textvariable=self.risk_summary_var, style="SmallMuted.TLabel", wraplength=1120).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )
        self.risk_tree = ttk.Treeview(
            risk_frame,
            columns=("endpoint", "score", "group", "event_risks", "c_index"),
            show="headings",
        )
        self.risk_tree.heading("endpoint", text="Endpoint")
        self.risk_tree.heading("score", text="Risk score")
        self.risk_tree.heading("group", text="Risk group")
        self.risk_tree.heading("event_risks", text="Event risks")
        self.risk_tree.heading("c_index", text="External C-index")
        self.risk_tree.column("endpoint", width=90, anchor=tk.W)
        self.risk_tree.column("score", width=120, anchor=tk.E)
        self.risk_tree.column("group", width=180, anchor=tk.W)
        self.risk_tree.column("event_risks", width=540, anchor=tk.W)
        self.risk_tree.column("c_index", width=120, anchor=tk.E)
        self.risk_tree.grid(row=1, column=0, sticky="nsew")
        risk_scrollbar = ttk.Scrollbar(risk_frame, orient=tk.VERTICAL, command=self.risk_tree.yview)
        risk_scrollbar.grid(row=1, column=1, sticky="ns")
        self.risk_tree.configure(yscrollcommand=risk_scrollbar.set)

    def _build_log_tab(self) -> None:
        ttk.Label(self.log_tab, text="Run log", foreground="#0F6E8C", font=("Helvetica", 10, "bold")).pack(anchor="w")
        ttk.Label(
            self.log_tab,
            text="The pipeline runs in a background thread. Segmentation can take a long time on CPU.",
            style="SmallMuted.TLabel",
            wraplength=1120,
        ).pack(anchor="w", pady=(4, 10))
        self.log_text = ScrolledText(self.log_tab, height=34, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.configure(state=tk.DISABLED)

    def _bind_events(self) -> None:
        self.ct_path_var.trace_add("write", lambda *_args: self._refresh_input_preview())
        self.pet_path_var.trace_add("write", lambda *_args: self._refresh_input_preview())
        self.output_dir_var.trace_add("write", lambda *_args: self._refresh_output_preview())

    def _choose_ct_file(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select CT NIfTI file",
            filetypes=[("NIfTI", "*.nii *.nii.gz"), ("All Files", "*.*")],
        )
        if file_path:
            self.ct_path_var.set(file_path)

    def _choose_pet_file(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select PET NIfTI file",
            filetypes=[("NIfTI", "*.nii *.nii.gz"), ("All Files", "*.*")],
        )
        if file_path:
            self.pet_path_var.set(file_path)

    def _choose_directory(self, variable: tk.StringVar, title: str) -> None:
        folder = filedialog.askdirectory(title=title)
        if folder:
            variable.set(folder)

    def _refresh_input_preview(self) -> None:
        ct = Path(self.ct_path_var.get().strip()) if self.ct_path_var.get().strip() else None
        pet = Path(self.pet_path_var.get().strip()) if self.pet_path_var.get().strip() else None
        pieces = []
        for label, path in (("CT", ct), ("PET", pet)):
            if path is None:
                pieces.append(f"{label}: not selected")
            elif path.exists():
                pieces.append(f"{label}: {path.name}")
            else:
                pieces.append(f"{label}: path not found")
        self.input_preview_var.set("Input preview: " + " | ".join(pieces))

    def _refresh_output_preview(self) -> None:
        raw = self.output_dir_var.get().strip()
        if raw:
            self.output_preview_var.set(f"Each run creates a timestamped folder under {raw}")
        else:
            self.output_preview_var.set("Select an output directory.")

    def _update_mask_widgets(self) -> None:
        state = tk.NORMAL if self.skip_segmentation_var.get() else tk.DISABLED
        self.mask_total_entry.configure(state=state)
        self.mask_total_button.configure(state=state)
        self.mask_4tissue_entry.configure(state=state)
        self.mask_4tissue_button.configure(state=state)

    def _update_risk_widgets(self) -> None:
        if not hasattr(self, "risk_widgets"):
            return
        enabled = self.enable_risk_prediction_var.get()
        for widget in self.risk_widgets:
            if isinstance(widget, ttk.Combobox):
                widget.configure(state="readonly" if enabled else tk.DISABLED)
            else:
                widget.configure(state=tk.NORMAL if enabled else tk.DISABLED)

    def _append_log(self, message: str) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _build_config(self) -> PipelineConfig:
        ct_path = self._validate_file(self.ct_path_var.get(), "CT image")
        pet_path = self._validate_file(self.pet_path_var.get(), "PET image")
        output_raw = self.output_dir_var.get().strip()
        if not output_raw:
            raise ValueError("Select an output directory.")
        output_root = Path(output_raw).expanduser()

        weight_kg = self._parse_float(self.weight_var.get(), "Body weight", minimum=1.0, maximum=300.0)
        dose_mbq = self._parse_float(self.dose_var.get(), "Injected activity", minimum=0.1, maximum=1000.0)
        height_value = self._parse_float(self.height_var.get(), "Height", minimum=0.0, maximum=250.0)
        height_cm = height_value if height_value > 0 else None

        mask_dir_total = None
        mask_dir_4tissue = None
        if self.skip_segmentation_var.get():
            mask_dir_total = self._validate_dir(self.mask_total_var.get(), "Whole-body mask folder")
            mask_dir_4tissue = self._validate_dir(self.mask_4tissue_var.get(), "4-tissue mask folder")
        elif self.use_gpu_var.get() and not self.cuda_available:
            raise ValueError(
                "GPU mode is selected, but CUDA is not available in this Python environment. "
                "Clear the GPU option and run on CPU, or install a CUDA-enabled PyTorch environment."
            )

        nr_thr_resamp = self._parse_int(self.nr_thr_resamp_var.get(), "Resampling threads", minimum=1, maximum=8)
        nr_thr_saving = self._parse_int(self.nr_thr_saving_var.get(), "Saving threads", minimum=1, maximum=8)
        timeout_total_min = self._parse_int(self.timeout_total_var.get(), "Whole-body timeout", minimum=10, maximum=240)
        timeout_tissue_min = self._parse_int(self.timeout_tissue_var.get(), "4-tissue timeout", minimum=5, maximum=240)
        enable_risk_prediction = self.enable_risk_prediction_var.get()
        age = None
        sex_label = None
        clinical_stage_label = None
        histology_label = None
        tumor_suvmax = None
        cutoff_reference = DEFAULT_RISK_CUTOFF_REFERENCE
        if enable_risk_prediction:
            if height_cm is None:
                raise ValueError("Height is required when research risk scoring is enabled.")
            age = self._parse_float(self.age_var.get(), "Age", minimum=18.0, maximum=120.0)
            sex_label = self.sex_var.get().strip()
            clinical_stage_label = self.stage_var.get().strip()
            histology_label = self.histology_var.get().strip()
            tumor_suvmax = self._parse_optional_float(
                self.tumor_suvmax_var.get(),
                "Tumor SUVmax",
                minimum=0.0,
                maximum=500.0,
            )

        return PipelineConfig(
            ct_path=ct_path,
            pet_path=pet_path,
            output_root=output_root,
            weight_kg=weight_kg,
            dose_mbq=dose_mbq,
            height_cm=height_cm,
            pet_is_suv=self.pet_is_suv_var.get(),
            skip_segmentation=self.skip_segmentation_var.get(),
            mask_dir_total=mask_dir_total,
            mask_dir_4tissue=mask_dir_4tissue,
            device="cuda" if self.use_gpu_var.get() else "cpu",
            force_split=self.force_split_var.get(),
            fast_mode=self.fast_mode_var.get(),
            nr_thr_resamp=nr_thr_resamp,
            nr_thr_saving=nr_thr_saving,
            timeout_total_min=timeout_total_min,
            timeout_tissue_min=timeout_tissue_min,
            enable_risk_prediction=enable_risk_prediction,
            age=age,
            sex_label=sex_label,
            clinical_stage_label=clinical_stage_label,
            histology_label=histology_label,
            tumor_suvmax=tumor_suvmax,
            cutoff_reference=cutoff_reference,
        )

    def _validate_file(self, raw_path: str, label: str) -> Path:
        path = Path(raw_path.strip()).expanduser()
        if not raw_path.strip():
            raise ValueError(f"Select a {label}.")
        if not path.is_file():
            raise ValueError(f"{label} was not found: {path}")
        name = path.name.lower()
        if not (name.endswith(".nii") or name.endswith(".nii.gz")):
            raise ValueError(f"{label} must be a NIfTI file (.nii or .nii.gz).")
        return path

    def _validate_dir(self, raw_path: str, label: str) -> Path:
        path = Path(raw_path.strip()).expanduser()
        if not raw_path.strip():
            raise ValueError(f"Select a {label}.")
        if not path.is_dir():
            raise ValueError(f"{label} was not found: {path}")
        return path

    def _parse_float(self, raw_value: str, label: str, minimum: float, maximum: float) -> float:
        try:
            value = float(raw_value.strip())
        except ValueError as exc:
            raise ValueError(f"{label} must be a number.") from exc
        if not minimum <= value <= maximum:
            raise ValueError(f"{label} must be between {minimum:g} and {maximum:g}.")
        return value

    def _parse_int(self, raw_value: str, label: str, minimum: int, maximum: int) -> int:
        try:
            value = int(raw_value.strip())
        except ValueError as exc:
            raise ValueError(f"{label} must be an integer.") from exc
        if not minimum <= value <= maximum:
            raise ValueError(f"{label} must be between {minimum} and {maximum}.")
        return value

    def _parse_optional_float(self, raw_value: str, label: str, minimum: float, maximum: float) -> float | None:
        text = raw_value.strip()
        if not text:
            return None
        return self._parse_float(text, label, minimum=minimum, maximum=maximum)

    def _start_run(self) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Task running", "A pipeline run is already in progress.")
            return

        try:
            config = self._build_config()
        except Exception as exc:
            messagebox.showerror("Configuration error", str(exc))
            return

        self.start_button.configure(state=tk.DISABLED)
        self.open_output_button.configure(state=tk.DISABLED)
        self.status_var.set("Pipeline running...")
        self._clear_results()
        self._append_log("=" * 18 + " New Run Started " + "=" * 18)

        def worker() -> None:
            try:
                result = self._run_pipeline(config)
                self.log_queue.put(("done", result))
            except Exception:
                self.log_queue.put(("error", traceback.format_exc()))

        self.worker = threading.Thread(target=worker, daemon=True)
        self.worker.start()

    def _run_pipeline(self, config: PipelineConfig) -> PipelineResult:
        run_dir = config.output_root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir.mkdir(parents=True, exist_ok=False)
        self._thread_log(f"Run directory: {run_dir}")
        self._thread_log(f"CT: {config.ct_path}")
        self._thread_log(f"PET: {config.pet_path}")
        self._thread_log(f"Segmentation device: {config.device}")

        temp_dir: Path | None = None
        mask_dir_total = config.mask_dir_total
        mask_dir_4tissue = config.mask_dir_4tissue

        if config.skip_segmentation:
            self._thread_log("Segmentation skipped. Using existing mask folders.")
            if mask_dir_total is None or mask_dir_4tissue is None:
                raise ValueError("Existing mask folders were not configured.")
        else:
            temp_dir = Path(tempfile.mkdtemp(prefix="petct_bodycomp_", dir=str(run_dir)))
            self._thread_log("Step 1/3: running TotalSegmentator.")
            mask_total, mask_4tissue, message = run_segmentation_pipeline(
                str(config.ct_path),
                str(temp_dir),
                device=config.device,
                timeout_total=config.timeout_total_min * 60,
                timeout_tissue=config.timeout_tissue_min * 60,
                fast=config.fast_mode,
                force_split=config.force_split,
                nr_thr_resamp=config.nr_thr_resamp,
                nr_thr_saving=config.nr_thr_saving,
            )
            self._thread_log(message)
            if mask_total is None or mask_4tissue is None:
                fallback = self._find_reusable_mask_pair(config.output_root, config.ct_path, run_dir)
                if fallback is None:
                    raise RuntimeError(message)
                self._thread_log("Automatic segmentation failed, but a complete existing mask pair was found for this CT case.")
                self._thread_log(f"Reusing masks from: {fallback.source_dir}")
                mask_dir_total = fallback.mask_dir_total
                mask_dir_4tissue = fallback.mask_dir_4tissue
            else:
                mask_dir_total = run_dir / "seg_total"
                mask_dir_4tissue = run_dir / "seg_4tissue"
                self._replace_dir(Path(mask_total), mask_dir_total)
                self._replace_dir(Path(mask_4tissue), mask_dir_4tissue)

        self._thread_log("Step 2/3: extracting body-composition parameters.")
        features_df = extract_all_features(
            str(config.ct_path),
            str(config.pet_path),
            str(mask_dir_4tissue),
            str(mask_dir_total),
            weight_kg=config.weight_kg,
            dose_mbq=config.dose_mbq,
            height_cm=config.height_cm,
            pet_is_suv=config.pet_is_suv,
        )

        csv_path = run_dir / CSV_FILE_NAME
        features_df[ALL_FEATURE_NAMES].to_csv(csv_path, index=False)
        self._thread_log(f"Saved CSV: {csv_path}")

        risk_df = None
        risk_csv_path = None
        combined_csv_path = None
        if config.enable_risk_prediction:
            self._thread_log("Step 3/3: calculating research-only OS/PFS risk scores.")
            clinical_inputs = build_clinical_inputs(
                age=float(config.age),
                sex_label=str(config.sex_label),
                clinical_stage_label=str(config.clinical_stage_label),
                histology_label=str(config.histology_label),
                tumor_suvmax=config.tumor_suvmax,
                spec=self.risk_spec,
            )
            raw_risk_df = predict_risk(
                features_df=features_df,
                clinical_inputs=clinical_inputs,
                cutoff_reference=config.cutoff_reference,
                spec=self.risk_spec,
            )
            risk_df = raw_risk_df.drop(columns=RISK_EXPORT_DROP_COLUMNS, errors="ignore")
            risk_csv_path = run_dir / RISK_CSV_FILE_NAME
            risk_df.to_csv(risk_csv_path, index=False)
            self._thread_log(f"Saved risk-score CSV: {risk_csv_path}")

            risk_wide = {}
            for _, row in risk_df.iterrows():
                endpoint = str(row["Endpoint"])
                for key, value in row.items():
                    if key == "Endpoint":
                        continue
                    safe_key = str(key).replace(" ", "_").replace("-", "_").replace("/", "_")
                    risk_wide[f"{endpoint}_{safe_key}"] = value
            combined_df = pd.concat(
                [features_df.reset_index(drop=True), pd.DataFrame([risk_wide])],
                axis=1,
            )
            combined_csv_path = run_dir / COMBINED_CSV_FILE_NAME
            combined_df.to_csv(combined_csv_path, index=False)
            self._thread_log(f"Saved combined feature/risk CSV: {combined_csv_path}")
        else:
            self._thread_log("Step 3/3: research risk scoring skipped.")

        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

        return PipelineResult(
            features_df=features_df,
            feature_csv_path=csv_path,
            run_dir=run_dir,
            risk_df=risk_df,
            risk_csv_path=risk_csv_path,
            combined_csv_path=combined_csv_path,
        )

    def _replace_dir(self, source: Path, target: Path) -> None:
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(source), str(target))

    def _case_id_from_ct_path(self, ct_path: Path) -> str:
        name = ct_path.name
        if name.lower().endswith(".nii.gz"):
            name = name[:-7]
        else:
            name = ct_path.stem
        cleaned = re.sub(r"[^a-zA-Z0-9]+", "", name).lower()
        return cleaned

    def _mask_pair_is_complete(self, mask_dir_total: Path, mask_dir_4tissue: Path) -> bool:
        if not mask_dir_total.is_dir() or not mask_dir_4tissue.is_dir():
            return False

        for tissue_name in TISSUE_MASK_MAP:
            if not any((mask_dir_4tissue / f"{tissue_name}{suffix}").is_file() for suffix in (".nii.gz", ".nii")):
                return False

        total_names = [
            path.name.replace(".nii.gz", "").replace(".nii", "").lower()
            for path in mask_dir_total.glob("*.nii*")
        ]
        has_liver = "liver" in total_names
        has_bone = any(any(keyword in name for keyword in BONE_KEYWORDS) for name in total_names)
        return has_liver and has_bone

    def _find_reusable_mask_pair(self, output_root: Path, ct_path: Path, current_run_dir: Path) -> MaskPair | None:
        case_id = self._case_id_from_ct_path(ct_path)
        if not case_id:
            return None

        candidates: list[MaskPair] = []
        for mask_dir_4tissue in output_root.rglob("seg_4tissue"):
            source_dir = mask_dir_4tissue.parent
            try:
                source_dir.relative_to(current_run_dir)
                continue
            except ValueError:
                pass
            source_key = re.sub(r"[^a-zA-Z0-9]+", "", source_dir.name).lower()
            if case_id not in source_key:
                continue
            mask_dir_total = source_dir / "seg_total"
            if self._mask_pair_is_complete(mask_dir_total, mask_dir_4tissue):
                candidates.append(MaskPair(mask_dir_total=mask_dir_total, mask_dir_4tissue=mask_dir_4tissue, source_dir=source_dir))

        if not candidates:
            return None
        return max(candidates, key=lambda item: item.source_dir.stat().st_mtime)

    def _thread_log(self, message: str) -> None:
        self.log_queue.put(("log", message))

    def _drain_log_queue(self) -> None:
        while True:
            try:
                kind, payload = self.log_queue.get_nowait()
            except queue.Empty:
                break

            if kind == "log":
                self._append_log(str(payload))
            elif kind == "done":
                result = payload
                if not isinstance(result, PipelineResult):
                    raise TypeError("Unexpected pipeline result payload.")
                self.result_df = result.features_df
                self.result_csv_path = result.feature_csv_path
                self.result_dir = result.run_dir
                self.risk_df = result.risk_df
                self.risk_csv_path = result.risk_csv_path
                self.combined_csv_path = result.combined_csv_path
                self._populate_results(result)
                self.start_button.configure(state=tk.NORMAL)
                self.open_output_button.configure(state=tk.NORMAL)
                self.status_var.set("Pipeline completed.")
                self._append_log(f"Run completed. Result directory: {result.run_dir}")
                messagebox.showinfo("Completed", f"Pipeline completed.\n\nResult directory:\n{result.run_dir}")
            elif kind == "error":
                self.start_button.configure(state=tk.NORMAL)
                self.status_var.set("Pipeline failed. Check the run log.")
                self._append_log(str(payload))
                messagebox.showerror("Run failed", "The pipeline failed. Check the Run Log tab for details.")

        self.root.after(150, self._drain_log_queue)

    def _clear_results(self) -> None:
        self.result_df = None
        self.result_csv_path = None
        self.result_dir = None
        self.risk_df = None
        self.risk_csv_path = None
        self.combined_csv_path = None
        self.result_summary_var.set("Pipeline running. Results will appear here when extraction completes.")
        self.risk_summary_var.set("Risk scoring has not been run.")
        for item in self.result_tree.get_children():
            self.result_tree.delete(item)
        for item in self.risk_tree.get_children():
            self.risk_tree.delete(item)

    def _populate_results(self, result: PipelineResult) -> None:
        features_df = result.features_df
        self.result_summary_var.set(f"Extracted {len(ALL_FEATURE_NAMES)} parameters. CSV saved to {result.feature_csv_path}")
        for item in self.result_tree.get_children():
            self.result_tree.delete(item)
        row = features_df.iloc[0].to_dict()
        for feature in ALL_FEATURE_NAMES:
            value = row.get(feature, 0.0)
            self.result_tree.insert("", tk.END, values=(feature, f"{float(value):.6g}"))

        for item in self.risk_tree.get_children():
            self.risk_tree.delete(item)
        if result.risk_df is None or result.risk_df.empty:
            self.risk_summary_var.set("Risk scoring was skipped for this run.")
            return

        self.risk_summary_var.set(
            f"Research risk scores saved to {result.risk_csv_path}. Combined CSV saved to {result.combined_csv_path}."
        )
        for _, risk_row in result.risk_df.iterrows():
            event_parts = []
            for column, value in risk_row.items():
                if str(column).endswith("_month_event_risk"):
                    horizon = str(column).replace("_month_event_risk", "")
                    event_parts.append(f"{horizon} mo: {float(value) * 100:.1f}%")
            self.risk_tree.insert(
                "",
                tk.END,
                values=(
                    risk_row["Endpoint"],
                    f"{float(risk_row['Risk score']):.4f}",
                    risk_row["Risk group"],
                    "; ".join(event_parts),
                    f"{float(risk_row['External C-index']):.3f}",
                ),
            )

    def _open_output_folder(self) -> None:
        target = self.result_dir or Path(self.output_dir_var.get().strip()).expanduser()
        if not target.exists():
            messagebox.showinfo("Output folder", "No output folder is available yet.")
            return
        try:
            if subprocess.run(["explorer", str(target)], check=False).returncode != 0:
                messagebox.showinfo("Output folder", str(target))
        except Exception:
            messagebox.showinfo("Output folder", str(target))

    def run(self) -> int:
        self.root.mainloop()
        return 0


def main() -> int:
    app = LauncherGUI()
    return app.run()


if __name__ == "__main__":
    raise SystemExit(main())
