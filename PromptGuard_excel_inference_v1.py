# pg_gui.py
# Modernized Tkinter GUI for Prompt Guard v1/v2 with fine-tuned selection.
# Functionality unchanged; improved layout, styling, and visual hierarchy.
#
# Author: Mikołaj Grajeta

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd
import torch
from torch.nn.functional import softmax
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ========= Core constants =========
MAX_TOKENS_DEFAULT = 512
TEMPERATURE_DEFAULT = 1.0

HUB_V1 = "meta-llama/Prompt-Guard-86M"               # 3 classes: [BENIGN, INJECTION, JAILBREAK]
HUB_V2 = "meta-llama/Llama-Prompt-Guard-2-86M"       # 2 classes: [BENIGN, JAILBREAK]

COL_BENIGN    = "BENIGN"
COL_INDIRECT  = "INJECTION"      # v1 only; NaN for v2
COL_JAILBREAK = "JAILBREAK"
COL_PREDICTED = "PREDICTED_CLASS"

MODELS_ROOT = os.path.abspath("models")

# ========= Data classes =========
@dataclass(frozen=True)
class PGConfig:
    model_id_or_path: str
    device: str = "cpu"
    temperature: float = TEMPERATURE_DEFAULT
    max_tokens: int = MAX_TOKENS_DEFAULT
    debug: bool = False
    debug_preview_chars: int = 200
    show_tokens: bool = False
    checkpoint_interval: int = 100
    overwrite_output: bool = False
    token_count_col: str = "PROMPT_TOKENS"
    cache_dir: Optional[str] = None
    local_files_only: bool = False
    output_suffix: str = "_output"

@dataclass
class PGResult:
    benign: float
    injection: Optional[float]
    jailbreak: float

# ========= Logger =========
class DebugLogger:
    def __init__(self, enabled: bool, preview_chars: int):
        self.enabled = enabled
        self.preview_chars = preview_chars
    def _short(self, s: str) -> str:
        s = (s or "").replace("\n", "\\n")
        n = self.preview_chars
        return s if len(s) <= n else f"{s[:n//2]}…{s[-n//2:]}"
    def log(self, *args):
        if self.enabled: print(*args)
    def header(self, rid: int, prompt: str):
        if not self.enabled: return
        print("\n" + "=" * 80)
        print(f"[PG-DEBUG] ROW_ID={rid} | len={len(prompt)}")
        print(f'[PG-DEBUG] preview: "{self._short(prompt)}"')
        print("=" * 80)
    def selected(self, best_idx: int, b0: float, b1: float, b2: float):
        if not self.enabled: return
        print(f"[PG-DEBUG] SELECTED CHUNK idx={best_idx+1} | "
              f"BENIGN={b0:.6f}, INJECTION={b1:.6f}, JAILBREAK={b2:.6f}")
    
    def show_tokens(self, text: str, tokenizer, max_tokens: int = 512):
        """Display tokenized representation of the text."""
        if not self.enabled: return
        try:
            # Tokenize the text
            tokens = tokenizer(text, add_special_tokens=True, truncation=True, max_length=max_tokens)
            token_ids = tokens['input_ids']
            
            # Convert token IDs back to tokens for display
            token_strings = tokenizer.convert_ids_to_tokens(token_ids)
            
            print(f"[PG-TOKENS] Token count: {len(token_ids)}")
            print(f"[PG-TOKENS] Tokens: {token_strings}")
            print(f"[PG-TOKENS] Token IDs: {token_ids}")
            print("-" * 80)
        except Exception as e:
            print(f"[PG-TOKENS] Error tokenizing text: {e}")

# ========= Model wrapper =========
class PromptGuardModel:
    """Supports v1 (3-class) and v2 (2-class)."""
    def __init__(self, cfg: PGConfig, logger: DebugLogger):
        self.cfg = cfg
        self.log = logger
        self.model, self.tokenizer, self.device, self.num_labels = self._load()
        self.model.eval()
    def _load(self):
        is_local_dir = os.path.isdir(self.cfg.model_id_or_path)
        kw = dict(local_files_only=self.cfg.local_files_only)
        if not is_local_dir and self.cfg.cache_dir:
            os.makedirs(self.cfg.cache_dir, exist_ok=True)
            kw["cache_dir"] = self.cfg.cache_dir
        model = AutoModelForSequenceClassification.from_pretrained(
            self.cfg.model_id_or_path, **kw
        ).to(self.cfg.device)
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_id_or_path, use_fast=True, **kw)
        num_labels = int(getattr(model.config, "num_labels", 3))
        self.log.log(f"[PG-DEBUG] Loaded '{self.cfg.model_id_or_path}' "
                     f"num_labels={num_labels} device={self.cfg.device} "
                     f"source={'local' if is_local_dir else 'hub'}")
        return model, tokenizer, self.cfg.device, num_labels
    def count_tokens(self, text: str) -> int:
        enc = self.tokenizer(text, add_special_tokens=True, truncation=False, return_length=True)
        return int(enc["length"][0])
    def _encode_chunks(self, text: str):
        return self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.cfg.max_tokens,
            return_overflowing_tokens=True,
            padding="max_length",
            add_special_tokens=True,
        )
    def classify_best_chunk(self, text: str):
        enc = self._encode_chunks(text)
        inputs = {k: v.to(self.device) for k, v in enc.items() if k in {"input_ids", "attention_mask"}}
        with torch.inference_mode():
            logits = self.model(**inputs).logits  # [num_chunks, num_labels]
        if self.num_labels == 3:
            best_idx = int(torch.argmax(logits[:, 2]).item())
            best_logits = logits[best_idx]
            probs = softmax(best_logits / self.cfg.temperature, dim=-1).tolist()
            b0, b1, b2 = map(float, probs)
            result = PGResult(benign=b0, injection=b1, jailbreak=b2)
            labels = [COL_BENIGN, COL_INDIRECT, COL_JAILBREAK]
            predicted = labels[int(torch.argmax(best_logits).item())]
            self.log.selected(best_idx, b0, b1, b2)
            return result, predicted, best_idx
        if self.num_labels == 2:
            best_idx = int(torch.argmax(logits[:, 1]).item())
            best_logits = logits[best_idx]
            probs = softmax(best_logits / self.cfg.temperature, dim=-1).tolist()
            b0, b2 = map(float, probs)
            result = PGResult(benign=b0, injection=None, jailbreak=b2)
            labels = [COL_BENIGN, COL_JAILBREAK]
            predicted = labels[int(torch.argmax(best_logits).item())]
            self.log.selected(best_idx, b0, 0.0, b2)
            return result, predicted, best_idx
        # Fallback
        best_idx = int(torch.argmax(logits[:, -1]).item())
        best_logits = logits[best_idx]
        probs = softmax(best_logits / self.cfg.temperature, dim=-1).tolist()
        b0 = float(probs[0]) if len(probs) > 0 else float("nan")
        b1 = float(probs[1]) if len(probs) > 1 else float("nan")
        b2 = float(probs[-1])
        result = PGResult(benign=b0, injection=b1, jailbreak=b2)
        predicted = COL_JAILBREAK if torch.argmax(best_logits).item() == (len(probs) - 1) else COL_BENIGN
        self.log.selected(best_idx, b0, b1 if b1 == b1 else 0.0, b2)
        return result, predicted, best_idx

# ========= Excel store =========
class ExcelStore:
    def __init__(self, input_path: str, prompt_col: str, cfg: PGConfig, logger: DebugLogger):
        self.input_path = input_path
        self.prompt_col = prompt_col
        self.cfg = cfg
        self.log = logger
        base, _ = os.path.splitext(input_path)
        suffix = (self.cfg.output_suffix or "_output").strip()
        if suffix and not suffix.startswith("_"): suffix = "_" + suffix
        self.output_path = f"{base}{suffix}" if suffix.endswith(".xlsx") else f"{base}{suffix}.xlsx"
        self.data = pd.read_excel(self.input_path, engine="openpyxl")
        if self.prompt_col not in self.data.columns:
            raise ValueError(f"Missing column '{self.prompt_col}'")
        self.data = self.data.reset_index().rename(columns={"index": "ROW_ID"})
        self.df = self._prepare_output()
    def _prepare_output(self):
        if os.path.exists(self.output_path) and not self.cfg.overwrite_output:
            existing = pd.read_excel(self.output_path, engine="openpyxl")
            if "ROW_ID" not in existing.columns:
                existing = existing.reset_index().rename(columns={"index": "ROW_ID"})
            merge_cols = ["ROW_ID", COL_BENIGN, COL_INDIRECT, COL_JAILBREAK, COL_PREDICTED, self.cfg.token_count_col]
            have = [c for c in merge_cols if c in existing.columns]
            out = self.data.merge(existing[have], on="ROW_ID", how="left")
        else:
            out = self.data.copy()
        for col in (COL_BENIGN, COL_INDIRECT, COL_JAILBREAK, COL_PREDICTED, self.cfg.token_count_col):
            if col not in out.columns:
                out[col] = pd.NA
        return out.set_index("ROW_ID")
    def iter_rows_fast(self):
        return self.data.set_index("ROW_ID").itertuples(index=True, name="Row")
    def save(self):
        self.df.reset_index().to_excel(self.output_path, index=False, engine="openpyxl")

# ========= Processor =========
class PromptProcessor:
    def __init__(self, cfg: PGConfig, progress_cb=None):
        self.cfg = cfg
        self.log = DebugLogger(cfg.debug, cfg.debug_preview_chars)
        self.model = PromptGuardModel(cfg, self.log)
        self._stop_requested = False
        self._progress_cb = progress_cb or (lambda a, b: None)
        self._show_tokens = cfg.show_tokens
    def request_stop(self): self._stop_requested = True
    def process_file(self, input_path: str, prompt_col: str):
        store = ExcelStore(input_path, prompt_col, self.cfg, self.log)
        done_ids = set(store.df.dropna(subset=[COL_BENIGN, COL_JAILBREAK, COL_PREDICTED]).index)
        total = len(store.data); processed = 0
        for row in tqdm(store.iter_rows_fast(), total=total, desc="Processing prompts"):
            rid = row.Index
            if rid in done_ids:
                processed += 1; self._progress_cb(processed, total); continue
            prompt_val = getattr(row, prompt_col)
            prompt = "" if pd.isna(prompt_val) else str(prompt_val)
            self.log.header(rid, prompt)
            
            # Show tokenized text if enabled
            if self._show_tokens:
                self.log.show_tokens(prompt, self.model.tokenizer, self.cfg.max_tokens)
            
            try:
                tok_count = self.model.count_tokens(prompt)
                store.df.at[rid, self.cfg.token_count_col] = tok_count
                res, predicted, _ = self.model.classify_best_chunk(prompt)
                store.df.at[rid, COL_BENIGN]    = res.benign
                store.df.at[rid, COL_JAILBREAK] = res.jailbreak
                store.df.at[rid, COL_PREDICTED] = predicted
                store.df.at[rid, COL_INDIRECT]  = (res.injection if res.injection is not None else pd.NA)
            except Exception as e:
                print(f"Error processing prompt ROW_ID={rid}: {e}", file=sys.stderr)
            processed += 1
            self._progress_cb(processed, total)
            if processed % self.cfg.checkpoint_interval == 0: store.save()
            if self._stop_requested:
                store.save(); print("[PG] Stop requested: saved current progress and exiting."); return
        store.save(); print(f"Processing complete. Results saved to {store.output_path}")

# ========= GUI =========
class App(tk.Tk):
    # --- Color palette (Meta/LLaMA inspired) ---
    BG      = "#0b132b"  # deep navy
    CARD    = "#151f3b"  # elevated surface
    TEXT    = "#e6eaff"  # high contrast
    MUTED   = "#b7c1ff"  # secondary
    ACCENT  = "#6b63ff"  # llama purple
    ACCENT2 = "#24a0ed"  # blue accent
    BORDER  = "#22305c"
    def __init__(self):
        super().__init__()
        os.makedirs(MODELS_ROOT, exist_ok=True)
        self.title("Prompt Guard Classifier – GUI")
        self.geometry("960x620")
        self.minsize(1000, 560)  # Increased minimum width to prevent text hiding
        self._setup_style()

        # State
        self.input_file = tk.StringVar()
        self.prompt_col = tk.StringVar(value="prompt")
        self.output_suffix = tk.StringVar(value="_output")
        self.debug_enabled = tk.BooleanVar(value=False)
        self.show_tokens = tk.BooleanVar(value=False)
        self.model_mode = tk.StringVar(value="original")  # original | finetuned
        self.original_choice = tk.StringVar(value="v1")   # v1 | v2
        self.finetuned_choice = tk.StringVar(value="")
        self.total_count = tk.IntVar(value=0)
        self.processed_count = tk.IntVar(value=0)
        self.percent_complete = tk.StringVar(value="0%")
        self.eta_var = tk.StringVar(value="--:--")
        self.running = False
        self.worker_thread = None
        self.processor = None

        self._build_ui()
        self._refresh_finetuned_list()

    # ---- Styling ----
    def _setup_style(self):
        self.configure(bg=self.BG)
        style = ttk.Style()
        try: style.theme_use("clam")
        except Exception: pass

        # Frames / cards
        style.configure("Card.TFrame", background=self.CARD)
        style.configure("Root.TFrame", background=self.BG)

        # Labels
        style.configure("Title.TLabel", background=self.BG, foreground=self.TEXT,
                        font=("SF Pro Display", 18, "bold"))
        style.configure("Label.TLabel", background=self.CARD, foreground=self.MUTED,
                        font=("SF Pro Text", 11))
        style.configure("Value.TLabel", background=self.CARD, foreground=self.TEXT,
                        font=("SF Pro Text", 12, "bold"))

        # Entry/Combo
        style.configure("TEntry", fieldbackground="#0f1730", background="#0f1730",
                        foreground=self.TEXT, bordercolor=self.BORDER)
        style.configure("TCombobox", fieldbackground="#0f1730", foreground=self.TEXT, background="#0f1730")
        style.map("TCombobox", fieldbackground=[("readonly", "#0f1730")])

        # Buttons
        style.configure("TButton", padding=8, font=("SF Pro Text", 11, "bold"))
        style.configure("Accent.TButton", background=self.ACCENT, foreground="#ffffff")
        style.map("Accent.TButton",
                  background=[("active", "#7a72ff")], foreground=[("active", "#ffffff")])
        style.configure("Danger.TButton", background="#f15b5b", foreground="#ffffff")
        style.map("Danger.TButton",
                  background=[("active", "#ff6b6b")], foreground=[("active", "#ffffff")])

        # Progressbar
        style.configure("Horizontal.TProgressbar", troughcolor=self.BORDER, background=self.ACCENT2)

    # ---- Layout helpers ----
    def _card(self, parent):
        f = ttk.Frame(parent, style="Card.TFrame")
        f.pack(fill="x", padx=18, pady=10, ipady=10)
        # Configure columns with different weights and minimum widths for better text visibility
        # Label columns (0, 4, 5, 9) get higher priority and minimum widths
        label_columns = [0, 4, 5, 9]
        for c in range(12): 
            if c in label_columns:
                f.grid_columnconfigure(c, weight=0, minsize=80)  # Fixed width for labels
            else:
                f.grid_columnconfigure(c, weight=1, minsize=40)  # Flexible width for content
        return f
    def _label(self, parent, text, row, col, columnspan=1):
        label = ttk.Label(parent, text=text, style="Label.TLabel")
        label.grid(row=row, column=col, columnspan=columnspan, sticky="e", padx=(12,6), pady=6)
        # Set minimum width for labels to prevent text hiding
        label.configure(width=len(text) + 2)
        return label

    # ---- Build UI ----
    def _build_ui(self):
        ttk.Label(self, text="Prompt Guard Classifier", style="Title.TLabel").pack(pady=(16, 6))

        root = ttk.Frame(self, style="Root.TFrame")
        root.pack(fill="both", expand=True)

        # ---- File & columns card ----
        file_card = self._card(root)
        r = 0
        self._label(file_card, "Excel file:", r, 0, columnspan=2)
        e_file = ttk.Entry(file_card, textvariable=self.input_file)
        e_file.grid(row=r, column=2, columnspan=8, sticky="ew", pady=6)
        ttk.Button(file_card, text="Browse…", command=self._choose_file)\
           .grid(row=r, column=10, columnspan=2, sticky="ew", padx=(6,12), pady=6)

        r += 1
        self._label(file_card, "Prompt column:", r, 0, columnspan=2)
        ttk.Entry(file_card, textvariable=self.prompt_col, width=24)\
           .grid(row=r, column=2, columnspan=2, sticky="w", pady=6)
        self._label(file_card, "Output suffix:", r, 4, columnspan=2)
        ttk.Entry(file_card, textvariable=self.output_suffix, width=24)\
           .grid(row=r, column=6, columnspan=2, sticky="w", pady=6)
        debug_cb = ttk.Checkbutton(file_card, text="Enable debug logs in console",
                        variable=self.debug_enabled, command=self._toggle_debug_options)
        debug_cb.grid(row=r, column=9, columnspan=3, sticky="w", pady=6)
        debug_cb.configure(width=30)  # Set minimum width

        r += 1
        # Show tokens button (only visible when debug is enabled)
        self.tokens_cb = ttk.Checkbutton(file_card, text="Show tokenized text",
                        variable=self.show_tokens, state="disabled")
        self.tokens_cb.grid(row=r, column=9, columnspan=3, sticky="w", pady=6)
        self.tokens_cb.configure(width=30)  # Set minimum width

        ttk.Separator(root, orient="horizontal").pack(fill="x", padx=18, pady=2)

        # ---- Model card ----
        model_card = self._card(root)
        r = 0
        self._label(model_card, "Model:", r, 0)
        orig_rb = ttk.Radiobutton(model_card, text="Original (Hub cached in ./models)",
                        variable=self.model_mode, value="original",
                        command=self._toggle_model_mode)
        orig_rb.grid(row=r, column=1, columnspan=5, sticky="w", pady=6)
        orig_rb.configure(width=35)  # Set minimum width
        
        finetuned_rb = ttk.Radiobutton(model_card, text="Fine-tuned (inside ./models)",
                        variable=self.model_mode, value="finetuned",
                        command=self._toggle_model_mode)
        finetuned_rb.grid(row=r, column=6, columnspan=6, sticky="w", pady=6)
        finetuned_rb.configure(width=35)  # Set minimum width

        r += 1
        # Original choices aligned in same row
        self.orig_v1 = ttk.Radiobutton(model_card, text="Prompt Guard v1",
                                       variable=self.original_choice, value="v1")
        self.orig_v1.grid(row=r, column=1, columnspan=3, sticky="w", padx=(0,12), pady=4)
        self.orig_v2 = ttk.Radiobutton(model_card, text="Prompt Guard v2",
                                       variable=self.original_choice, value="v2")
        self.orig_v2.grid(row=r, column=4, columnspan=3, sticky="w", pady=4)

        r += 1
        self._label(model_card, "Fine-tuned dir:", r, 0, columnspan=2)
        self.finetuned_combo = ttk.Combobox(model_card, textvariable=self.finetuned_choice, state="readonly")
        self.finetuned_combo.grid(row=r, column=2, columnspan=6, sticky="ew", pady=6)
        ttk.Button(model_card, text="Refresh", command=self._refresh_finetuned_list)\
           .grid(row=r, column=9, columnspan=2, sticky="ew", padx=(6,12), pady=6)

        ttk.Separator(root, orient="horizontal").pack(fill="x", padx=18, pady=2)

        # ---- Progress & controls card ----
        ctrl_card = self._card(root)
        # Special configuration for progress card with more elements
        progress_label_columns = [0, 2, 5, 9]
        for c in range(12): 
            if c in progress_label_columns:
                ctrl_card.grid_columnconfigure(c, weight=0, minsize=90)  # Larger minimum for progress labels
            else:
                ctrl_card.grid_columnconfigure(c, weight=1, minsize=40)
        r = 0

        # Total prompts
        self._label(ctrl_card, "Total prompts:", r, 0)
        ttk.Label(ctrl_card, textvariable=self.total_count, style="Value.TLabel")\
           .grid(row=r, column=1, sticky="w", pady=6)

        # Processed prompts
        self._label(ctrl_card, "Processed:", r, 2)
        ttk.Label(ctrl_card, textvariable=self.processed_count, style="Value.TLabel")\
           .grid(row=r, column=3, sticky="w", pady=6)

        # Percentage completed
        self._label(ctrl_card, "Completed:", r, 5)
        ttk.Label(ctrl_card, textvariable=self.percent_complete, style="Value.TLabel")\
           .grid(row=r, column=6, sticky="w", pady=6)

        # Graphical donut/circular progress (canvas)
        donut_size = 36
        self.donut_canvas = tk.Canvas(ctrl_card, width=donut_size, height=donut_size, highlightthickness=0, bg=self.CARD)
        self.donut_canvas.grid(row=r, column=8, rowspan=2, padx=(18,0), pady=4)
        self._draw_donut_progress(0)  # initial

        # Estimated time remaining
        self._label(ctrl_card, "ETA:", r, 9)
        ttk.Label(ctrl_card, textvariable=self.eta_var, style="Value.TLabel")\
           .grid(row=r, column=10, sticky="w", pady=6)


        # Progressbar spanning the row
        r += 1
        pbar = ttk.Progressbar(ctrl_card, mode="determinate", style="Horizontal.TProgressbar")
        pbar.grid(row=r, column=0, columnspan=12, sticky="ew", padx=12, pady=(2,10))
        self._pbar = pbar  # keep ref

        # Buttons row (uniform widths)
        r += 1
        self.start_btn = ttk.Button(ctrl_card, text="Start", style="Accent.TButton", command=self._start_run)
        self.start_btn.grid(row=r, column=0, columnspan=2, sticky="ew", padx=(12,6), pady=6)

        self.stop_btn = ttk.Button(ctrl_card, text="Save & Exit", command=self._save_and_exit, state="disabled")
        self.stop_btn.grid(row=r, column=2, columnspan=2, sticky="ew", padx=6, pady=6)

        quit_btn = ttk.Button(ctrl_card, text="Quit", style="Danger.TButton", command=self._quit_app)
        quit_btn.grid(row=r, column=10, columnspan=2, sticky="ew", padx=(6,12), pady=6)

        self._toggle_model_mode()  # initialize enabled fields

        # Keyboard focus
        self.bind("<Return>", lambda _e: self._start_run() if self.start_btn["state"] == "normal" else None)

    # ---- UI actions ----
    def _choose_file(self):
        path = filedialog.askopenfilename(
            title="Select Excel file",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        if path:
            self.input_file.set(path)
            try:
                df = pd.read_excel(path, engine="openpyxl")
                self.total_count.set(len(df)); self.processed_count.set(0)
                self._update_pbar()
            except Exception as e:
                messagebox.showerror("Read error", str(e))

    def _refresh_finetuned_list(self):
        os.makedirs(MODELS_ROOT, exist_ok=True)
        dirs = sorted([d for d in os.listdir(MODELS_ROOT) if os.path.isdir(os.path.join(MODELS_ROOT, d))])
        self.finetuned_combo["values"] = dirs
        if dirs and not self.finetuned_choice.get():
            self.finetuned_choice.set(dirs[0])

    def _toggle_model_mode(self):
        is_orig = (self.model_mode.get() == "original")
        self.orig_v1.configure(state=("normal" if is_orig else "disabled"))
        self.orig_v2.configure(state=("normal" if is_orig else "disabled"))
        self.finetuned_combo.configure(state=("disabled" if is_orig else "readonly"))

    def _toggle_debug_options(self):
        """Enable/disable the Show Tokens button based on debug logs setting."""
        is_debug_enabled = self.debug_enabled.get()
        self.tokens_cb.configure(state=("normal" if is_debug_enabled else "disabled"))
        if not is_debug_enabled:
            self.show_tokens.set(False)  # Uncheck if debug is disabled

    # ---- Threaded processing ----
    def _start_run(self):
        excel = self.input_file.get().strip()
        if not excel:
            messagebox.showwarning("Missing file", "Please select an Excel file."); return
        if not os.path.isfile(excel):
            messagebox.showwarning("Invalid file", "Selected Excel file does not exist."); return
        prompt_col = (self.prompt_col.get().strip() or "prompt")
        out_suffix = (self.output_suffix.get() or "_output").strip()

        try:
            cfg = self._resolve_cfg(self.debug_enabled.get(), self.show_tokens.get(), out_suffix)
        except Exception as e:
            messagebox.showerror("Model error", str(e)); return

        self.running = True
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        try:
            df = pd.read_excel(excel, engine="openpyxl")
            self.total_count.set(len(df))
        except Exception:
            pass
        self.processed_count.set(0); self._update_pbar()

        def on_progress(processed, total):
            self.after(0, lambda: (self._set_progress(processed, total)))

        self.processor = PromptProcessor(cfg, progress_cb=on_progress)

        def worker():
            try:
                self.processor.process_file(excel, prompt_col)
            except Exception as e:
                print(f"[GUI] Fatal error: {e}", file=sys.stderr)
            finally:
                self.after(0, self._on_worker_done)

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()

    def _set_progress(self, processed, total):
        self.total_count.set(total); self.processed_count.set(processed); self._update_pbar()

    def _update_pbar(self):
        total = max(1, self.total_count.get())
        self._pbar["maximum"] = total
        self._pbar["value"] = self.processed_count.get()
        
        # Update percentage and donut progress
        if total > 0:
            percent = (self.processed_count.get() / total) * 100
            self.percent_complete.set(f"{percent:.1f}%")
            self._draw_donut_progress(percent)

    def _draw_donut_progress(self, percent):
        """Draw a circular progress donut on the canvas."""
        self.donut_canvas.delete("all")
        
        # Canvas dimensions
        size = 36
        center = size // 2
        radius = 14
        
        # Background circle (full donut)
        self.donut_canvas.create_oval(
            center - radius, center - radius,
            center + radius, center + radius,
            outline=self.BORDER, width=3, fill=""
        )
        
        # Progress arc
        if percent > 0:
            # Calculate the end angle (0 degrees is at 3 o'clock, we want to start at 12 o'clock)
            start_angle = -90  # Start at 12 o'clock
            extent = (percent / 100) * 360  # Convert percentage to degrees
            
            self.donut_canvas.create_arc(
                center - radius, center - radius,
                center + radius, center + radius,
                start=start_angle, extent=extent,
                outline=self.ACCENT2, width=3, style="arc"
            )

    def _on_worker_done(self):
        self.running = False
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        messagebox.showinfo("Done", "Processing finished (or stopped). See console for details.")

    def _save_and_exit(self):
        if not self.running or not self.processor:
            self.destroy(); return
        if messagebox.askokcancel("Confirm", "Stop after current row, save results, and exit?"):
            self.processor.request_stop()

    def _quit_app(self):
        if self.running and not messagebox.askokcancel("Quit", "Processing is running. Quit anyway?"):
            return
        self.destroy()

    # ---- Model resolver ----
    def _resolve_cfg(self, debug_enabled: bool, show_tokens: bool, output_suffix: str) -> PGConfig:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.model_mode.get() == "original":
            hub_id = HUB_V1 if self.original_choice.get() == "v1" else HUB_V2
            return PGConfig(
                model_id_or_path=hub_id, device=device,
                temperature=TEMPERATURE_DEFAULT, max_tokens=MAX_TOKENS_DEFAULT,
                debug=debug_enabled, show_tokens=show_tokens, checkpoint_interval=100, overwrite_output=False,
                token_count_col="PROMPT_TOKENS", cache_dir=MODELS_ROOT,
                local_files_only=False, output_suffix=output_suffix,
            )
        name = self.finetuned_choice.get().strip()
        if not name: raise ValueError("Choose a fine-tuned directory from ./models")
        local_dir = os.path.join(MODELS_ROOT, name)
        if not os.path.isdir(local_dir): raise FileNotFoundError(f"Fine-tuned model directory not found: {local_dir}")
        return PGConfig(
            model_id_or_path=local_dir, device=device,
            temperature=TEMPERATURE_DEFAULT, max_tokens=MAX_TOKENS_DEFAULT,
            debug=debug_enabled, show_tokens=show_tokens, checkpoint_interval=100, overwrite_output=False,
            token_count_col="PROMPT_TOKENS", cache_dir=None,
            local_files_only=True, output_suffix=output_suffix,
        )

if __name__ == "__main__":
    os.makedirs(MODELS_ROOT, exist_ok=True)
    app = App()
    app.mainloop()
