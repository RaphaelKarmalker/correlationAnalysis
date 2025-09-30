import customtkinter as ctk
from tkinter import filedialog, messagebox
import pandas as pd
import threading
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import sys

# Add logic folder to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'logic'))

from logic.correlation_analyser import AdvancedCorrelationAnalyser
from logic.data_loader import _parse_any_timestamp, merge_trade_features
from logic.converter import apply_feature_transforms
from logic.vis import quick_dashboard, plot_selected_cross_correlations, plot_lag_analysis, plot_correlation_comparison
import matplotlib.pyplot as plt

# Import help text constants
from logic.utils.utils import (
    WORKFLOW_HELP_TEXT, INDICATORS_HELP_TEXT, CONVERSIONS_HELP_TEXT, 
    FILE_FORMATS_HELP_TEXT, ADVANCED_HELP_TEXT, CONVERSION_OPTIONS
)

# Set appearance mode and default color theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class CorrelationAnalysisGUI:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("Correlation Analysis Dashboard")
        self.root.geometry("1400x900")
        
        # Variables
        self.csv_paths = []
        self.target_csv = ctk.StringVar()
        self.target_feature = ctk.StringVar()
        
        # Feature data
        self.available_features = {}  # {csv_path: [feature_names]}
        self.all_features = []  # flattened list of all features
        self.feature_selections = {}  # {feature_name: (selected_var, conversion_var, params_var)}
        
        # NEW: Instrument splitting
        self.has_instruments = False
        self.split_by_instruments = ctk.BooleanVar(value=False)
        # NEW: aggregation method for instrument split
        self.instrument_agg_method = ctk.StringVar(value="mean")
        
        self.create_widgets()
        self.center_window()
        
    def center_window(self):
        """Center the window on screen"""
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (self.root.winfo_width() // 2)
        y = (self.root.winfo_screenheight() // 2) - (self.root.winfo_height() // 2)
        self.root.geometry(f'+{x}+{y}')
        
    def create_widgets(self):
        """Create the main GUI layout"""
        # Main container with padding
        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title with help button
        title_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        title_frame.pack(fill="x", pady=(0, 20))
        
        title_label = ctk.CTkLabel(title_frame, text="Correlation Analysis Dashboard", 
                                  font=ctk.CTkFont(size=24, weight="bold"))
        title_label.pack(side="left")
        
        ctk.CTkButton(title_frame, text="❓ Help", command=self.show_help_window, 
                     width=80, height=30).pack(side="right")
        
        # Create tabview for organized sections
        self.tabview = ctk.CTkTabview(main_frame, width=1300, height=700)
        self.tabview.pack(fill="both", expand=True)
        
        # Add tabs
        self.tabview.add("📁 Data Selection")
        self.tabview.add("🎯 Target & Features")
        self.tabview.add("⚙️ Analysis Settings")
        
        # Create tab contents
        self.create_data_selection_tab()
        self.create_target_features_tab()
        self.create_settings_tab()
        
        # Bottom control panel
        self.create_control_panel(main_frame)
        
    def show_help_window(self):
        """Show help window with usage instructions"""
        help_window = ctk.CTkToplevel(self.root)
        help_window.title("Correlation Analysis Help")
        help_window.geometry("800x600")
        help_window.transient(self.root)
        
        # Create scrollable text widget
        help_text = ctk.CTkTextbox(help_window, width=760, height=550)
        help_text.pack(padx=20, pady=20, fill="both", expand=True)
        
        # Add help content
        help_content = """
CORRELATION ANALYSIS DASHBOARD HELP

=== WORKFLOW ===
1. Data Selection: Add CSV files containing your data
2. Target & Features: Select target variable and features to analyze
3. Analysis Settings: Configure correlation analysis options
4. Run Analysis: Execute correlation analysis and view results

=== DATA REQUIREMENTS ===
- CSV files must have a 'timestamp' column
- Timestamps can be in various formats (ISO, epoch ns/ms, etc.)
- Data will be automatically aligned by timestamp
- Missing values are handled with forward-fill

=== FEATURE CONVERSIONS ===
Available transformations:
- None: Use feature as-is
- log: Natural logarithm
- sqrt: Square root
- normalize: Z-score normalization
- standardize: Same as normalize
- minmax: Min-max scaling (0-1)
- ema: Exponential moving average (requires window parameter)
- sma: Simple moving average (requires window parameter)
- diff: First difference
- pct_change: Percentage change
- lag: Lag the series (requires periods parameter)

=== ANALYSIS TYPES ===
- Pearson: Linear correlation
- Spearman: Rank correlation
- Distance: Distance correlation (captures non-linear relationships)
- Mutual Information: Information-theoretic dependency
- Partial: Correlation controlling for other variables
- Cross-correlation: Time-lagged correlations
- Rolling: Time-varying correlations

=== OUTPUT ===
Results are saved to ./data/correlation_analysis/ with:
- CSV files with correlation matrices and summary
- Overview plots showing key relationships
- Individual feature analysis plots
- Summary statistics and reports
"""
        
        help_text.insert("0.0", help_content)
        help_text.configure(state="disabled")
        
    def create_data_selection_tab(self):
        """Create data selection tab"""
        tab_frame = self.tabview.tab("📁 Data Selection")
        
        # CSV Files section
        csv_frame = ctk.CTkFrame(tab_frame)
        csv_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        csv_title = ctk.CTkLabel(csv_frame, text="CSV Data Files", 
                                font=ctk.CTkFont(size=16, weight="bold"))
        csv_title.pack(anchor="w", padx=15, pady=(15, 5))
        
        csv_desc = ctk.CTkLabel(csv_frame, text="Add CSV files containing your features and target variable:")
        csv_desc.pack(anchor="w", padx=15)
        
        # CSV files scrollable list
        self.csv_textbox = ctk.CTkTextbox(csv_frame, height=200)
        self.csv_textbox.pack(fill="both", expand=True, padx=15, pady=(10, 10))
        
        # Buttons for CSV files
        csv_btn_frame = ctk.CTkFrame(csv_frame, fg_color="transparent")
        csv_btn_frame.pack(fill="x", padx=15, pady=(0, 15))
        
        ctk.CTkButton(csv_btn_frame, text="Add CSV File", 
                     command=self.add_csv_file).pack(side="left", padx=(0, 10))
        ctk.CTkButton(csv_btn_frame, text="Remove Last", 
                     command=self.remove_csv_file).pack(side="left", padx=(0, 10))
        ctk.CTkButton(csv_btn_frame, text="Load Features", 
                     command=self.load_available_features, 
                     fg_color="#1f538d").pack(side="right")
        
    def create_target_features_tab(self):
        """Create target and feature selection tab"""
        tab_frame = self.tabview.tab("🎯 Target & Features")
        
        # Create main container
        main_container = ctk.CTkFrame(tab_frame)
        main_container.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Target selection section
        target_frame = ctk.CTkFrame(main_container)
        target_frame.pack(fill="x", pady=(0, 15))
        
        target_title = ctk.CTkLabel(target_frame, text="Target Variable Selection", 
                                   font=ctk.CTkFont(size=16, weight="bold"))
        target_title.pack(anchor="w", padx=15, pady=(15, 5))
        
        # Target CSV selection
        target_csv_frame = ctk.CTkFrame(target_frame, fg_color="transparent")
        target_csv_frame.pack(fill="x", padx=15, pady=(5, 10))
        
        ctk.CTkLabel(target_csv_frame, text="Target CSV:").pack(side="left", padx=(0, 10))
        self.target_csv_combo = ctk.CTkComboBox(target_csv_frame, variable=self.target_csv,
                                               values=[], width=300, state="readonly",
                                               command=self.on_target_csv_changed)
        self.target_csv_combo.pack(side="left", padx=(0, 10))
        
        # Target feature selection
        target_feature_frame = ctk.CTkFrame(target_frame, fg_color="transparent")
        target_feature_frame.pack(fill="x", padx=15, pady=(0, 10))
        
        ctk.CTkLabel(target_feature_frame, text="Target Feature:").pack(side="left", padx=(0, 10))
        self.target_feature_combo = ctk.CTkComboBox(target_feature_frame, variable=self.target_feature,
                                                   values=[], width=300, state="readonly")
        self.target_feature_combo.pack(side="left")
        
        # NEW: Instrument splitting option
        instrument_frame = ctk.CTkFrame(target_frame, fg_color="transparent")
        instrument_frame.pack(fill="x", padx=15, pady=(0, 15))
        
        self.instrument_checkbox = ctk.CTkCheckBox(
            instrument_frame, 
            text="Split by instruments (combine multiple instrument_id)",
            variable=self.split_by_instruments
        )
        # NEW: aggregation selector (hidden unless instruments exist)
        self.instrument_agg_combo = ctk.CTkComboBox(
            instrument_frame,
            variable=self.instrument_agg_method,
            values=["mean", "sum", "median"],
            state="readonly",
            width=120
        )
        # Initially hidden, will be shown if target CSV has instrument_id
        
        # Feature selection section
        self.feature_frame = ctk.CTkScrollableFrame(main_container, label_text="Feature Selection & Conversion")
        self.feature_frame.pack(fill="both", expand=True)
        
        # Initial message
        self.no_features_label = ctk.CTkLabel(self.feature_frame, 
                                             text="Please load CSV files first in the Data Selection tab",
                                             font=ctk.CTkFont(size=14, slant="italic"))
        self.no_features_label.pack(pady=50)
        
    def create_settings_tab(self):
        """Create analysis settings tab"""
        tab_frame = self.tabview.tab("⚙️ Analysis Settings")
        
        # Analysis settings
        settings_frame = ctk.CTkFrame(tab_frame)
        settings_frame.pack(fill="x", padx=20, pady=(20, 15))
        
        settings_title = ctk.CTkLabel(settings_frame, text="Correlation Analysis Configuration", 
                                     font=ctk.CTkFont(size=16, weight="bold"))
        settings_title.pack(anchor="w", padx=15, pady=(15, 10))
        
        # Analysis options
        options_frame = ctk.CTkFrame(settings_frame, fg_color="transparent")
        options_frame.pack(fill="x", padx=15, pady=(0, 15))
        
        # Checkboxes for analysis types
        self.enable_lag_analysis = ctk.BooleanVar(value=True)
        self.enable_rolling_corr = ctk.BooleanVar(value=True)
        self.enable_partial_corr = ctk.BooleanVar(value=False)
        
        ctk.CTkCheckBox(options_frame, text="Enable lag analysis", 
                       variable=self.enable_lag_analysis).pack(anchor="w", pady=2)
        ctk.CTkCheckBox(options_frame, text="Enable rolling correlation", 
                       variable=self.enable_rolling_corr).pack(anchor="w", pady=2)
        ctk.CTkCheckBox(options_frame, text="Enable partial correlation", 
                       variable=self.enable_partial_corr).pack(anchor="w", pady=2)
        
        # Output settings
        output_frame = ctk.CTkFrame(tab_frame)
        output_frame.pack(fill="x", padx=20)
        
        output_title = ctk.CTkLabel(output_frame, text="Output Configuration", 
                                   font=ctk.CTkFont(size=16, weight="bold"))
        output_title.pack(anchor="w", padx=15, pady=(15, 5))
        
        ctk.CTkLabel(output_frame, text="Save directory: ./data/correlation_analysis").pack(anchor="w", padx=15)
        ctk.CTkLabel(output_frame, text="(Results will include correlation matrices, plots, and summary tables)", 
                    font=ctk.CTkFont(size=12, slant="italic")).pack(anchor="w", padx=15, pady=(5, 15))
        
    def create_control_panel(self, parent):
        """Create bottom control panel"""
        control_frame = ctk.CTkFrame(parent)
        control_frame.pack(fill="x", pady=(20, 0))
        
        # Progress bar
        self.progress = ctk.CTkProgressBar(control_frame, mode="indeterminate")
        self.progress.pack(fill="x", padx=15, pady=(15, 10))
        
        # Status label
        self.status_label = ctk.CTkLabel(control_frame, text="Ready to start analysis")
        self.status_label.pack(pady=(0, 10))
        
        # Buttons
        btn_frame = ctk.CTkFrame(control_frame, fg_color="transparent")
        btn_frame.pack(pady=(0, 15))
        
        ctk.CTkButton(btn_frame, text="Start Analysis", command=self.start_analysis, 
                     fg_color="#2fa572", hover_color="#106a43", width=140, height=40,
                     font=ctk.CTkFont(size=14, weight="bold")).pack(side="left", padx=(0, 10))
        ctk.CTkButton(btn_frame, text="Reset All", command=self.reset_all, 
                     width=100, height=40).pack(side="left")
        
    def add_csv_file(self):
        """Add CSV file"""
        filename = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename and filename not in self.csv_paths:
            self.csv_paths.append(filename)
            self.update_csv_list_display()
            
    def remove_csv_file(self):
        """Remove last CSV file"""
        if self.csv_paths:
            self.csv_paths.pop()
            self.update_csv_list_display()
            
    def update_csv_list_display(self):
        """Update the CSV files display"""
        self.csv_textbox.delete("0.0", "end")
        for i, path in enumerate(self.csv_paths, 1):
            self.csv_textbox.insert("end", f"{i}. {os.path.basename(path)}\n")
            
    def load_available_features(self):
        """Load available features from selected CSV files"""
        if not self.csv_paths:
            messagebox.showwarning("No Files", "Please add at least one CSV file first.")
            return
            
        try:
            self.status_label.configure(text="Loading features...")
            self.progress.start()
            
            # Load features from each CSV
            self.available_features = {}
            all_features = set()
            
            for path in self.csv_paths:
                df = pd.read_csv(path)
                # Exclude timestamp column
                features = [col for col in df.columns if col.lower() not in ['timestamp', 'date', 'time']]
                self.available_features[path] = features
                all_features.update(features)
            
            self.all_features = sorted(list(all_features))
            
            # Update target CSV dropdown
            csv_names = [os.path.basename(path) for path in self.csv_paths]
            self.target_csv_combo.configure(values=csv_names)
            
            self.create_feature_selection_ui()
            
            self.progress.stop()
            self.status_label.configure(text=f"Loaded {len(self.all_features)} available features from {len(self.csv_paths)} CSV files")
            
        except Exception as e:
            self.progress.stop()
            self.status_label.configure(text="Error loading features")
            messagebox.showerror("Error", f"Failed to load features: {str(e)}")
            
    def on_target_csv_changed(self, selection):
        """Handle target CSV selection change"""
        if not selection:
            return
            
        # Find the full path for the selected CSV
        selected_path = None
        for path in self.csv_paths:
            if os.path.basename(path) == selection:
                selected_path = path
                break
                
        if selected_path and selected_path in self.available_features:
            # Update target feature dropdown
            features = self.available_features[selected_path]
            self.target_feature_combo.configure(values=features)
            self.target_feature.set("")
            
            # NEW: Check if target CSV has instrument_id column
            try:
                df = pd.read_csv(selected_path)
                self.has_instruments = 'instrument_id' in df.columns
                self.update_instrument_checkbox_visibility()
            except Exception:
                self.has_instruments = False
                self.update_instrument_checkbox_visibility()
                
    def update_instrument_checkbox_visibility(self):
        """Show/hide instrument splitting checkbox based on target CSV"""
        # This will be called after the target selection UI is created
        if hasattr(self, 'instrument_checkbox'):
            if self.has_instruments:
                self.instrument_checkbox.pack(anchor="w", pady=2)
                # Show aggregation selector
                try:
                    # simple inline layout
                    ctk.CTkLabel(self.instrument_checkbox.master, text="Aggregation:").pack(side="left", padx=(10, 6))
                except Exception:
                    pass
                self.instrument_agg_combo.pack(side="left", padx=(0, 10))
            else:
                self.instrument_checkbox.pack_forget()
                self.split_by_instruments.set(False)
                try:
                    self.instrument_agg_combo.pack_forget()
                except Exception:
                    pass
                
    def create_feature_selection_ui(self):
        """Create dynamic feature selection interface"""
        # Clear existing widgets
        for widget in self.feature_frame.winfo_children():
            widget.destroy()
            
        self.feature_selections = {}
        
        if self.all_features:
            # Select all/none buttons
            btn_frame = ctk.CTkFrame(self.feature_frame, fg_color="transparent")
            btn_frame.pack(fill="x", padx=15, pady=(0, 10))
            
            ctk.CTkButton(btn_frame, text="Select All", 
                         command=lambda: self.toggle_all_features(True), 
                         width=100).pack(side="left", padx=(0, 10))
            ctk.CTkButton(btn_frame, text="Select None", 
                         command=lambda: self.toggle_all_features(False), 
                         width=100).pack(side="left")
            
            # Create grid for features
            self.create_feature_grid(self.feature_frame, self.all_features, 
                                   self.feature_selections, CONVERSION_OPTIONS)
        else:
            ctk.CTkLabel(self.feature_frame, text="No features available. Please load CSV files first.", 
                        font=ctk.CTkFont(size=14)).pack(pady=50)
                        
    # Helper methods (same as trade_analyser_gui.py)
    def _conversion_requires_params(self, name: str) -> bool:
        if not name or name == "None":
            return False
        n = str(name).strip().lower()
        if n in {
            "ema", "sma", "ewm", "lag", "lead", "shift",
            "resample", "quantile", "winsorize", "diff", "pct_change", "percent_change",
        }:
            return True
        if n.startswith("rolling") or "rolling" in n:
            return True
        return False

    def _on_conversion_changed(self, choice: str, params_entry: ctk.CTkEntry, params_var: ctk.StringVar | None = None):
        needs = self._conversion_requires_params(choice)
        if needs:
            try:
                params_entry.grid()
            except Exception:
                pass
            params_entry.configure(state="normal")
            params_entry.configure(placeholder_text="window=20")
        else:
            params_entry.configure(state="disabled")
            try:
                params_entry.delete(0, "end")
            except Exception:
                pass
            if params_var is not None:
                params_var.set("")
            try:
                params_entry.grid_remove()
            except Exception:
                pass

    def create_feature_grid(self, parent, features, selections_dict, conversion_options):
        """Create a grid of feature checkboxes and conversion dropdowns"""
        grid_frame = ctk.CTkFrame(parent)
        grid_frame.pack(fill="x", padx=15, pady=(0, 15))
        
        # Header
        header_frame = ctk.CTkFrame(grid_frame, fg_color=("#3B8ED0", "#1F6AA5"))
        header_frame.pack(fill="x", padx=5, pady=5)
        header_frame.grid_columnconfigure(0, weight=0, minsize=80)
        header_frame.grid_columnconfigure(1, weight=2, minsize=200)
        header_frame.grid_columnconfigure(2, weight=1, minsize=150)
        header_frame.grid_columnconfigure(3, weight=1, minsize=120)
        
        ctk.CTkLabel(header_frame, text="Select", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, padx=10, pady=8, sticky="w")
        ctk.CTkLabel(header_frame, text="Feature", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=1, padx=10, pady=8, sticky="w")
        ctk.CTkLabel(header_frame, text="Conversion", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=2, padx=10, pady=8, sticky="w")
        ctk.CTkLabel(header_frame, text="Parameters", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=3, padx=10, pady=8, sticky="w")
        
        # Features
        for i, feature in enumerate(features):
            selected_var = ctk.BooleanVar()
            conversion_var = ctk.StringVar(value="None")
            params_var = ctk.StringVar(value="")
            selections_dict[feature] = (selected_var, conversion_var, params_var)
            
            row_frame = ctk.CTkFrame(grid_frame, fg_color=("gray90", "gray13"))
            row_frame.pack(fill="x", padx=5, pady=1)
            row_frame.grid_columnconfigure(0, weight=0, minsize=80)
            row_frame.grid_columnconfigure(1, weight=2, minsize=200)
            row_frame.grid_columnconfigure(2, weight=1, minsize=150)
            row_frame.grid_columnconfigure(3, weight=1, minsize=120)
            
            def on_enter(event, frame=row_frame):
                frame.configure(fg_color=("#E1E8ED", "#212121"))
            def on_leave(event, frame=row_frame):
                frame.configure(fg_color=("gray90", "gray13"))
                
            row_frame.bind("<Enter>", on_enter)
            row_frame.bind("<Leave>", on_leave)
            
            checkbox = ctk.CTkCheckBox(row_frame, text="", variable=selected_var, width=20)
            checkbox.grid(row=0, column=0, padx=10, pady=8, sticky="w")
            checkbox.bind("<Enter>", on_enter)
            checkbox.bind("<Leave>", on_leave)
            
            feature_label = ctk.CTkLabel(row_frame, text=feature, anchor="w")
            feature_label.grid(row=0, column=1, padx=10, pady=8, sticky="ew")
            feature_label.bind("<Enter>", on_enter)
            feature_label.bind("<Leave>", on_leave)
            
            params_entry = ctk.CTkEntry(
                row_frame, textvariable=params_var, width=110,
                placeholder_text="window=20"
            )
            params_entry.grid(row=0, column=3, padx=10, pady=8, sticky="w")

            combo = ctk.CTkComboBox(
                row_frame,
                variable=conversion_var,
                values=conversion_options,
                width=140,
                state="readonly",
                command=lambda choice, entry=params_entry, pvar=params_var: self._on_conversion_changed(choice, entry, pvar)
            )
            combo.grid(row=0, column=2, padx=10, pady=8, sticky="w")

            self._on_conversion_changed(conversion_var.get(), params_entry, params_var)

    def toggle_all_features(self, select: bool):
        """Toggle all feature selections"""
        for selected_var, _, _ in self.feature_selections.values():
            selected_var.set(select)
            
    def validate_inputs(self) -> bool:
        """Validate user inputs before starting analysis"""
        if not self.csv_paths:
            messagebox.showerror("Missing Input", "Please add at least one CSV file.")
            return False
            
        if not self.target_csv.get() or not self.target_feature.get():
            messagebox.showerror("Missing Target", "Please select target CSV and target feature.")
            return False
            
        # Check if any features are selected
        selected = any(selected.get() for selected, _, _ in self.feature_selections.values())
        if not selected:
            messagebox.showerror("No Features", "Please select at least one feature to analyze.")
            return False
            
        return True
        
    def start_analysis(self):
        """Start the correlation analysis in a separate thread"""
        if not self.validate_inputs():
            return
            
        self.progress.start()
        self.status_label.configure(text="Running correlation analysis...")
        
        # Run analysis in separate thread
        analysis_thread = threading.Thread(target=self.run_analysis_thread)
        analysis_thread.daemon = True
        analysis_thread.start()
        
    def run_analysis_thread(self):
        """Run the analysis in a separate thread"""
        try:
            # Get selected target and features
            target_csv_name = self.target_csv.get()
            target_feature_name = self.target_feature.get()
            
            # Find target CSV path
            target_csv_path = None
            for path in self.csv_paths:
                if os.path.basename(path) == target_csv_name:
                    target_csv_path = path
                    break
                    
            if not target_csv_path:
                raise ValueError("Target CSV not found")
                
            # Get selected features and transformations
            selected_features = []
            feature_transforms = {}
            
            def parse_params(params_str: str) -> dict:
                params = {}
                if params_str.strip():
                    try:
                        for param in params_str.split(','):
                            if '=' in param:
                                key, value = param.strip().split('=', 1)
                                try:
                                    if '.' in value:
                                        params[key.strip()] = float(value.strip())
                                    else:
                                        params[key.strip()] = int(value.strip())
                                except ValueError:
                                    params[key.strip()] = value.strip()
                    except Exception:
                        pass
                return params
            
            for feature, (selected_var, conversion_var, params_var) in self.feature_selections.items():
                if selected_var.get():
                    selected_features.append(feature)
                    conversion = conversion_var.get()
                    if conversion != "None":
                        params = parse_params(params_var.get())
                        if params:
                            feature_transforms[feature] = {"mode": conversion, **params}
                        else:
                            feature_transforms[feature] = conversion
            
            # Load and merge data
            merged_df = self.load_and_merge_data(target_csv_path, target_feature_name, selected_features)
            
            # Apply feature transforms
            if feature_transforms:
                merged_df = apply_feature_transforms(
                    merged_df,
                    feature_transforms,
                    inplace=True,
                    default_output='replace'
                )
            
            # Run correlation analysis
            self.run_correlation_analysis(merged_df, target_feature_name, selected_features)
            
            # Update GUI on main thread
            self.root.after(0, self.analysis_completed_success)
            
        except Exception as e:
            # Capture the error message immediately to avoid closure issues
            error_msg = str(e)
            # Update GUI on main thread with error
            self.root.after(0, lambda msg=error_msg: self.analysis_completed_error(msg))
            
    def load_and_merge_data(self, target_csv_path: str, target_feature: str, selected_features: List[str]) -> pd.DataFrame:
        """Load and merge data from multiple CSVs with timestamp alignment"""
        # NEW: Check if we should split by instruments
        if self.split_by_instruments.get() and self.has_instruments:
            return self.load_and_merge_data_by_instruments(target_csv_path, target_feature, selected_features)
        
        # Original single-dataset logic
        return self.load_and_merge_data_single(target_csv_path, target_feature, selected_features)
    
    def load_and_merge_data_single(self, target_csv_path: str, target_feature: str, selected_features: List[str]) -> pd.DataFrame:
        """Original single dataset loading logic"""
        # Load target data
        target_df = pd.read_csv(target_csv_path)
        if 'timestamp' not in target_df.columns:
            raise ValueError(f"Target CSV must have 'timestamp' column")
        if target_feature not in target_df.columns:
            raise ValueError(f"Target feature '{target_feature}' not found in target CSV")
            
        # Parse timestamp and set as index
        target_df['timestamp'] = _parse_any_timestamp(target_df['timestamp'])
        target_df = target_df.dropna(subset=['timestamp', target_feature])
        target_df = target_df.set_index('timestamp').sort_index()
        
        # Start with target column
        result_df = target_df[[target_feature]].copy()
        
        # Add features from same CSV if available
        for feature in selected_features:
            if feature in target_df.columns and feature != target_feature:
                result_df[feature] = target_df[feature]
        
        # Load and merge features from other CSVs
        for csv_path in self.csv_paths:
            if csv_path == target_csv_path:
                continue
                
            # Check if any selected features are in this CSV
            csv_df = pd.read_csv(csv_path)
            if 'timestamp' not in csv_df.columns:
                continue
                
            features_in_csv = [f for f in selected_features if f in csv_df.columns]
            if not features_in_csv:
                continue
                
            # Parse timestamp and prepare for merge
            csv_df['timestamp'] = _parse_any_timestamp(csv_df['timestamp'])
            csv_df = csv_df.dropna(subset=['timestamp'])
            csv_df = csv_df.set_index('timestamp').sort_index()
            
            # Forward fill to propagate values
            csv_df = csv_df.ffill()
            
            # Select only needed features
            feature_df = csv_df[features_in_csv].copy()
            
            # Merge using asof (backward looking, no leakage)
            result_reset = result_df.reset_index()
            feature_reset = feature_df.reset_index()
            
            merged_reset = pd.merge_asof(
                result_reset.sort_values('timestamp'),
                feature_reset.sort_values('timestamp'),
                left_on='timestamp',
                right_on='timestamp',
                direction='backward',
                allow_exact_matches=True
            )
            
            result_df = merged_reset.set_index('timestamp')
        
        # Sort by timestamp and ensure unique datetime index
        result_df = result_df.sort_index()
        result_df = result_df[~result_df.index.duplicated(keep='last')]
        return result_df
        
    def load_and_merge_data_by_instruments(self, target_csv_path: str, target_feature: str, selected_features: List[str]) -> pd.DataFrame:
        """Load and merge data split by instruments, then combine results"""
        # Load target data
        target_df = pd.read_csv(target_csv_path)
        if 'timestamp' not in target_df.columns:
            raise ValueError(f"Target CSV must have 'timestamp' column")
        if target_feature not in target_df.columns:
            raise ValueError(f"Target feature '{target_feature}' not found in target CSV")
        if 'instrument_id' not in target_df.columns:
            raise ValueError(f"Target CSV must have 'instrument_id' column for instrument splitting")
            
        # Parse timestamp
        target_df['timestamp'] = _parse_any_timestamp(target_df['timestamp'])
        target_df = target_df.dropna(subset=['timestamp', target_feature])
        # Ensure per-instrument timestamps are unique after we split
        target_df = target_df.sort_values(['instrument_id', 'timestamp'])

        instruments = target_df['instrument_id'].unique()
        print(f"Found {len(instruments)} instruments: {instruments}")

        combined_dfs = []

        for instrument in instruments:
            print(f"Processing instrument: {instrument}")
            instrument_target = target_df[target_df['instrument_id'] == instrument].copy()
            instrument_target = instrument_target.set_index('timestamp').sort_index()
            # drop duplicate timestamps within instrument
            instrument_target = instrument_target[~instrument_target.index.duplicated(keep='last')]

            result_df = instrument_target[[target_feature]].copy()

            # Add features from same CSV if available (excluding instrument_id)
            for feature in selected_features:
                if feature in instrument_target.columns and feature != target_feature and feature != 'instrument_id':
                    result_df[feature] = instrument_target[feature]

            # Load and merge features from other CSVs (same logic as single dataset)
            for csv_path in self.csv_paths:
                if csv_path == target_csv_path:
                    continue

                csv_df = pd.read_csv(csv_path)
                if 'timestamp' not in csv_df.columns:
                    continue

                # If instrument_id exists in this CSV, filter by current instrument
                if 'instrument_id' in csv_df.columns:
                    csv_df = csv_df[csv_df['instrument_id'] == instrument]

                features_in_csv = [f for f in selected_features if f in csv_df.columns]
                if not features_in_csv:
                    continue

                csv_df['timestamp'] = _parse_any_timestamp(csv_df['timestamp'])
                csv_df = csv_df.dropna(subset=['timestamp'])
                csv_df = csv_df.set_index('timestamp').sort_index()
                # drop duplicates per instrument in this CSV
                csv_df = csv_df[~csv_df.index.duplicated(keep='last')]

                csv_df = csv_df.ffill()

                feature_df = csv_df[features_in_csv].copy()

                result_reset = result_df.reset_index()
                feature_reset = feature_df.reset_index()

                merged_reset = pd.merge_asof(
                    result_reset.sort_values('timestamp'),
                    feature_reset.sort_values('timestamp'),
                    left_on='timestamp',
                    right_on='timestamp',
                    direction='backward',
                    allow_exact_matches=True
                )

                result_df = merged_reset.set_index('timestamp')

            combined_dfs.append(result_df)

        # Combine all instruments into one DataFrame and aggregate per timestamp
        if not combined_dfs:
            raise ValueError("No instrument data assembled.")

        final_df = pd.concat(combined_dfs, ignore_index=False).sort_index()

        # Aggregate across instruments per timestamp to avoid duplicate indices
        agg = self.instrument_agg_method.get().lower().strip()
        if agg == "sum":
            final_df = final_df.groupby(level=0).sum()
        elif agg == "median":
            final_df = final_df.groupby(level=0).median()
        else:
            # default mean
            final_df = final_df.groupby(level=0).mean()

        final_df = final_df.sort_index()
        return final_df
        
    def run_correlation_analysis(self, df: pd.DataFrame, target_col: str, feature_cols: List[str]):
        """Run correlation analysis and save results"""
        # Prepare structured output directory
        output_dir = Path("./data/correlation_analysis")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subfolders for organized output
        overview_dir = output_dir / "overview"
        csv_dir = output_dir / "csv_results"
        
        overview_dir.mkdir(exist_ok=True)
        csv_dir.mkdir(exist_ok=True)
        
        # Initialize correlation analyser with proper parameters
        analyser = AdvancedCorrelationAnalyser(
            max_lag=30,
            rolling_window=50,
            compute_distance=True,
            compute_partial=self.enable_partial_corr.get(),
            compute_cross=self.enable_lag_analysis.get(),
            compute_rolling=self.enable_rolling_corr.get(),
            verbose=True
        )
        
        # Prepare features and target
        features_df = df[feature_cols].copy()
        target_series = df[target_col].copy()
        
        # Run analysis
        results = analyser.analyse(features_df, target_series)
        
        # Save all CSV results to csv_results folder
        results.save_results(str(csv_dir))
        
        # Create and save plots
        plot_count = 0
        
        try:
            # Overview dashboard plots - save to overview folder
            dashboard_figs = quick_dashboard(results, top_features=15)
            for i, fig in enumerate(dashboard_figs):
                if fig:
                    fig.savefig(overview_dir / f"dashboard_{i+1}.png", dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    plot_count += 1
        except Exception as e:
            print(f"Error creating dashboard plots: {e}")
        
        try:
            # Correlation comparison - save to overview folder
            comp_fig = plot_correlation_comparison(results)
            if comp_fig:
                comp_fig.savefig(overview_dir / "correlation_comparison.png", dpi=150, bbox_inches='tight')
                plt.close(comp_fig)
                plot_count += 1
        except Exception as e:
            print(f"Error creating comparison plot: {e}")
        
        try:
            # General lag analysis - save to overview folder
            if hasattr(results, 'cross_correlation_curves') and results.cross_correlation_curves:
                lag_fig = plot_lag_analysis(results)
                if lag_fig:
                    lag_fig.savefig(overview_dir / "lag_analysis_overview.png", dpi=150, bbox_inches='tight')
                    plt.close(lag_fig)
                    plot_count += 1
        except Exception as e:
            print(f"Error creating lag analysis plot: {e}")
        
        # Feature-specific analysis - create individual folders for each feature
        try:
            top_features = results.summary.nlargest(min(len(feature_cols), 20), 'pearson')['feature'].tolist()
            
            for feature in top_features:
                # Create feature-specific folder
                feature_name_safe = "".join(c for c in feature if c.isalnum() or c in (' ', '_')).strip().replace(' ', '_')
                feature_dir = output_dir / "features" / feature_name_safe
                feature_dir.mkdir(parents=True, exist_ok=True)
                
                # Individual cross-correlation plot for this feature
                if results.cross_correlation_curves and feature in results.cross_correlation_curves:
                    curve = results.cross_correlation_curves[feature]
                    if curve is not None and not curve.empty:
                        try:
                            from logic.vis import plot_cross_correlation
                            fig = plot_cross_correlation(curve, feature)
                            if fig:
                                fig.savefig(feature_dir / f"{feature_name_safe}_cross_correlation.png", dpi=150, bbox_inches='tight')
                                plt.close(fig)
                                plot_count += 1
                        except Exception as e:
                            print(f"Could not create cross-correlation plot for {feature}: {e}")
                
                # Individual rolling correlation plot for this feature
                if results.rolling_correlations and feature in results.rolling_correlations:
                    rolling_series = results.rolling_correlations[feature]
                    if rolling_series is not None and not rolling_series.empty:
                        try:
                            fig, ax = plt.subplots(figsize=(10, 4))
                            rolling_series.plot(ax=ax)
                            ax.set_title(f"Rolling Correlation: {feature}")
                            ax.axhline(0, color="black", linewidth=0.8)
                            ax.set_ylabel("Correlation")
                            ax.grid(True, alpha=0.3)
                            plt.tight_layout()
                            fig.savefig(feature_dir / f"{feature_name_safe}_rolling_correlation.png", dpi=150, bbox_inches='tight')
                            plt.close(fig)
                            plot_count += 1
                        except Exception as e:
                            print(f"Could not create rolling correlation plot for {feature}: {e}")
                
                # Feature summary info
                try:
                    feature_info = results.summary[results.summary['feature'] == feature].iloc[0]

                    def _fmt(x):
                        try:
                            if pd.isna(x):
                                return "N/A"
                        except Exception:
                            pass
                        return f"{x:.4f}" if isinstance(x, (int, float)) else str(x)

                    lines = [
                        f"Feature: {feature}",
                        f"Pearson: {_fmt(feature_info.get('pearson'))}",
                        f"Spearman: {_fmt(feature_info.get('spearman'))}",
                    ]
                    if 'distance_corr' in results.summary.columns:
                        lines.append(f"Distance Corr: {_fmt(feature_info.get('distance_corr'))}")
                    if 'mutual_info_norm' in results.summary.columns:
                        lines.append(f"Mutual Info (norm): {_fmt(feature_info.get('mutual_info_norm'))}")
                    if 'partial_corr' in results.summary.columns:
                        lines.append(f"Partial Corr: {_fmt(feature_info.get('partial_corr'))}")
                    if 'best_lag' in results.summary.columns:
                        lines.append(f"Best Lag: {feature_info.get('best_lag')}")
                    if 'best_cross_corr' in results.summary.columns:
                        lines.append(f"Cross-Corr @ Best Lag: {_fmt(feature_info.get('best_cross_corr'))}")

                    summary_text = "\n".join(lines) + "\n"
                    with open(feature_dir / f"{feature_name_safe}_summary.txt", "w", encoding="utf-8") as f:
                        f.write(summary_text)
                except Exception as e:
                    print(f"Could not write summary for {feature}: {e}")
        except Exception as e:
            print(f"Error creating feature-specific outputs: {e}")
        
        self.status_label.configure(text=f"Analysis completed with {plot_count} plots saved")
        self.progress.stop()
        
    def analysis_completed_success(self):
        """Handle successful analysis completion"""
        self.status_label.configure(text="Analysis completed successfully")
        self.progress.stop()
        
    def analysis_completed_error(self, error_msg: str):
        """Handle analysis completion with error"""
        self.status_label.configure(text="Analysis failed")
        self.progress.stop()
        messagebox.showerror("Analysis Error", f"An error occurred during analysis:\n\n{error_msg}")
        
    def reset_all(self):
        """Reset all inputs and selections"""
        self.csv_paths = []
        self.target_csv.set("")
        self.target_feature.set("")
        self.available_features = {}
        self.all_features = []
        self.feature_selections = {}
        self.has_instruments = False
        self.split_by_instruments.set(False)
        
        self.update_csv_list_display()
        self.create_feature_selection_ui()
        self.status_label.configure(text="Ready to start analysis")
        
    def run(self):
        """Run the main GUI loop"""
        self.root.mainloop()

if __name__ == "__main__":
    app = CorrelationAnalysisGUI()
    app.run()
