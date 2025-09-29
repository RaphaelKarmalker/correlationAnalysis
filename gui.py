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

from logic.trade_analyser import run_trade_analysis
from logic.data_loader import load_features
from logic.chart_analyser import ChartAnalyser

# Import help text constants
from logic.utils.utils import (
    WORKFLOW_HELP_TEXT, INDICATORS_HELP_TEXT, CONVERSIONS_HELP_TEXT, 
    FILE_FORMATS_HELP_TEXT, ADVANCED_HELP_TEXT, CONVERSION_OPTIONS
)

# Set appearance mode and default color theme
ctk.set_appearance_mode("dark")  # Modes: "System", "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue", "green", "dark-blue"

class TradeAnalysisGUI:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("Trade Performance Analysis Dashboard")
        self.root.geometry("1400x900")
        
        # Variables
        self.trades_path = ctk.StringVar()
        self.ohlcv_path = ctk.StringVar()
        self.feature_paths = []
        self.group_by = ctk.StringVar(value="amount")
        self.num_bins = ctk.IntVar(value=10)
        
        # Feature data
        self.available_features = []
        self.available_chart_indicators = []
        self.feature_selections = {}  # {feature_name: (selected_var, conversion_var, params_var)}
        self.chart_selections = {}    # {indicator_name: (selected_var, conversion_var, params_var)}
        
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
        
        title_label = ctk.CTkLabel(title_frame, text="Trade Performance Analysis Dashboard", 
                                  font=ctk.CTkFont(size=24, weight="bold"))
        title_label.pack(side="left")
        
        ctk.CTkButton(title_frame, text="❓ Help", command=self.show_help_window, 
                     width=80, height=30).pack(side="right")
        
        # Create tabview for organized sections
        self.tabview = ctk.CTkTabview(main_frame, width=1300, height=700)
        self.tabview.pack(fill="both", expand=True)
        
        # Add tabs
        self.tabview.add("📁 File Selection")
        self.tabview.add("⚙️ Feature Selection")
        self.tabview.add("📊 Analysis Settings")
        
        # Create tab contents
        self.create_file_selection_tab()
        self.create_feature_selection_tab()
        self.create_settings_tab()
        
        # Bottom control panel
        self.create_control_panel(main_frame)
        
    def create_file_selection_tab(self):
        """Create file selection tab"""
        tab_frame = self.tabview.tab("📁 File Selection")
        
        # Trades CSV (Mandatory)
        trades_frame = ctk.CTkFrame(tab_frame)
        trades_frame.pack(fill="x", padx=20, pady=(20, 15))
        
        trades_title = ctk.CTkLabel(trades_frame, text="Trades CSV (Required)", 
                                   font=ctk.CTkFont(size=16, weight="bold"))
        trades_title.pack(anchor="w", padx=15, pady=(15, 5))
        
        trades_desc = ctk.CTkLabel(trades_frame, text="Select your trades CSV file:")
        trades_desc.pack(anchor="w", padx=15)
        
        trades_input_frame = ctk.CTkFrame(trades_frame, fg_color="transparent")
        trades_input_frame.pack(fill="x", padx=15, pady=(5, 15))
        
        self.trades_entry = ctk.CTkEntry(trades_input_frame, textvariable=self.trades_path, 
                                        placeholder_text="No file selected", state="readonly")
        self.trades_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        
        ctk.CTkButton(trades_input_frame, text="Browse", command=self.select_trades_file, 
                     width=100).pack(side="right")
        
        # OHLCV CSV (Optional)
        ohlcv_frame = ctk.CTkFrame(tab_frame)
        ohlcv_frame.pack(fill="x", padx=20, pady=(0, 15))
        
        ohlcv_title = ctk.CTkLabel(ohlcv_frame, text="OHLCV CSV (Optional - for chart indicators)", 
                                  font=ctk.CTkFont(size=16, weight="bold"))
        ohlcv_title.pack(anchor="w", padx=15, pady=(15, 5))
        
        ohlcv_desc = ctk.CTkLabel(ohlcv_frame, text="Select OHLCV data for chart indicator analysis:")
        ohlcv_desc.pack(anchor="w", padx=15)
        
        ohlcv_input_frame = ctk.CTkFrame(ohlcv_frame, fg_color="transparent")
        ohlcv_input_frame.pack(fill="x", padx=15, pady=(5, 15))
        
        self.ohlcv_entry = ctk.CTkEntry(ohlcv_input_frame, textvariable=self.ohlcv_path, 
                                       placeholder_text="No file selected", state="readonly")
        self.ohlcv_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        
        ctk.CTkButton(ohlcv_input_frame, text="Browse", command=self.select_ohlcv_file, 
                     width=100).pack(side="right")
        
        # Feature CSVs
        features_frame = ctk.CTkFrame(tab_frame)
        features_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        
        features_title = ctk.CTkLabel(features_frame, text="Feature CSV Files", 
                                     font=ctk.CTkFont(size=16, weight="bold"))
        features_title.pack(anchor="w", padx=15, pady=(15, 5))
        
        features_desc = ctk.CTkLabel(features_frame, text="Add feature CSV files:")
        features_desc.pack(anchor="w", padx=15)
        
        # Feature files scrollable list
        self.feature_textbox = ctk.CTkTextbox(features_frame, height=150)
        self.feature_textbox.pack(fill="both", expand=True, padx=15, pady=(10, 10))
        
        # Buttons for feature files
        feature_btn_frame = ctk.CTkFrame(features_frame, fg_color="transparent")
        feature_btn_frame.pack(fill="x", padx=15, pady=(0, 15))
        
        ctk.CTkButton(feature_btn_frame, text="Add Feature CSV", 
                     command=self.add_feature_file).pack(side="left", padx=(0, 10))
        ctk.CTkButton(feature_btn_frame, text="Remove Selected", 
                     command=self.remove_feature_file).pack(side="left", padx=(0, 10))
        ctk.CTkButton(feature_btn_frame, text="Load Features", 
                     command=self.load_available_features, 
                     fg_color="#1f538d").pack(side="right")
        
    def create_feature_selection_tab(self):
        """Create feature selection and conversion tab"""
        tab_frame = self.tabview.tab("⚙️ Feature Selection")
        
        # Create scrollable frame for feature selection
        self.scrollable_frame = ctk.CTkScrollableFrame(tab_frame, label_text="Feature Selection & Conversion")
        self.scrollable_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Initial message
        self.no_features_label = ctk.CTkLabel(self.scrollable_frame, 
                                             text="Please load feature files first in the File Selection tab",
                                             font=ctk.CTkFont(size=14, slant="italic"))
        self.no_features_label.pack(pady=50)
        
    def create_settings_tab(self):
        """Create analysis settings tab"""
        tab_frame = self.tabview.tab("📊 Analysis Settings")
        
        # Binning settings
        binning_frame = ctk.CTkFrame(tab_frame)
        binning_frame.pack(fill="x", padx=20, pady=(20, 15))
        
        binning_title = ctk.CTkLabel(binning_frame, text="Binning Configuration", 
                                    font=ctk.CTkFont(size=16, weight="bold"))
        binning_title.pack(anchor="w", padx=15, pady=(15, 10))
        
        # Number of bins
        bins_frame = ctk.CTkFrame(binning_frame, fg_color="transparent")
        bins_frame.pack(fill="x", padx=15, pady=(0, 10))
        
        ctk.CTkLabel(bins_frame, text="Number of bins:").pack(side="left")
        self.bins_spinbox = ctk.CTkEntry(bins_frame, textvariable=self.num_bins, width=80)
        self.bins_spinbox.pack(side="left", padx=(10, 0))
        
        # Group by method
        method_frame = ctk.CTkFrame(binning_frame, fg_color="transparent")
        method_frame.pack(fill="x", padx=15, pady=(0, 15))
        
        ctk.CTkLabel(method_frame, text="Binning method:").pack(anchor="w", pady=(10, 5))
        
        self.radio_var = ctk.StringVar(value="amount")
        ctk.CTkRadioButton(method_frame, text="Equal width (size)", 
                          variable=self.radio_var, value="size").pack(anchor="w", pady=2)
        ctk.CTkRadioButton(method_frame, text="Equal count (amount)", 
                          variable=self.radio_var, value="amount").pack(anchor="w", pady=2)
        
        # Output settings
        output_frame = ctk.CTkFrame(tab_frame)
        output_frame.pack(fill="x", padx=20)
        
        output_title = ctk.CTkLabel(output_frame, text="Output Configuration", 
                                   font=ctk.CTkFont(size=16, weight="bold"))
        output_title.pack(anchor="w", padx=15, pady=(15, 5))
        
        ctk.CTkLabel(output_frame, text="Save directory: ./data/trade_analysis").pack(anchor="w", padx=15)
        ctk.CTkLabel(output_frame, text="(Results will be organized by feature in subfolders)", 
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
        
    def select_trades_file(self):
        """Select trades CSV file"""
        filename = filedialog.askopenfilename(
            title="Select Trades CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.trades_path.set(filename)
            
    def select_ohlcv_file(self):
        """Select OHLCV CSV file"""
        filename = filedialog.askopenfilename(
            title="Select OHLCV CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.ohlcv_path.set(filename)
            self.load_chart_indicators()
            
    def add_feature_file(self):
        """Add feature CSV file"""
        filename = filedialog.askopenfilename(
            title="Select Feature CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename and filename not in self.feature_paths:
            self.feature_paths.append(filename)
            self.update_feature_list_display()
            
    def remove_feature_file(self):
        """Remove selected feature file"""
        if self.feature_paths:
            # For simplicity, remove the last added file
            self.feature_paths.pop()
            self.update_feature_list_display()
            
    def update_feature_list_display(self):
        """Update the feature files display"""
        self.feature_textbox.delete("0.0", "end")
        for i, path in enumerate(self.feature_paths, 1):
            self.feature_textbox.insert("end", f"{i}. {os.path.basename(path)}\n")
            
    def load_available_features(self):
        """Load available features from selected CSV files"""
        if not self.feature_paths:
            messagebox.showwarning("No Files", "Please add at least one feature CSV file first.")
            return
            
        try:
            self.status_label.configure(text="Loading features...")
            self.progress.start()
            
            # Load all features to get available columns
            all_features = set()
            for path in self.feature_paths:
                df = pd.read_csv(path)
                # Exclude timestamp column
                features = [col for col in df.columns if col.lower() not in ['timestamp', 'date', 'time']]
                all_features.update(features)
            
            self.available_features = sorted(list(all_features))
            self.create_feature_selection_ui()
            
            self.progress.stop()
            self.status_label.configure(text=f"Loaded {len(self.available_features)} available features")
            
        except Exception as e:
            self.progress.stop()
            self.status_label.configure(text="Error loading features")
            messagebox.showerror("Error", f"Failed to load features: {str(e)}")
            
    def load_chart_indicators(self):
        """Load available chart indicators from OHLCV"""
        if not self.ohlcv_path.get():
            return
            
        try:
            ca = ChartAnalyser(self.ohlcv_path.get()).load()
            indicators_df = ca.compute_all()
            # Get all columns except timestamp
            self.available_chart_indicators = [col for col in indicators_df.columns if col != 'timestamp']
            
            # Refresh feature selection UI if it exists
            if hasattr(self, 'feature_selections') and self.feature_selections:
                self.create_feature_selection_ui()
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load chart indicators: {str(e)}")
            
    def create_feature_selection_ui(self):
        """Create dynamic feature selection interface"""
        # Clear existing widgets
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
            
        self.feature_selections = {}
        self.chart_selections = {}
        
        # CSV Features section
        if self.available_features:
            csv_frame = ctk.CTkFrame(self.scrollable_frame)
            csv_frame.pack(fill="x", pady=(0, 15))
            
            csv_title = ctk.CTkLabel(csv_frame, text="CSV Features", 
                                    font=ctk.CTkFont(size=16, weight="bold"))
            csv_title.pack(anchor="w", padx=15, pady=(15, 5))
            
            # Select all/none buttons
            btn_frame = ctk.CTkFrame(csv_frame, fg_color="transparent")
            btn_frame.pack(fill="x", padx=15, pady=(0, 10))
            
            ctk.CTkButton(btn_frame, text="Select All", 
                         command=lambda: self.toggle_all_features(True), 
                         width=100).pack(side="left", padx=(0, 10))
            ctk.CTkButton(btn_frame, text="Select None", 
                         command=lambda: self.toggle_all_features(False), 
                         width=100).pack(side="left")
            
            # Create grid for features
            self.create_feature_grid(csv_frame, self.available_features, 
                                   self.feature_selections, CONVERSION_OPTIONS)
            
        # Chart indicators section
        if self.available_chart_indicators:
            chart_frame = ctk.CTkFrame(self.scrollable_frame)
            chart_frame.pack(fill="x", pady=(0, 15))
            
            chart_title = ctk.CTkLabel(chart_frame, text="Chart Indicators", 
                                      font=ctk.CTkFont(size=16, weight="bold"))
            chart_title.pack(anchor="w", padx=15, pady=(15, 5))
            
            # Select all/none buttons for chart indicators
            btn_frame = ctk.CTkFrame(chart_frame, fg_color="transparent")
            btn_frame.pack(fill="x", padx=15, pady=(0, 10))
            
            ctk.CTkButton(btn_frame, text="Select All", 
                         command=lambda: self.toggle_all_chart_features(True), 
                         width=100).pack(side="left", padx=(0, 10))
            ctk.CTkButton(btn_frame, text="Select None", 
                         command=lambda: self.toggle_all_chart_features(False), 
                         width=100).pack(side="left")
            
            # Create grid for chart indicators
            self.create_feature_grid(chart_frame, self.available_chart_indicators, 
                                   self.chart_selections, CONVERSION_OPTIONS)
            
        if not self.available_features and not self.available_chart_indicators:
            ctk.CTkLabel(self.scrollable_frame, text="No features available. Please load feature files first.", 
                        font=ctk.CTkFont(size=14)).pack(pady=50)
                     
    # Helper: determine if a conversion needs parameters
    def _conversion_requires_params(self, name: str) -> bool:
        if not name or name == "None":
            return False
        n = str(name).strip().lower()
        # Common parametric conversions; adjust as needed to match your utils.CONVERSION_OPTIONS
        if n in {
            "ema", "sma", "ewm", "lag", "lead", "shift",
            "resample", "quantile", "winsorize", "diff", "pct_change", "percent_change",
        }:
            return True
        # Rolling-family generally needs a window/size param
        if n.startswith("rolling") or "rolling" in n:
            return True
        # Some custom converters may be parameterless by design
        return False

    # Callback to toggle params entry visibility/state when conversion changes
    def _on_conversion_changed(self, choice: str, params_entry: ctk.CTkEntry, params_var: ctk.StringVar | None = None):
        needs = self._conversion_requires_params(choice)
        if needs:
            # Show and enable params field
            try:
                params_entry.grid()
            except Exception:
                pass
            params_entry.configure(state="normal")
            params_entry.configure(placeholder_text="window=20")
        else:
            # Hide and clear params field
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
        """Create a grid of feature checkboxes and conversion dropdowns with parameter fields"""
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
            # Checkbox for selection
            selected_var = ctk.BooleanVar()
            conversion_var = ctk.StringVar(value="None")
            params_var = ctk.StringVar(value="")
            selections_dict[feature] = (selected_var, conversion_var, params_var)
            
            # Row frame with hover effect
            row_frame = ctk.CTkFrame(grid_frame, fg_color=("gray90", "gray13"))
            row_frame.pack(fill="x", padx=5, pady=1)
            row_frame.grid_columnconfigure(0, weight=0, minsize=80)
            row_frame.grid_columnconfigure(1, weight=2, minsize=200)
            row_frame.grid_columnconfigure(2, weight=1, minsize=150)
            row_frame.grid_columnconfigure(3, weight=1, minsize=120)
            
            # Add hover effect
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
            
            # Feature name label
            feature_label = ctk.CTkLabel(row_frame, text=feature, anchor="w")
            feature_label.grid(row=0, column=1, padx=10, pady=8, sticky="ew")
            feature_label.bind("<Enter>", on_enter)
            feature_label.bind("<Leave>", on_leave)
            
            # Parameters entry (created/gridded, but will be hidden if not needed)
            params_entry = ctk.CTkEntry(
                row_frame, textvariable=params_var, width=110,
                placeholder_text="window=20"
            )
            params_entry.grid(row=0, column=3, padx=10, pady=8, sticky="w")

            # Conversion dropdown with handler to show/hide params entry
            combo = ctk.CTkComboBox(
                row_frame,
                variable=conversion_var,
                values=conversion_options,
                width=140,
                state="readonly",
                command=lambda choice, entry=params_entry, pvar=params_var: self._on_conversion_changed(choice, entry, pvar)
            )
            combo.grid(row=0, column=2, padx=10, pady=8, sticky="w")

            # Initialize visibility based on default conversion ("None")
            self._on_conversion_changed(conversion_var.get(), params_entry, params_var)

    def toggle_all_features(self, select: bool):
        """Toggle all CSV feature selections"""
        for selected_var, _, _ in self.feature_selections.values():  # UPDATED: 3-tuple now
            selected_var.set(select)
            
    def toggle_all_chart_features(self, select: bool):
        """Toggle all chart indicator selections"""
        for selected_var, _, _ in self.chart_selections.values():  # UPDATED: 3-tuple now
            selected_var.set(select)
            
    def validate_inputs(self) -> bool:
        """Validate user inputs before starting analysis"""
        if not self.trades_path.get():
            messagebox.showerror("Missing Input", "Please select a trades CSV file.")
            return False
            
        if not self.feature_paths and not self.ohlcv_path.get():
            messagebox.showerror("Missing Input", "Please select at least one feature CSV or OHLCV file.")
            return False
            
        # Check if any features are selected
        csv_selected = any(selected.get() for selected, _, _ in self.feature_selections.values())  # UPDATED: 3-tuple
        chart_selected = any(selected.get() for selected, _, _ in self.chart_selections.values()) if self.chart_selections else False  # UPDATED: 3-tuple
        
        if not csv_selected and not chart_selected:
            messagebox.showerror("No Features", "Please select at least one feature to analyze.")
            return False
            
        return True
        
    def get_selected_features_and_transforms(self):
        """Get selected features and their transformations"""
        selected_features = []
        feature_transforms = {}
        
        # Helper to parse parameters string
        def parse_params(params_str: str) -> dict:
            params = {}
            if params_str.strip():
                try:
                    for param in params_str.split(','):
                        if '=' in param:
                            key, value = param.strip().split('=', 1)
                            # Try to convert to int, float, or keep as string
                            try:
                                if '.' in value:
                                    params[key.strip()] = float(value.strip())
                                else:
                                    params[key.strip()] = int(value.strip())
                            except ValueError:
                                params[key.strip()] = value.strip()
                except Exception:
                    pass  # Ignore malformed parameters
            return params
        
        # CSV features
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
                    
        # Chart indicators - ADD SELECTED ONES TO features LIST
        for indicator, (selected_var, conversion_var, params_var) in self.chart_selections.items():
            if selected_var.get():
                selected_features.append(indicator)  # ADD to main features list
                conversion = conversion_var.get()
                if conversion != "None":
                    params = parse_params(params_var.get())
                    if params:
                        feature_transforms[indicator] = {"mode": conversion, **params}
                    else:
                        feature_transforms[indicator] = conversion
        
        return selected_features, feature_transforms
        
    def start_analysis(self):
        """Start the trade analysis in a separate thread"""
        if not self.validate_inputs():
            return
            
        self.progress.start()
        self.status_label.configure(text="Running trade analysis...")
        
        # Update group_by from radio button
        self.group_by.set(self.radio_var.get())
        
        # Run analysis in separate thread to prevent GUI freezing
        analysis_thread = threading.Thread(target=self.run_analysis_thread)
        analysis_thread.daemon = True
        analysis_thread.start()
        
    def run_analysis_thread(self):
        """Run the analysis in a separate thread"""
        try:
            selected_features, feature_transforms = self.get_selected_features_and_transforms()
            
            # Prepare arguments
            kwargs = {
                'feature_csv_paths': self.feature_paths,
                'features': selected_features,
                'trades_csv_path': self.trades_path.get(),
                'num_bins': self.num_bins.get(),
                'group_by': self.group_by.get(),
                'save_dir': "./data/trade_analysis",
                'verbose': True,
                'feature_transforms': feature_transforms if feature_transforms else None
            }
            
            # Add OHLCV path if provided
            if self.ohlcv_path.get():
                kwargs['ohlcv_csv_path'] = self.ohlcv_path.get()
            
            # Run the analysis
            run_trade_analysis(**kwargs)
            
            # Update GUI on main thread
            self.root.after(0, self.analysis_completed_success)
            
        except Exception as e:
            # Update GUI on main thread with error
            self.root.after(0, lambda: self.analysis_completed_error(str(e)))
            
    def analysis_completed_success(self):
        """Called when analysis completes successfully"""
        self.progress.stop()
        self.status_label.configure(text="Analysis completed successfully!")
        messagebox.showinfo("Success", 
                           "Trade analysis completed successfully!\n\n"
                           "Results saved to: ./data/trade_analysis\n"
                           "Check the output folder for CSV files and plots.")
        
    def analysis_completed_error(self, error_msg: str):
        """Called when analysis encounters an error"""
        self.progress.stop()
        self.status_label.configure(text="Analysis failed")
        messagebox.showerror("Analysis Error", f"Analysis failed with error:\n\n{error_msg}")
        
    def reset_all(self):
        """Reset all inputs and selections"""
        # Clear file paths
        self.trades_path.set("")
        self.ohlcv_path.set("")
        self.feature_paths.clear()
        
        # Reset settings
        self.num_bins.set(10)
        self.radio_var.set("amount")
        
        # Clear feature selections
        self.available_features.clear()
        self.available_chart_indicators.clear()
        self.feature_selections.clear()
        self.chart_selections.clear()
        
        # Update displays
        self.update_feature_list_display()
        self.create_initial_feature_tab()
        
        # Reset status
        self.progress.stop()
        self.status_label.configure(text="Ready to start analysis")
        
    def create_initial_feature_tab(self):
        """Reset feature selection tab to initial state"""
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
            
        self.no_features_label = ctk.CTkLabel(self.scrollable_frame, 
                                             text="Please load feature files first in the File Selection tab",
                                             font=ctk.CTkFont(size=14, slant="italic"))
        self.no_features_label.pack(pady=50)
        
    def show_help_window(self):
        """Show help window with usage instructions"""
        help_window = ctk.CTkToplevel(self.root)
        help_window.title("Help - Trade Analysis Dashboard")
        help_window.geometry("900x700")
        help_window.resizable(True, True)
        
        # Center the help window
        help_window.update_idletasks()
        x = (help_window.winfo_screenwidth() // 2) - (450)
        y = (help_window.winfo_screenheight() // 2) - (350)
        help_window.geometry(f'+{x}+{y}')
        
        # Make it stay on top initially
        help_window.transient(self.root)
        help_window.focus()
        
        # Main frame
        main_frame = ctk.CTkFrame(help_window)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        title_label = ctk.CTkLabel(main_frame, text="Trade Analysis Dashboard - Help Guide", 
                                  font=ctk.CTkFont(size=20, weight="bold"))
        title_label.pack(pady=(0, 20))
        
        # Create tabview for help sections
        help_tabview = ctk.CTkTabview(main_frame, width=800, height=550)
        help_tabview.pack(fill="both", expand=True)
        
        # Add help tabs
        help_tabview.add("📋 Workflow")
        help_tabview.add("📊 Indicators")
        help_tabview.add("🔧 Conversions")
        help_tabview.add("📁 File Formats")
        help_tabview.add("⚙️ Advanced")
        
        # Create help content for each tab
        self.create_workflow_help(help_tabview.tab("📋 Workflow"))
        self.create_indicators_help(help_tabview.tab("📊 Indicators"))
        self.create_conversions_help(help_tabview.tab("🔧 Conversions"))
        self.create_file_formats_help(help_tabview.tab("📁 File Formats"))
        self.create_advanced_help(help_tabview.tab("⚙️ Advanced"))
        
        # Close button
        close_btn = ctk.CTkButton(main_frame, text="Close", command=help_window.destroy, 
                                 width=100, height=35)
        close_btn.pack(pady=(15, 0))
        
    def create_workflow_help(self, parent):
        """Create workflow help content"""
        scrollable = ctk.CTkScrollableFrame(parent)
        scrollable.pack(fill="both", expand=True, padx=10, pady=10)
        
        text_widget = ctk.CTkTextbox(scrollable, height=400, wrap="word")
        text_widget.pack(fill="both", expand=True)
        text_widget.insert("0.0", WORKFLOW_HELP_TEXT)
        text_widget.configure(state="disabled")
        
    def create_indicators_help(self, parent):
        """Create indicators help content"""
        scrollable = ctk.CTkScrollableFrame(parent)
        scrollable.pack(fill="both", expand=True, padx=10, pady=10)
        
        text_widget = ctk.CTkTextbox(scrollable, height=400, wrap="word")
        text_widget.pack(fill="both", expand=True)
        text_widget.insert("0.0", INDICATORS_HELP_TEXT)
        text_widget.configure(state="disabled")
        
    def create_conversions_help(self, parent):
        """Create conversions help content"""
        scrollable = ctk.CTkScrollableFrame(parent)
        scrollable.pack(fill="both", expand=True, padx=10, pady=10)
        
        text_widget = ctk.CTkTextbox(scrollable, height=400, wrap="word")
        text_widget.pack(fill="both", expand=True)
        text_widget.insert("0.0", CONVERSIONS_HELP_TEXT)
        text_widget.configure(state="disabled")
        
    def create_file_formats_help(self, parent):
        """Create file formats help content"""
        scrollable = ctk.CTkScrollableFrame(parent)
        scrollable.pack(fill="both", expand=True, padx=10, pady=10)
        
        text_widget = ctk.CTkTextbox(scrollable, height=400, wrap="word")
        text_widget.pack(fill="both", expand=True)
        text_widget.insert("0.0", FILE_FORMATS_HELP_TEXT)
        text_widget.configure(state="disabled")
        
    def create_advanced_help(self, parent):
        """Create advanced help content"""
        scrollable = ctk.CTkScrollableFrame(parent)
        scrollable.pack(fill="both", expand=True, padx=10, pady=10)
        
        text_widget = ctk.CTkTextbox(scrollable, height=400, wrap="word")
        text_widget.pack(fill="both", expand=True)
        text_widget.insert("0.0", ADVANCED_HELP_TEXT)
        text_widget.configure(state="disabled")

if __name__ == "__main__":
    app = TradeAnalysisGUI()
    app.root.mainloop()
