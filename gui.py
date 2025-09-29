import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
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

class DarkTheme:
    """Dark theme color scheme and styling"""
    BG_DARK = "#1e1e1e"
    BG_MEDIUM = "#2d2d2d"
    BG_LIGHT = "#3c3c3c"
    BG_HOVER = "#404040"
    TEXT_PRIMARY = "#ffffff"
    TEXT_SECONDARY = "#cccccc"
    TEXT_DISABLED = "#666666"
    ACCENT_BLUE = "#0078d4"
    ACCENT_GREEN = "#107c10"
    ACCENT_RED = "#d13438"
    BORDER = "#555555"

class TradeAnalysisGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Trade Performance Analysis Dashboard")
        self.root.geometry("1400x900")
        self.root.configure(bg=DarkTheme.BG_DARK)
        
        # Variables
        self.trades_path = tk.StringVar()
        self.ohlcv_path = tk.StringVar()
        self.feature_paths = []
        self.group_by = tk.StringVar(value="amount")
        self.num_bins = tk.IntVar(value=10)
        
        # Feature data
        self.available_features = []
        self.available_chart_indicators = []
        self.feature_selections = {}  # {feature_name: (selected_var, conversion_var)}
        self.chart_selections = {}    # {indicator_name: (selected_var, conversion_var)}
        
        self.setup_styles()
        self.create_widgets()
        self.center_window()
        
    def setup_styles(self):
        """Configure dark theme styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure('Dark.TFrame', background=DarkTheme.BG_DARK)
        style.configure('Medium.TFrame', background=DarkTheme.BG_MEDIUM, relief='solid', borderwidth=1)
        style.configure('Light.TFrame', background=DarkTheme.BG_LIGHT)
        
        style.configure('Dark.TLabel', background=DarkTheme.BG_DARK, foreground=DarkTheme.TEXT_PRIMARY, font=('Segoe UI', 10))
        style.configure('Header.TLabel', background=DarkTheme.BG_DARK, foreground=DarkTheme.TEXT_PRIMARY, font=('Segoe UI', 12, 'bold'))
        style.configure('Section.TLabel', background=DarkTheme.BG_MEDIUM, foreground=DarkTheme.TEXT_PRIMARY, font=('Segoe UI', 11, 'bold'))
        
        style.configure('Dark.TButton', background=DarkTheme.BG_LIGHT, foreground=DarkTheme.TEXT_PRIMARY, borderwidth=1, relief='solid')
        style.map('Dark.TButton', background=[('active', DarkTheme.BG_HOVER)])
        
        style.configure('Accent.TButton', background=DarkTheme.ACCENT_BLUE, foreground=DarkTheme.TEXT_PRIMARY, borderwidth=0)
        style.map('Accent.TButton', background=[('active', '#106ebe')])
        
        style.configure('Success.TButton', background=DarkTheme.ACCENT_GREEN, foreground=DarkTheme.TEXT_PRIMARY, borderwidth=0)
        style.map('Success.TButton', background=[('active', '#0e6e0e')])
        
        style.configure('Dark.TEntry', fieldbackground=DarkTheme.BG_LIGHT, foreground=DarkTheme.TEXT_PRIMARY, borderwidth=1)
        style.configure('Dark.TCombobox', fieldbackground=DarkTheme.BG_LIGHT, foreground=DarkTheme.TEXT_PRIMARY, borderwidth=1)
        style.configure('Dark.TCheckbutton', background=DarkTheme.BG_MEDIUM, foreground=DarkTheme.TEXT_PRIMARY, focuscolor='none')
        style.configure('Dark.TRadiobutton', background=DarkTheme.BG_MEDIUM, foreground=DarkTheme.TEXT_PRIMARY, focuscolor='none')
        
        style.configure('Dark.Horizontal.TProgressbar', background=DarkTheme.ACCENT_BLUE, troughcolor=DarkTheme.BG_LIGHT)
        
    def center_window(self):
        """Center the window on screen"""
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (self.root.winfo_width() // 2)
        y = (self.root.winfo_screenheight() // 2) - (self.root.winfo_height() // 2)
        self.root.geometry(f'+{x}+{y}')
        
    def create_widgets(self):
        """Create the main GUI layout"""
        # Main container with padding
        main_frame = ttk.Frame(self.root, style='Dark.TFrame', padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Trade Performance Analysis Dashboard", style='Header.TLabel')
        title_label.pack(pady=(0, 20))
        
        # Create notebook for organized sections
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: File Selection
        self.create_file_selection_tab(notebook)
        
        # Tab 2: Feature Selection
        self.create_feature_selection_tab(notebook)
        
        # Tab 3: Analysis Settings
        self.create_settings_tab(notebook)
        
        # Bottom control panel
        self.create_control_panel(main_frame)
        
    def create_file_selection_tab(self, parent):
        """Create file selection tab"""
        tab_frame = ttk.Frame(parent, style='Dark.TFrame', padding="20")
        parent.add(tab_frame, text="📁 File Selection")
        
        # Trades CSV (Mandatory)
        trades_frame = ttk.LabelFrame(tab_frame, text=" Trades CSV (Required) ", style='Medium.TFrame', padding="15")
        trades_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(trades_frame, text="Select your trades CSV file:", style='Dark.TLabel').pack(anchor=tk.W)
        
        trades_input_frame = ttk.Frame(trades_frame, style='Dark.TFrame')
        trades_input_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.trades_entry = ttk.Entry(trades_input_frame, textvariable=self.trades_path, style='Dark.TEntry', state='readonly')
        self.trades_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        ttk.Button(trades_input_frame, text="Browse", command=self.select_trades_file, style='Dark.TButton').pack(side=tk.RIGHT)
        
        # OHLCV CSV (Optional)
        ohlcv_frame = ttk.LabelFrame(tab_frame, text=" OHLCV CSV (Optional - for chart indicators) ", style='Medium.TFrame', padding="15")
        ohlcv_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(ohlcv_frame, text="Select OHLCV data for chart indicator analysis:", style='Dark.TLabel').pack(anchor=tk.W)
        
        ohlcv_input_frame = ttk.Frame(ohlcv_frame, style='Dark.TFrame')
        ohlcv_input_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.ohlcv_entry = ttk.Entry(ohlcv_input_frame, textvariable=self.ohlcv_path, style='Dark.TEntry', state='readonly')
        self.ohlcv_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        ttk.Button(ohlcv_input_frame, text="Browse", command=self.select_ohlcv_file, style='Dark.TButton').pack(side=tk.RIGHT)
        
        # Feature CSVs
        features_frame = ttk.LabelFrame(tab_frame, text=" Feature CSV Files ", style='Medium.TFrame', padding="15")
        features_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(features_frame, text="Add feature CSV files:", style='Dark.TLabel').pack(anchor=tk.W)
        
        # Feature files list with scrollbar
        list_frame = ttk.Frame(features_frame, style='Dark.TFrame')
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Listbox with scrollbar
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.feature_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, 
                                         bg=DarkTheme.BG_LIGHT, fg=DarkTheme.TEXT_PRIMARY,
                                         selectbackground=DarkTheme.ACCENT_BLUE, borderwidth=0,
                                         font=('Segoe UI', 10))
        self.feature_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.feature_listbox.yview)
        
        # Buttons for feature files
        feature_btn_frame = ttk.Frame(features_frame, style='Dark.TFrame')
        feature_btn_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(feature_btn_frame, text="Add Feature CSV", command=self.add_feature_file, style='Dark.TButton').pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(feature_btn_frame, text="Remove Selected", command=self.remove_feature_file, style='Dark.TButton').pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(feature_btn_frame, text="Load Features", command=self.load_available_features, style='Accent.TButton').pack(side=tk.RIGHT)
        
    def create_feature_selection_tab(self, parent):
        """Create feature selection and conversion tab"""
        tab_frame = ttk.Frame(parent, style='Dark.TFrame', padding="20")
        parent.add(tab_frame, text="⚙️ Feature Selection")
        
        # Create canvas and scrollbar for scrollable content
        canvas = tk.Canvas(tab_frame, bg=DarkTheme.BG_DARK, highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas, style='Dark.TFrame')
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Bind mousewheel to canvas
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Initial message
        self.no_features_label = ttk.Label(self.scrollable_frame, 
                                          text="Please load feature files first in the File Selection tab",
                                          style='Dark.TLabel', font=('Segoe UI', 11, 'italic'))
        self.no_features_label.pack(pady=50)
        
    def create_settings_tab(self, parent):
        """Create analysis settings tab"""
        tab_frame = ttk.Frame(parent, style='Dark.TFrame', padding="20")
        parent.add(tab_frame, text="📊 Analysis Settings")
        
        # Binning settings
        binning_frame = ttk.LabelFrame(tab_frame, text=" Binning Configuration ", style='Medium.TFrame', padding="15")
        binning_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Number of bins
        bins_frame = ttk.Frame(binning_frame, style='Dark.TFrame')
        bins_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(bins_frame, text="Number of bins:", style='Dark.TLabel').pack(side=tk.LEFT)
        bins_spinbox = ttk.Spinbox(bins_frame, from_=5, to=50, textvariable=self.num_bins, width=10, style='Dark.TEntry')
        bins_spinbox.pack(side=tk.LEFT, padx=(10, 0))
        
        # Group by method
        ttk.Label(binning_frame, text="Binning method:", style='Dark.TLabel').pack(anchor=tk.W, pady=(10, 5))
        
        radio_frame = ttk.Frame(binning_frame, style='Dark.TFrame')
        radio_frame.pack(anchor=tk.W)
        
        ttk.Radiobutton(radio_frame, text="Equal width (size)", variable=self.group_by, value="size", style='Dark.TRadiobutton').pack(side=tk.LEFT, padx=(0, 20))
        ttk.Radiobutton(radio_frame, text="Equal count (amount)", variable=self.group_by, value="amount", style='Dark.TRadiobutton').pack(side=tk.LEFT)
        
        # Output settings
        output_frame = ttk.LabelFrame(tab_frame, text=" Output Configuration ", style='Medium.TFrame', padding="15")
        output_frame.pack(fill=tk.X)
        
        ttk.Label(output_frame, text="Save directory: ./data/trade_analysis", style='Dark.TLabel').pack(anchor=tk.W)
        ttk.Label(output_frame, text="(Results will be organized by feature in subfolders)", 
                 style='Dark.TLabel', font=('Segoe UI', 9, 'italic')).pack(anchor=tk.W, pady=(5, 0))
        
    def create_control_panel(self, parent):
        """Create bottom control panel"""
        control_frame = ttk.Frame(parent, style='Medium.TFrame', padding="15")
        control_frame.pack(fill=tk.X, pady=(20, 0))
        
        # Progress bar
        self.progress = ttk.Progressbar(control_frame, mode='indeterminate', style='Dark.Horizontal.TProgressbar')
        self.progress.pack(fill=tk.X, pady=(0, 10))
        
        # Status label
        self.status_label = ttk.Label(control_frame, text="Ready to start analysis", style='Dark.TLabel')
        self.status_label.pack(pady=(0, 10))
        
        # Buttons
        btn_frame = ttk.Frame(control_frame, style='Dark.TFrame')
        btn_frame.pack()
        
        ttk.Button(btn_frame, text="Start Analysis", command=self.start_analysis, style='Success.TButton').pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(btn_frame, text="Reset All", command=self.reset_all, style='Dark.TButton').pack(side=tk.LEFT)
        
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
            self.feature_listbox.insert(tk.END, os.path.basename(filename))
            
    def remove_feature_file(self):
        """Remove selected feature file"""
        selection = self.feature_listbox.curselection()
        if selection:
            index = selection[0]
            self.feature_listbox.delete(index)
            del self.feature_paths[index]
            
    def load_available_features(self):
        """Load available features from selected CSV files"""
        if not self.feature_paths:
            messagebox.showwarning("No Files", "Please add at least one feature CSV file first.")
            return
            
        try:
            self.status_label.config(text="Loading features...")
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
            self.status_label.config(text=f"Loaded {len(self.available_features)} available features")
            
        except Exception as e:
            self.progress.stop()
            self.status_label.config(text="Error loading features")
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
        
        # Conversion options
        conversion_options = [
            "None",
            "zscore", "robust_zscore", "rolling_zscore",
            "expanding_zscore", "ewm_zscore", "rolling_robust_zscore",
            "minmax", "rolling_minmax", "expanding_minmax",
            "pct_change", "diff", "log_return"
        ]
        
        # CSV Features section
        if self.available_features:
            csv_frame = ttk.LabelFrame(self.scrollable_frame, text=" CSV Features ", style='Medium.TFrame', padding="15")
            csv_frame.pack(fill=tk.X, pady=(0, 15))
            
            # Select all/none buttons
            btn_frame = ttk.Frame(csv_frame, style='Dark.TFrame')
            btn_frame.pack(fill=tk.X, pady=(0, 10))
            
            ttk.Button(btn_frame, text="Select All", command=lambda: self.toggle_all_features(True), style='Dark.TButton').pack(side=tk.LEFT, padx=(0, 5))
            ttk.Button(btn_frame, text="Select None", command=lambda: self.toggle_all_features(False), style='Dark.TButton').pack(side=tk.LEFT)
            
            # Create grid for features
            self.create_feature_grid(csv_frame, self.available_features, self.feature_selections, conversion_options)
            
        # Chart indicators section
        if self.available_chart_indicators:
            chart_frame = ttk.LabelFrame(self.scrollable_frame, text=" Chart Indicators ", style='Medium.TFrame', padding="15")
            chart_frame.pack(fill=tk.X, pady=(0, 15))
            
            # Select all/none buttons for chart indicators
            btn_frame = ttk.Frame(chart_frame, style='Dark.TFrame')
            btn_frame.pack(fill=tk.X, pady=(0, 10))
            
            ttk.Button(btn_frame, text="Select All", command=lambda: self.toggle_all_chart_features(True), style='Dark.TButton').pack(side=tk.LEFT, padx=(0, 5))
            ttk.Button(btn_frame, text="Select None", command=lambda: self.toggle_all_chart_features(False), style='Dark.TButton').pack(side=tk.LEFT)
            
            # Create grid for chart indicators
            self.create_feature_grid(chart_frame, self.available_chart_indicators, self.chart_selections, conversion_options)
            
        if not self.available_features and not self.available_chart_indicators:
            ttk.Label(self.scrollable_frame, text="No features available. Please load feature files first.", 
                     style='Dark.TLabel').pack(pady=50)
                     
    def create_feature_grid(self, parent, features, selections_dict, conversion_options):
        """Create a grid of feature checkboxes and conversion dropdowns"""
        # Header
        header_frame = ttk.Frame(parent, style='Dark.TFrame')
        header_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(header_frame, text="Feature", style='Dark.TLabel', font=('Segoe UI', 10, 'bold')).grid(row=0, column=0, sticky=tk.W, padx=(0, 20))
        ttk.Label(header_frame, text="Convert", style='Dark.TLabel', font=('Segoe UI', 10, 'bold')).grid(row=0, column=1, sticky=tk.W)
        
        # Features grid in chunks for better performance
        features_frame = ttk.Frame(parent, style='Dark.TFrame')
        features_frame.pack(fill=tk.X)
        
        for i, feature in enumerate(features):
            row = i // 2
            col_offset = (i % 2) * 3
            
            # Checkbox for selection
            selected_var = tk.BooleanVar()
            conversion_var = tk.StringVar(value="None")
            selections_dict[feature] = (selected_var, conversion_var)
            
            checkbox = ttk.Checkbutton(features_frame, text=feature, variable=selected_var, style='Dark.TCheckbutton')
            checkbox.grid(row=row, column=col_offset, sticky=tk.W, padx=(0, 10), pady=2)
            
            # Conversion dropdown
            combo = ttk.Combobox(features_frame, textvariable=conversion_var, values=conversion_options, 
                               state="readonly", style='Dark.TCombobox', width=15)
            combo.grid(row=row, column=col_offset+1, sticky=tk.W, padx=(0, 30), pady=2)
            
    def toggle_all_features(self, select: bool):
        """Toggle all CSV feature selections"""
        for selected_var, _ in self.feature_selections.values():
            selected_var.set(select)
            
    def toggle_all_chart_features(self, select: bool):
        """Toggle all chart indicator selections"""
        for selected_var, _ in self.chart_selections.values():
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
        csv_selected = any(selected.get() for selected, _ in self.feature_selections.values())
        chart_selected = any(selected.get() for selected, _ in self.chart_selections.values()) if self.chart_selections else False
        
        if not csv_selected and not chart_selected:
            messagebox.showerror("No Features", "Please select at least one feature to analyze.")
            return False
            
        return True
        
    def get_selected_features_and_transforms(self):
        """Get selected features and their transformations"""
        selected_features = []
        feature_transforms = {}
        
        # CSV features
        for feature, (selected_var, conversion_var) in self.feature_selections.items():
            if selected_var.get():
                selected_features.append(feature)
                conversion = conversion_var.get()
                if conversion != "None":
                    feature_transforms[feature] = conversion
                    
        # Chart indicators (these will be added automatically by run_trade_analysis)
        chart_transforms = {}
        for indicator, (selected_var, conversion_var) in self.chart_selections.items():
            if selected_var.get():
                conversion = conversion_var.get()
                if conversion != "None":
                    chart_transforms[indicator] = conversion
                    
        # Merge chart transforms into feature transforms
        feature_transforms.update(chart_transforms)
        
        return selected_features, feature_transforms
        
    def start_analysis(self):
        """Start the trade analysis in a separate thread"""
        if not self.validate_inputs():
            return
            
        # Disable start button and show progress
        for widget in self.scrollable_frame.winfo_children():
            if isinstance(widget, ttk.Button):
                widget.configure(state='disabled')
                
        self.progress.start()
        self.status_label.config(text="Running trade analysis...")
        
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
                'merge_tolerance': None,
                'feature_transforms': feature_transforms if feature_transforms else None,
                'ohlcv_csv_path': self.ohlcv_path.get() if self.ohlcv_path.get() else None,
                'chart_config': None
            }
            
            # Run the analysis
            run_trade_analysis(**kwargs)
            
            # Update GUI on success
            self.root.after(0, self.analysis_complete_success)
            
        except Exception as e:
            # Update GUI on error
            self.root.after(0, lambda: self.analysis_complete_error(str(e)))
            
    def analysis_complete_success(self):
        """Handle successful analysis completion"""
        self.progress.stop()
        self.status_label.config(text="Analysis completed successfully!")
        
        # Re-enable buttons
        for widget in self.scrollable_frame.winfo_children():
            if isinstance(widget, ttk.Button):
                widget.configure(state='normal')
                
        messagebox.showinfo("Success", "Trade analysis completed successfully!\nResults saved to ./data/trade_analysis")
        
    def analysis_complete_error(self, error_msg: str):
        """Handle analysis error"""
        self.progress.stop()
        self.status_label.config(text="Analysis failed")
        
        # Re-enable buttons
        for widget in self.scrollable_frame.winfo_children():
            if isinstance(widget, ttk.Button):
                widget.configure(state='normal')
                
        messagebox.showerror("Error", f"Analysis failed:\n{error_msg}")
        
    def reset_all(self):
        """Reset all inputs and selections"""
        if messagebox.askyesno("Confirm Reset", "Are you sure you want to reset all inputs?"):
            self.trades_path.set("")
            self.ohlcv_path.set("")
            self.feature_paths.clear()
            self.feature_listbox.delete(0, tk.END)
            self.available_features.clear()
            self.available_chart_indicators.clear()
            self.feature_selections.clear()
            self.chart_selections.clear()
            
            # Clear feature selection UI
            for widget in self.scrollable_frame.winfo_children():
                widget.destroy()
                
            self.no_features_label = ttk.Label(self.scrollable_frame, 
                                              text="Please load feature files first in the File Selection tab",
                                              style='Dark.TLabel', font=('Segoe UI', 11, 'italic'))
            self.no_features_label.pack(pady=50)
            
            self.status_label.config(text="Ready to start analysis")
            
    def run(self):
        """Start the GUI application"""
        self.root.mainloop()

if __name__ == "__main__":
    app = TradeAnalysisGUI()
    app.run()
