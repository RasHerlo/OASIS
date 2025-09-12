#!/usr/bin/env python3
"""
Neuropil Compensation Tool
Interactive GUI for neuropil compensation analysis with real-time visualization.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import os
import threading
from datetime import datetime

# OASIS imports
from oasis.functions import deconvolve
from oasis.oasis_methods import oasisAR1


class NeuropilCompensationTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Neuropil Compensation Tool")
        self.root.geometry("2200x1000")
        
        # Data storage
        self.F = None
        self.Fneu = None
        self.F_normalized = None  # Cached normalized F
        self.Fneu_normalized = None  # Cached normalized Fneu
        self.current_folder = None
        
        # Compensation parameters
        self.compensation_factor = tk.DoubleVar(value=0.0)
        
        # Row selection parameters
        self.selected_row = tk.IntVar(value=1)  # 1-based indexing for UI
        
        # CC-compensation parameters
        self.cc_compensation_enabled = tk.BooleanVar(value=False)
        
        # Local Normalization parameters
        self.local_norm_enabled = tk.BooleanVar(value=False)
        self.capping_percentage = tk.DoubleVar(value=5.0)
        
        # Deconvolution parameters and results
        self.sampling_rate = 2.6  # Hz (jRGECO1a)
        self.jrgeco1a_tau_decay = 1.35  # seconds
        self.jrgeco1a_g_default = np.exp(-1 / (self.jrgeco1a_tau_decay * self.sampling_rate))
        
        # Spike matrices storage
        self.F_spikes = None
        self.Fcomp_slider_spikes = None
        self.Fcomp_cc_spikes = None
        self.Fcomp_local_norm_spikes = None
        self.deconvolution_completed = False
        
        self.setup_gui()
        
    def setup_gui(self):
        """Set up the main GUI components."""
        # File selection frame
        file_frame = ttk.Frame(self.root, padding="10")
        file_frame.pack(fill=tk.X)
        
        ttk.Label(file_frame, text="Select folder containing F.npy and Fneu.npy:").pack(side=tk.LEFT)
        ttk.Button(file_frame, text="Browse Folder", command=self.browse_folder).pack(side=tk.LEFT, padx=(10, 0))
        
        self.folder_label = ttk.Label(file_frame, text="No folder selected", foreground="gray")
        self.folder_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Create matplotlib figure with subplots
        self.fig = Figure(figsize=(12, 8), tight_layout=True)
        self.setup_subplots()
        
        # Canvas for matplotlib
        canvas_frame = ttk.Frame(self.root)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        self.canvas = FigureCanvasTkAgg(self.fig, canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Slider frame
        slider_frame = ttk.Frame(self.root, padding="10")
        slider_frame.pack(fill=tk.X)
        
        ttk.Label(slider_frame, text="Compensation Factor:").pack(side=tk.LEFT)
        
        self.slider = ttk.Scale(
            slider_frame,
            from_=-1.5,
            to=1.5,
            variable=self.compensation_factor,
            orient=tk.HORIZONTAL,
            length=300,
            command=self.on_slider_change
        )
        self.slider.pack(side=tk.LEFT, padx=(10, 0))
        
        self.factor_label = ttk.Label(slider_frame, text="0.000")
        self.factor_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Deconvolution button (same height as slider, centered)
        self.deconvolve_button = ttk.Button(
            slider_frame, 
            text="Deconvolve (OASIS)", 
            command=self.start_deconvolution
        )
        self.deconvolve_button.pack(side=tk.LEFT, padx=(50, 10))  # Center with padding
        
        # Save button (same height as slider, on the right)
        self.save_button = ttk.Button(
            slider_frame, 
            text="Save Spikes", 
            command=self.save_spike_matrices
        )
        self.save_button.pack(side=tk.LEFT, padx=(10, 0))
        
        # Row selector frame
        row_frame = ttk.Frame(self.root, padding="10")
        row_frame.pack(fill=tk.X)
        
        ttk.Label(row_frame, text="Row Selection:").pack(side=tk.LEFT)
        
        # Row selector with up/down arrows
        row_control_frame = ttk.Frame(row_frame)
        row_control_frame.pack(side=tk.LEFT, padx=(10, 0))
        
        self.row_entry = ttk.Entry(row_control_frame, textvariable=self.selected_row, width=8)
        self.row_entry.pack(side=tk.LEFT)
        self.row_entry.bind('<Return>', self.on_row_entry_change)
        self.row_entry.bind('<FocusOut>', self.on_row_entry_change)
        
        arrow_frame = ttk.Frame(row_control_frame)
        arrow_frame.pack(side=tk.LEFT, padx=(2, 0))
        
        self.up_button = ttk.Button(arrow_frame, text="▲", width=3, command=self.row_up)
        self.up_button.pack()
        
        self.down_button = ttk.Button(arrow_frame, text="▼", width=3, command=self.row_down)
        self.down_button.pack()
        
        # CC-compensation toggle
        self.cc_toggle = ttk.Checkbutton(
            row_frame, 
            text="CC-compensation", 
            variable=self.cc_compensation_enabled,
            command=self.on_toggle_change
        )
        self.cc_toggle.pack(side=tk.LEFT, padx=(20, 0))
        
        # Local Normalization toggle
        self.local_norm_toggle = ttk.Checkbutton(
            row_frame,
            text="Local Normalization",
            variable=self.local_norm_enabled,
            command=self.on_toggle_change
        )
        self.local_norm_toggle.pack(side=tk.LEFT, padx=(20, 0))
        
        # Capping percentage spinbox
        ttk.Label(row_frame, text="Cap %:").pack(side=tk.LEFT, padx=(10, 5))
        self.capping_spinbox = ttk.Spinbox(
            row_frame,
            from_=1.0,
            to=20.0,
            increment=0.5,
            textvariable=self.capping_percentage,
            width=6,
            command=self.on_capping_change
        )
        self.capping_spinbox.pack(side=tk.LEFT)
        
        # Progress bar for deconvolution (separate frame)
        self.progress_frame = ttk.Frame(self.root, padding="10")
        self.progress_frame.pack(fill=tk.X)
        
        self.progress_var = tk.StringVar(value="")
        self.progress_label = ttk.Label(self.progress_frame, textvariable=self.progress_var)
        self.progress_label.pack()
        
        self.progress_bar = ttk.Progressbar(self.progress_frame, mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=(5, 0))
        
        # Initially hide progress elements
        self.progress_frame.pack_forget()
        
    def setup_subplots(self):
        """Create the 3x3 subplot grid."""
        # Create 3x3 grid of subplots
        self.axes = []
        for i in range(3):
            row_axes = []
            for j in range(3):
                ax = self.fig.add_subplot(3, 3, i*3 + j + 1)
                ax.set_title(f"Subplot ({i+1},{j+1})")
                ax.axis('off')  # Turn off axes for placeholders
                row_axes.append(ax)
            self.axes.append(row_axes)
        
        # Set titles for the left column (functional subplots)
        self.axes[0][0].set_title("F (normalized)")
        self.axes[1][0].set_title("Fneu (normalized)")
        self.axes[2][0].set_title("Fcomp (normalized)")
        
        # Set titles for middle column (time series plots)
        self.axes[0][1].set_title("Row 1 Time Series")
        self.axes[1][1].set_title("Row 2 Time Series")
        self.axes[2][1].set_title("Row 3 Time Series")
        
        # Set titles for right column (spike matrices)
        self.axes[0][2].set_title("F Spikes")
        self.axes[1][2].set_title("Fcomp Spikes (Slider)")
        self.axes[2][2].set_title("Fcomp Spikes (CC)")
        
        # Initially show placeholder text
        for i in range(3):
            self.axes[i][2].text(0.5, 0.5, "Press Deconvolve", 
                               ha='center', va='center', transform=self.axes[i][2].transAxes,
                               fontsize=12, alpha=0.5)
            self.axes[i][2].axis('off')
        
    def browse_folder(self):
        """Open folder selection dialog and load data."""
        folder_path = filedialog.askdirectory(title="Select folder containing F.npy and Fneu.npy")
        
        if folder_path:
            self.current_folder = folder_path
            self.folder_label.config(text=os.path.basename(folder_path), foreground="black")
            self.load_data()
    
    def load_data(self):
        """Load F.npy and Fneu.npy from the selected folder."""
        try:
            # Check if required files exist
            f_path = os.path.join(self.current_folder, "F.npy")
            fneu_path = os.path.join(self.current_folder, "Fneu.npy")
            
            if not os.path.exists(f_path):
                raise FileNotFoundError(f"F.npy not found in {self.current_folder}")
            
            if not os.path.exists(fneu_path):
                raise FileNotFoundError(f"Fneu.npy not found in {self.current_folder}")
            
            # Load the arrays
            self.F = np.load(f_path)
            self.Fneu = np.load(fneu_path)
            
            # Validate dimensions
            if self.F.shape != self.Fneu.shape:
                raise ValueError(f"Matrix dimensions don't match: F {self.F.shape} vs Fneu {self.Fneu.shape}")
            
            # Remove artifacts before normalization
            self.remove_artifacts()
            
            # Cache normalized matrices
            self.F_normalized = self.normalize_matrix(self.F)
            self.Fneu_normalized = self.normalize_matrix(self.Fneu)
            
            # Reset row selector to valid range and compensation states
            self.selected_row.set(1)
            self.cc_compensation_enabled.set(False)
            self.local_norm_enabled.set(False)
            self.on_toggle_change()  # Update UI state
            
            # Reset deconvolution results
            self.F_spikes = None
            self.Fcomp_slider_spikes = None
            self.Fcomp_cc_spikes = None
            self.Fcomp_local_norm_spikes = None
            self.deconvolution_completed = False
            self.clear_spike_displays()
            
            # Update displays
            self.update_displays()
            self.update_timeseries_plots()
            
            # Show success message
            messagebox.showinfo("Success", f"Successfully loaded matrices of shape {self.F.shape}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
            self.F = None
            self.Fneu = None
            self.F_normalized = None
            self.Fneu_normalized = None
    
    def remove_artifacts(self):
        """Remove artifacts by replacing specific columns with NaN values."""
        if self.F is None or self.Fneu is None:
            return
        
        n_cols = self.F.shape[1]
        
        # Define artifact column ranges based on matrix width
        if n_cols == 1520:
            # Replace columns 725-732 (0-based: 724-731)
            artifact_start, artifact_end = 724, 731
        elif n_cols == 2890:
            # Replace columns 1379-1387 (0-based: 1378-1386)
            artifact_start, artifact_end = 1378, 1386
        else:
            # No artifact removal for other sizes
            return
        
        # Replace artifact columns with NaN
        self.F[:, artifact_start:artifact_end+1] = np.nan
        self.Fneu[:, artifact_start:artifact_end+1] = np.nan
        
        print(f"Removed artifacts: columns {artifact_start+1}-{artifact_end+1} (1-based) set to NaN")
    
    def normalize_matrix(self, matrix):
        """Normalize each row of the matrix to [0, 1] range, handling NaN values."""
        normalized = np.zeros_like(matrix, dtype=np.float64)
        
        for i in range(matrix.shape[0]):
            row = matrix[i, :].astype(np.float64)
            
            # Use nanmin and nanmax to ignore NaN values
            row_min = np.nanmin(row)
            row_max = np.nanmax(row)
            
            # Handle case where all values are NaN
            if np.isnan(row_min) or np.isnan(row_max):
                normalized[i, :] = np.nan
            # Handle case where max == min (constant row, ignoring NaNs)
            elif row_max == row_min:
                normalized[i, :] = np.where(np.isnan(row), np.nan, 0.0)
            else:
                normalized[i, :] = (row - row_min) / (row_max - row_min)
                # Preserve NaN values in the normalized result
                normalized[i, :] = np.where(np.isnan(row), np.nan, normalized[i, :])
        
        return normalized
    
    def calculate_compensation(self, factor):
        """Calculate compensated matrix: Fcomp = F - factor * Fneu."""
        if self.F is None or self.Fneu is None:
            return None
        
        # Calculate compensation on raw matrices (NaN values will propagate correctly)
        Fcomp = self.F - factor * self.Fneu
        return self.normalize_matrix(Fcomp)
    
    def calculate_cc_compensation(self):
        """Calculate compensated matrix using correlation-based compensation for each row."""
        if self.F is None or self.Fneu is None:
            return None
        
        # Initialize Fcomp with F values
        Fcomp = self.F.copy()
        
        # Calculate row-specific compensation factors based on correlations
        for row_idx in range(self.F.shape[0]):
            # Get normalized rows for correlation calculation
            f_row = self.F_normalized[row_idx, :]
            fneu_row = self.Fneu_normalized[row_idx, :]
            
            # Calculate correlation (compensation factor)
            correlation = self.calculate_correlation(f_row, fneu_row)
            
            # Handle NaN correlations by using 0.0 (no compensation)
            if np.isnan(correlation):
                compensation_factor = 0.0
            else:
                compensation_factor = correlation  # Use correlation directly (including negative values)
            
            # Apply row-specific compensation to raw data
            Fcomp[row_idx, :] = self.F[row_idx, :] - compensation_factor * self.Fneu[row_idx, :]
        
        return self.normalize_matrix(Fcomp)
    
    def calculate_local_normalization(self):
        """Calculate Local Normalization: F/Fneu with capping and re-normalization for each row."""
        if self.F_normalized is None or self.Fneu_normalized is None:
            return None
        
        capping_percent = self.capping_percentage.get()
        Fcomp = np.zeros_like(self.F_normalized, dtype=np.float64)
        
        # Process each row independently
        for row_idx in range(self.F_normalized.shape[0]):
            f_row = self.F_normalized[row_idx, :].copy()
            fneu_row = self.Fneu_normalized[row_idx, :].copy()
            
            # Step 1: Divide F by Fneu (element-wise)
            with np.errstate(divide='ignore', invalid='ignore'):
                divided_row = f_row / fneu_row
            
            # Step 2: Apply capping to avoid artifacts
            # Find valid (non-NaN) values for percentile calculation
            valid_mask = ~np.isnan(divided_row)
            
            if np.sum(valid_mask) > 0:
                valid_values = divided_row[valid_mask]
                
                if len(valid_values) > 1:
                    # Calculate percentile thresholds
                    low_threshold = np.percentile(valid_values, capping_percent)
                    high_threshold = np.percentile(valid_values, 100 - capping_percent)
                    
                    # Apply capping only to valid values
                    capped_row = divided_row.copy()
                    capped_row[valid_mask] = np.clip(valid_values, low_threshold, high_threshold)
                else:
                    # If only one valid value, no capping needed
                    capped_row = divided_row.copy()
            else:
                # If all values are NaN, keep them as NaN
                capped_row = divided_row.copy()
            
            # Step 3: Re-normalize the capped row to [0,1]
            valid_mask_capped = ~np.isnan(capped_row)
            
            if np.sum(valid_mask_capped) > 0:
                valid_capped = capped_row[valid_mask_capped]
                
                if len(valid_capped) > 1:
                    row_min = np.min(valid_capped)
                    row_max = np.max(valid_capped)
                    
                    if row_max > row_min:
                        # Normalize valid values to [0,1]
                        normalized_valid = (valid_capped - row_min) / (row_max - row_min)
                        capped_row[valid_mask_capped] = normalized_valid
                    else:
                        # If all valid values are the same, set them to 0
                        capped_row[valid_mask_capped] = 0.0
                else:
                    # Single valid value, set to 0
                    capped_row[valid_mask_capped] = 0.0
            
            # Store the processed row
            Fcomp[row_idx, :] = capped_row
        
        return Fcomp
    
    def update_displays(self):
        """Update all subplot displays."""
        if self.F is None or self.Fneu is None:
            return
        
        # Clear existing images
        for i in range(3):
            self.axes[i][0].clear()
        
        # Display F (normalized)
        self.axes[0][0].imshow(self.F_normalized, cmap='viridis', aspect='auto')
        self.axes[0][0].set_title("F (normalized)")
        self.axes[0][0].axis('on')
        
        # Display Fneu (normalized)
        self.axes[1][0].imshow(self.Fneu_normalized, cmap='viridis', aspect='auto')
        self.axes[1][0].set_title("Fneu (normalized)")
        self.axes[1][0].axis('on')
        
        # Display Fcomp (normalized) - use appropriate compensation method
        if self.cc_compensation_enabled.get():
            Fcomp_normalized = self.calculate_cc_compensation()
            title = "Fcomp (CC-compensated)"
        elif self.local_norm_enabled.get():
            Fcomp_normalized = self.calculate_local_normalization()
            title = "Fcomp (Local Normalized)"
        else:
            Fcomp_normalized = self.calculate_compensation(self.compensation_factor.get())
            title = "Fcomp (normalized)"
        
        if Fcomp_normalized is not None:
            self.axes[2][0].imshow(Fcomp_normalized, cmap='viridis', aspect='auto')
            self.axes[2][0].set_title(title)
            self.axes[2][0].axis('on')
        
        # Update time series plots
        self.update_timeseries_plots()
        
        # Refresh canvas
        self.canvas.draw()
    
    def on_slider_change(self, value):
        """Handle slider value changes."""
        factor = float(value)
        self.factor_label.config(text=f"{factor:.3f}")
        
        # Only update if CC-compensation is disabled
        if not self.cc_compensation_enabled.get() and self.F is not None and self.Fneu is not None:
            self.axes[2][0].clear()
            Fcomp_normalized = self.calculate_compensation(factor)
            if Fcomp_normalized is not None:
                self.axes[2][0].imshow(Fcomp_normalized, cmap='viridis', aspect='auto')
                self.axes[2][0].set_title("Fcomp (normalized)")
                self.axes[2][0].axis('on')
            
            # Update only Fcomp traces in time series plots (correlation doesn't change)
            self.update_fcomp_traces_only()
            
            self.canvas.draw()
    
    def on_toggle_change(self):
        """Handle compensation toggle changes with mutual exclusivity."""
        cc_enabled = self.cc_compensation_enabled.get()
        local_norm_enabled = self.local_norm_enabled.get()
        
        # Implement mutual exclusivity
        if cc_enabled and local_norm_enabled:
            # If both are enabled, disable the one that wasn't just clicked
            # We can determine this by checking which one changed
            self.local_norm_enabled.set(False)
            local_norm_enabled = False
        
        # Enable/disable controls based on active compensation method
        if cc_enabled or local_norm_enabled:
            self.slider.config(state='disabled')
            self.factor_label.config(foreground='gray')
        else:
            self.slider.config(state='normal')
            self.factor_label.config(foreground='black')
        
        # Enable/disable toggles based on state
        if cc_enabled:
            self.local_norm_toggle.config(state='disabled')
            self.capping_spinbox.config(state='disabled')
        elif local_norm_enabled:
            self.cc_toggle.config(state='disabled')
            self.capping_spinbox.config(state='normal')
        else:
            self.cc_toggle.config(state='normal')
            self.local_norm_toggle.config(state='normal')
            self.capping_spinbox.config(state='normal')
        
        # Update all displays with new compensation method
        if self.F is not None and self.Fneu is not None:
            self.update_displays()
            self.update_timeseries_plots()
            self.canvas.draw()
    
    def on_capping_change(self):
        """Handle capping percentage changes."""
        if self.local_norm_enabled.get() and self.F is not None and self.Fneu is not None:
            self.update_displays()
            self.update_timeseries_plots()
            self.canvas.draw()


    def calculate_correlation(self, f_row, fneu_row):
        """Calculate Pearson correlation between F and Fneu rows, handling NaN values."""
        # Create masks for valid (non-NaN) values
        valid_mask = ~(np.isnan(f_row) | np.isnan(fneu_row))
        
        # Check if we have enough valid data points
        if np.sum(valid_mask) < 2:
            return np.nan
        
        # Extract valid values
        f_valid = f_row[valid_mask]
        fneu_valid = fneu_row[valid_mask]
        
        # Calculate Pearson correlation
        if len(f_valid) < 2 or np.std(f_valid) == 0 or np.std(fneu_valid) == 0:
            return np.nan
        
        correlation_matrix = np.corrcoef(f_valid, fneu_valid)
        return correlation_matrix[0, 1]
    
    def update_timeseries_plots(self):
        """Update the middle column time series plots."""
        if self.F is None or self.Fneu is None:
            return
        
        start_row = self.selected_row.get() - 1  # Convert to 0-based indexing
        max_row = self.F.shape[0]
        
        # Ensure we don't go out of bounds
        if start_row + 2 >= max_row:
            return
        
        # Calculate Fcomp using appropriate compensation method
        if self.cc_compensation_enabled.get():
            Fcomp_normalized = self.calculate_cc_compensation()
        elif self.local_norm_enabled.get():
            Fcomp_normalized = self.calculate_local_normalization()
        else:
            factor = self.compensation_factor.get()
            Fcomp = self.F - factor * self.Fneu
            Fcomp_normalized = self.normalize_matrix(Fcomp)
        
        # Update each of the three middle subplots
        for i in range(3):
            current_row = start_row + i
            ax = self.axes[i][1]
            ax.clear()
            
            # Get normalized rows
            f_row = self.F_normalized[current_row, :]
            fneu_row = self.Fneu_normalized[current_row, :]
            fcomp_row = Fcomp_normalized[current_row, :]
            
            # Calculate correlation between F and Fneu (independent of compensation)
            correlation = self.calculate_correlation(f_row, fneu_row)
            
            # Create time axis
            time_points = np.arange(len(f_row))
            
            # Plot the three traces
            ax.plot(time_points, f_row, 'g-', label='F', linewidth=1.5)
            ax.plot(time_points, fneu_row, 'r-', label='Fneu', linewidth=1.5)
            ax.plot(time_points, fcomp_row, 'k-', label='Fcomp', linewidth=1.5)
            
            # Configure plot with correlation in title
            ax.set_ylim(0, 1)
            if np.isnan(correlation):
                ax.set_title(f"Row {current_row + 1} Time Series (corr=NaN)")
            else:
                ax.set_title(f"Row {current_row + 1} Time Series (corr={correlation:.2f})")
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=8)
            
            # Only show x-label on bottom subplot
            if i == 2:
                ax.set_xlabel('Time Points')
            
            ax.set_ylabel('Normalized Signal')
    
    def validate_row_selection(self):
        """Validate and constrain row selection within valid bounds."""
        if self.F is None:
            return False
        
        current_row = self.selected_row.get()
        max_allowed = self.F.shape[0] - 2  # Ensure we can show 3 consecutive rows
        
        if current_row < 1:
            self.selected_row.set(1)
            return True
        elif current_row > max_allowed:
            self.selected_row.set(max_allowed)
            return True
        
        return False
    
    def on_row_entry_change(self, event=None):
        """Handle changes to the row entry field."""
        try:
            # Validate and update if needed
            if self.validate_row_selection():
                pass  # Value was corrected
            
            # Update plots
            self.update_timeseries_plots()
            self.canvas.draw()
            
        except tk.TclError:
            # Handle invalid input by resetting to 1
            self.selected_row.set(1)
    
    def row_up(self):
        """Increment row selection."""
        if self.F is None:
            return
        
        current = self.selected_row.get()
        max_allowed = self.F.shape[0] - 2
        
        if current < max_allowed:
            self.selected_row.set(current + 1)
            self.update_timeseries_plots()
            self.canvas.draw()
    
    def row_down(self):
        """Decrement row selection."""
        current = self.selected_row.get()
        
        if current > 1:
            self.selected_row.set(current - 1)
            self.update_timeseries_plots()
            self.canvas.draw()


    def update_fcomp_traces_only(self):
        """Update only the Fcomp traces in time series plots when slider moves."""
        if self.F is None or self.Fneu is None:
            return
        
        start_row = self.selected_row.get() - 1  # Convert to 0-based indexing
        max_row = self.F.shape[0]
        
        # Ensure we don't go out of bounds
        if start_row + 2 >= max_row:
            return
        
        # Calculate Fcomp using appropriate compensation method
        if self.cc_compensation_enabled.get():
            Fcomp_normalized = self.calculate_cc_compensation()
        elif self.local_norm_enabled.get():
            Fcomp_normalized = self.calculate_local_normalization()
        else:
            factor = self.compensation_factor.get()
            Fcomp = self.F - factor * self.Fneu
            Fcomp_normalized = self.normalize_matrix(Fcomp)
        
        # Update only the Fcomp traces in each subplot
        for i in range(3):
            current_row = start_row + i
            ax = self.axes[i][1]
            
            # Get the Fcomp row
            fcomp_row = Fcomp_normalized[current_row, :]
            
            # Remove only the Fcomp line (black line, should be the last one added)
            lines = ax.get_lines()
            if len(lines) >= 3:  # Should have F, Fneu, and Fcomp lines
                lines[-1].remove()  # Remove the last line (Fcomp)
            
            # Create time axis and plot new Fcomp trace
            time_points = np.arange(len(fcomp_row))
            ax.plot(time_points, fcomp_row, 'k-', label='Fcomp', linewidth=1.5)
            
            # Update legend
            ax.legend(loc='upper right', fontsize=8)


    def preprocess_data_for_deconv(self, data):
        """Preprocess data by removing artifact columns for deconvolution (mirrors batch script)."""
        n_traces, n_timepoints = data.shape
        
        # Handle different matrix sizes
        if n_timepoints == 1520:
            # Remove columns 725-732 (0-based: 724-731) 
            clean_data = np.concatenate([data[:, :724], data[:, 732:]], axis=1)
            artifact_info = {'start': 724, 'end': 731, 'original_cols': n_timepoints}
        elif n_timepoints == 2890:
            # Remove columns 1379-1387 (0-based: 1378-1386)
            clean_data = np.concatenate([data[:, :1378], data[:, 1387:]], axis=1)
            artifact_info = {'start': 1378, 'end': 1386, 'original_cols': n_timepoints}
        else:
            # No artifact removal for other sizes - just return original data
            clean_data = data.copy()
            artifact_info = {'start': None, 'end': None, 'original_cols': n_timepoints}
        
        return clean_data, artifact_info
    
    def postprocess_results_for_deconv(self, deconv_result, spike_result, artifact_info):
        """Insert NaN columns back into results at artifact positions (mirrors batch script)."""
        if artifact_info['start'] is None:
            # No artifacts to reinsert
            return deconv_result, spike_result
        
        n_traces = deconv_result.shape[0]
        original_cols = artifact_info['original_cols']
        artifact_start = artifact_info['start']
        artifact_end = artifact_info['end']
        artifact_width = artifact_end - artifact_start + 1
        
        # Create output arrays with NaNs at artifact positions
        deconv_with_nans = np.full((n_traces, original_cols), np.nan, dtype=np.float64)
        spike_with_nans = np.full((n_traces, original_cols), np.nan, dtype=np.float64)
        
        # Fill in the deconvolved data
        deconv_with_nans[:, :artifact_start] = deconv_result[:, :artifact_start]  # Before artifact
        deconv_with_nans[:, artifact_end+1:] = deconv_result[:, artifact_start:]  # After artifact
        
        # Fill in the spike data  
        spike_with_nans[:, :artifact_start] = spike_result[:, :artifact_start]   # Before artifact
        spike_with_nans[:, artifact_end+1:] = spike_result[:, artifact_start:]   # After artifact
        
        # Columns at artifact positions remain as NaN
        
        return deconv_with_nans, spike_with_nans
    
    def calculate_jrgeco1a_params(self):
        """Calculate jRGECO1a-specific parameters for OASIS deconvolution."""
        return {
            'g': self.jrgeco1a_g_default,
            'baseline': None,  # Let OASIS estimate
            'noise': None,     # Let OASIS estimate
            'penalty': 0,      # L0 penalty for sparse results
            'optimize_g': False  # Use fixed g parameter
        }
    
    def process_single_trace_oasis(self, trace, params):
        """Process a single calcium trace with OASIS deconvolution."""
        try:
            # Use jRGECO1a defaults
            c, s, b, g, lam = deconvolve(
                trace, 
                g=(params['g'],),
                penalty=params['penalty'],
                optimize_g=params['optimize_g']
            )
            
            return {
                'success': True,
                'denoised': c,
                'spikes': s,
                'baseline': b,
                'g_param': g,
                'lambda': lam,
                'error': None
            }
            
        except Exception as e:
            # Return NaN arrays for failed processing
            return {
                'success': False,
                'denoised': np.full_like(trace, np.nan),
                'spikes': np.full_like(trace, np.nan),
                'baseline': np.nan,
                'g_param': np.nan,
                'lambda': np.nan,
                'error': str(e)
            }
    
    def deconvolve_matrix(self, matrix, matrix_name):
        """Deconvolve an entire matrix and return spike matrix (mirrors batch script approach)."""
        if matrix is None:
            return None
        
        n_traces, n_timepoints = matrix.shape
        params = self.calculate_jrgeco1a_params()
        failed_traces = []
        
        # Preprocess: Remove artifact columns for deconvolution
        self.progress_var.set(f"Preprocessing {matrix_name} (removing artifact columns)...")
        self.root.update()
        clean_matrix, artifact_info = self.preprocess_data_for_deconv(matrix)
        clean_traces, clean_timepoints = clean_matrix.shape
        
        # Prepare output arrays for clean data
        denoised_traces_clean = np.zeros_like(clean_matrix, dtype=np.float64)
        spike_traces_clean = np.zeros_like(clean_matrix, dtype=np.float64)
        
        # Process each clean trace
        for trace_idx in range(n_traces):
            # Update progress
            progress = int((trace_idx / n_traces) * 100)
            self.progress_var.set(f"Processing {matrix_name}: {trace_idx+1}/{n_traces} traces")
            self.progress_bar['value'] = progress
            self.root.update()
            
            clean_trace = clean_matrix[trace_idx, :].astype(np.float64)
            
            # Skip traces that are all NaN (shouldn't happen after preprocessing, but safety check)
            if np.all(np.isnan(clean_trace)):
                denoised_traces_clean[trace_idx, :] = np.nan
                spike_traces_clean[trace_idx, :] = np.nan
                failed_traces.append(trace_idx + 1)  # 1-based indexing for user
                continue
            
            # Process the clean trace
            result = self.process_single_trace_oasis(clean_trace, params)
            
            if result['success']:
                denoised_traces_clean[trace_idx, :] = result['denoised']
                spike_traces_clean[trace_idx, :] = result['spikes']
            else:
                denoised_traces_clean[trace_idx, :] = np.nan
                spike_traces_clean[trace_idx, :] = np.nan
                failed_traces.append(trace_idx + 1)  # 1-based indexing for user
        
        # Postprocess: Reinsert NaN columns at artifact positions
        self.progress_var.set(f"Postprocessing {matrix_name} (reinserting artifact columns)...")
        self.root.update()
        denoised_final, spike_final = self.postprocess_results_for_deconv(
            denoised_traces_clean, spike_traces_clean, artifact_info
        )
        
        return spike_final, failed_traces
    
    def start_deconvolution(self):
        """Start deconvolution process in background thread."""
        if self.F is None or self.Fneu is None:
            messagebox.showwarning("Warning", "Please load data first!")
            return
        
        # Disable button during processing
        self.deconvolve_button.config(state='disabled')
        
        # Show progress elements
        self.progress_frame.pack(fill=tk.X, pady=(10, 0))
        self.progress_var.set("Starting deconvolution...")
        self.progress_bar['value'] = 0
        
        # Start processing in background thread
        processing_thread = threading.Thread(target=self._deconvolution_worker)
        processing_thread.daemon = True
        processing_thread.start()
    
    def _deconvolution_worker(self):
        """Background worker for deconvolution processing."""
        try:
            all_failed_traces = []
            
            # 1. Deconvolve F matrix
            self.progress_var.set("Deconvolving F matrix...")
            self.root.update()
            F_spikes, F_failed = self.deconvolve_matrix(self.F, "F")
            self.F_spikes = F_spikes
            all_failed_traces.extend([(trace, "F") for trace in F_failed])
            
            # 2. Deconvolve Fcomp (slider compensation)
            self.progress_var.set("Calculating Fcomp (slider)...")
            self.root.update()
            factor = self.compensation_factor.get()
            Fcomp_slider = self.F - factor * self.Fneu
            Fcomp_slider_spikes, Fcomp_slider_failed = self.deconvolve_matrix(Fcomp_slider, "Fcomp (slider)")
            self.Fcomp_slider_spikes = Fcomp_slider_spikes
            all_failed_traces.extend([(trace, "Fcomp_slider") for trace in Fcomp_slider_failed])
            
            # 3. Deconvolve Fcomp (CC compensation)
            self.progress_var.set("Calculating Fcomp (CC)...")
            self.root.update()
            Fcomp_cc = self.calculate_cc_compensation_raw()  # Get raw CC-compensated matrix
            Fcomp_cc_spikes, Fcomp_cc_failed = self.deconvolve_matrix(Fcomp_cc, "Fcomp (CC)")
            self.Fcomp_cc_spikes = Fcomp_cc_spikes
            all_failed_traces.extend([(trace, "Fcomp_CC") for trace in Fcomp_cc_failed])
            
            # 4. Deconvolve Fcomp (Local Normalization)
            self.progress_var.set("Calculating Fcomp (Local Norm)...")
            self.root.update()
            Fcomp_local_norm = self.calculate_local_normalization_raw()  # Get raw Local Norm matrix
            Fcomp_local_norm_spikes, Fcomp_local_norm_failed = self.deconvolve_matrix(Fcomp_local_norm, "Fcomp (Local Norm)")
            self.Fcomp_local_norm_spikes = Fcomp_local_norm_spikes
            all_failed_traces.extend([(trace, "Fcomp_LocalNorm") for trace in Fcomp_local_norm_failed])
            
            # Update displays
            self.progress_var.set("Updating displays...")
            self.root.update()
            self.update_spike_displays()
            
            # Mark completion
            self.deconvolution_completed = True
            
            # Show completion message
            total_traces = self.F.shape[0]
            total_failed = len(all_failed_traces)
            success_rate = ((total_traces * 4 - total_failed) / (total_traces * 4)) * 100
            
            completion_msg = f"Deconvolution completed!\\nSuccess rate: {success_rate:.1f}%"
            if total_failed > 0:
                completion_msg += f"\\n{total_failed} traces failed (shown as white)"
            
            self.progress_var.set("Deconvolution completed!")
            messagebox.showinfo("Deconvolution Complete", completion_msg)
            
            # Show failed traces log if any
            if all_failed_traces:
                self.show_failed_traces_log(all_failed_traces)
                
        except Exception as e:
            messagebox.showerror("Error", f"Deconvolution failed: {str(e)}")
            
        finally:
            # Re-enable button and hide progress
            self.deconvolve_button.config(state='normal')
            self.progress_frame.pack_forget()
    
    def calculate_cc_compensation_raw(self):
        """Calculate raw CC-compensated matrix (not normalized)."""
        if self.F is None or self.Fneu is None:
            return None
        
        # Initialize Fcomp with F values
        Fcomp = self.F.copy()
        
        # Calculate row-specific compensation factors based on correlations
        for row_idx in range(self.F.shape[0]):
            # Get normalized rows for correlation calculation
            f_row = self.F_normalized[row_idx, :]
            fneu_row = self.Fneu_normalized[row_idx, :]
            
            # Calculate correlation (compensation factor)
            correlation = self.calculate_correlation(f_row, fneu_row)
            
            # Handle NaN correlations by using 0.0 (no compensation)
            if np.isnan(correlation):
                compensation_factor = 0.0
            else:
                compensation_factor = correlation
            
            # Apply row-specific compensation to raw data
            Fcomp[row_idx, :] = self.F[row_idx, :] - compensation_factor * self.Fneu[row_idx, :]
        
        return Fcomp
    
    def calculate_local_normalization_raw(self):
        """Calculate raw Local Normalization matrix (not normalized) for deconvolution."""
        if self.F is None or self.Fneu is None:
            return None
        
        # First normalize the raw matrices
        F_normalized = self.normalize_matrix(self.F)
        Fneu_normalized = self.normalize_matrix(self.Fneu)
        
        capping_percent = self.capping_percentage.get()
        Fcomp = np.zeros_like(self.F, dtype=np.float64)
        
        # Process each row independently
        for row_idx in range(F_normalized.shape[0]):
            f_row = F_normalized[row_idx, :].copy()
            fneu_row = Fneu_normalized[row_idx, :].copy()
            
            # Step 1: Divide F by Fneu (element-wise)
            with np.errstate(divide='ignore', invalid='ignore'):
                divided_row = f_row / fneu_row
            
            # Step 2: Apply capping to avoid artifacts
            valid_mask = ~np.isnan(divided_row)
            
            if np.sum(valid_mask) > 0:
                valid_values = divided_row[valid_mask]
                
                if len(valid_values) > 1:
                    # Calculate percentile thresholds
                    low_threshold = np.percentile(valid_values, capping_percent)
                    high_threshold = np.percentile(valid_values, 100 - capping_percent)
                    
                    # Apply capping only to valid values
                    capped_row = divided_row.copy()
                    capped_row[valid_mask] = np.clip(valid_values, low_threshold, high_threshold)
                else:
                    capped_row = divided_row.copy()
            else:
                capped_row = divided_row.copy()
            
            # Step 3: Re-normalize the capped row to [0,1] and scale to original F range
            valid_mask_capped = ~np.isnan(capped_row)
            
            if np.sum(valid_mask_capped) > 0:
                valid_capped = capped_row[valid_mask_capped]
                
                if len(valid_capped) > 1:
                    row_min = np.min(valid_capped)
                    row_max = np.max(valid_capped)
                    
                    if row_max > row_min:
                        # Normalize to [0,1] then scale to original F range
                        normalized_valid = (valid_capped - row_min) / (row_max - row_min)
                        
                        # Scale to original F row range for deconvolution
                        f_raw_row = self.F[row_idx, :]
                        f_valid_mask = ~np.isnan(f_raw_row)
                        
                        if np.sum(f_valid_mask) > 0:
                            f_min = np.nanmin(f_raw_row)
                            f_max = np.nanmax(f_raw_row)
                            scaled_valid = normalized_valid * (f_max - f_min) + f_min
                            capped_row[valid_mask_capped] = scaled_valid
                        else:
                            capped_row[valid_mask_capped] = normalized_valid
                    else:
                        # Use original F values if no variation
                        f_raw_values = self.F[row_idx, valid_mask_capped]
                        capped_row[valid_mask_capped] = f_raw_values
                else:
                    # Single valid value, use original F value
                    f_raw_values = self.F[row_idx, valid_mask_capped]
                    capped_row[valid_mask_capped] = f_raw_values
            
            # Store the processed row
            Fcomp[row_idx, :] = capped_row
        
        return Fcomp
    
    def update_spike_displays(self):
        """Update the right column spike matrix displays."""
        # Determine which spike matrices to show based on active compensation method
        if self.local_norm_enabled.get():
            spike_matrices = [
                (self.F_spikes, "F Spikes"),
                (self.Fcomp_local_norm_spikes, "Fcomp Spikes (Local Norm)"),
                (self.Fcomp_cc_spikes, "Fcomp Spikes (CC)")
            ]
        else:
            spike_matrices = [
                (self.F_spikes, "F Spikes"),
                (self.Fcomp_slider_spikes, "Fcomp Spikes (Slider)"),
                (self.Fcomp_cc_spikes, "Fcomp Spikes (CC)")
            ]
        
        for i, (spike_matrix, title) in enumerate(spike_matrices):
            ax = self.axes[i][2]
            ax.clear()
            
            if spike_matrix is not None:
                # Display spike matrix in grayscale (1=black, 0=white)
                ax.imshow(spike_matrix, cmap='gray_r', aspect='auto', vmin=0, vmax=1)
                ax.set_title(title)
                ax.axis('on')
            else:
                ax.set_title(title)
                ax.text(0.5, 0.5, "Press Deconvolve", 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=12, alpha=0.5)
                ax.axis('off')
        
        self.canvas.draw()
    
    def clear_spike_displays(self):
        """Clear the spike matrix displays."""
        for i in range(3):
            ax = self.axes[i][2]
            ax.clear()
            ax.text(0.5, 0.5, "Press Deconvolve", 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, alpha=0.5)
            ax.axis('off')
        self.canvas.draw()
    
    def show_failed_traces_log(self, failed_traces):
        """Show a log of failed traces in a popup window."""
        log_window = tk.Toplevel(self.root)
        log_window.title("Deconvolution Failed Traces")
        log_window.geometry("400x300")
        
        # Create scrollable text widget
        frame = ttk.Frame(log_window)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_widget = tk.Text(frame, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        # Group failed traces by matrix type
        f_failed = [trace for trace, matrix in failed_traces if matrix == "F"]
        slider_failed = [trace for trace, matrix in failed_traces if matrix == "Fcomp_slider"]
        cc_failed = [trace for trace, matrix in failed_traces if matrix == "Fcomp_CC"]
        local_norm_failed = [trace for trace, matrix in failed_traces if matrix == "Fcomp_LocalNorm"]
        
        log_content = f"Deconvolution Failed Traces Log\\n"
        log_content += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n"
        log_content += "=" * 40 + "\\n\\n"
        
        if f_failed:
            log_content += f"F matrix failed traces ({len(f_failed)}): {', '.join(map(str, f_failed))}\\n\\n"
        
        if slider_failed:
            log_content += f"Fcomp (slider) failed traces ({len(slider_failed)}): {', '.join(map(str, slider_failed))}\\n\\n"
        
        if cc_failed:
            log_content += f"Fcomp (CC) failed traces ({len(cc_failed)}): {', '.join(map(str, cc_failed))}\\n\\n"
        
        if local_norm_failed:
            log_content += f"Fcomp (Local Norm) failed traces ({len(local_norm_failed)}): {', '.join(map(str, local_norm_failed))}\\n\\n"
        
        log_content += "These traces are displayed as white (NaN) in the spike matrices."
        
        text_widget.insert(tk.END, log_content)
        text_widget.config(state=tk.DISABLED)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Add close button
        ttk.Button(log_window, text="Close", command=log_window.destroy).pack(pady=10)
    
    def save_spike_matrices(self):
        """Save the three deconvolved spike matrices to the original data folder."""
        # Check if deconvolution has been completed
        if not self.deconvolution_completed or self.current_folder is None:
            messagebox.showwarning("Warning", "Please complete deconvolution first!")
            return
        
        # Check if all spike matrices are available
        if (self.F_spikes is None or self.Fcomp_slider_spikes is None or 
            self.Fcomp_cc_spikes is None or self.Fcomp_local_norm_spikes is None):
            messagebox.showwarning("Warning", "Spike matrices are not available. Please run deconvolution first!")
            return
        
        try:
            # Get the slider value used for compensation
            slider_value = self.compensation_factor.get()
            
            # Define filenames
            f_filename = os.path.join(self.current_folder, "F_deconv.npy")
            fcomp_slider_filename = os.path.join(self.current_folder, f"Fcomp_{slider_value:.3f}.npy")
            fcomp_cc_filename = os.path.join(self.current_folder, "Fcomp_CC.npy")
            fcomp_local_norm_filename = os.path.join(self.current_folder, "Fcomp_FneuNorm_Spks.npy")
            
            # Save the matrices
            np.save(f_filename, self.F_spikes)
            np.save(fcomp_slider_filename, self.Fcomp_slider_spikes)
            np.save(fcomp_cc_filename, self.Fcomp_cc_spikes)
            np.save(fcomp_local_norm_filename, self.Fcomp_local_norm_spikes)
            
            # Show success message with file locations
            success_msg = f"Successfully saved spike matrices:\\n\\n"
            success_msg += f"• F_deconv.npy\\n"
            success_msg += f"• Fcomp_{slider_value:.3f}.npy\\n"
            success_msg += f"• Fcomp_CC.npy\\n"
            success_msg += f"• Fcomp_FneuNorm_Spks.npy\\n\\n"
            success_msg += f"Location: {self.current_folder}"
            
            messagebox.showinfo("Save Complete", success_msg)
            
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save spike matrices:\\n{str(e)}")


def main():
    """Main function to run the application."""
    root = tk.Tk()
    app = NeuropilCompensationTool(root)
    root.mainloop()


if __name__ == "__main__":
    main()
