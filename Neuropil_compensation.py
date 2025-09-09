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


class NeuropilCompensationTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Neuropil Compensation Tool")
        self.root.geometry("1800x900")
        
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
        
        # Set titles for right column placeholder subplots
        for i in range(3):
            self.axes[i][2].set_title(f"Placeholder ({i+1},{3})")
            self.axes[i][2].text(0.5, 0.5, "Coming Soon", 
                               ha='center', va='center', transform=self.axes[i][2].transAxes,
                               fontsize=12, alpha=0.5)
        
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
            
            # Reset row selector to valid range
            self.selected_row.set(1)
            
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
        
        # Display Fcomp (normalized)
        Fcomp_normalized = self.calculate_compensation(self.compensation_factor.get())
        if Fcomp_normalized is not None:
            self.axes[2][0].imshow(Fcomp_normalized, cmap='viridis', aspect='auto')
            self.axes[2][0].set_title("Fcomp (normalized)")
            self.axes[2][0].axis('on')
        
        # Update time series plots
        self.update_timeseries_plots()
        
        # Refresh canvas
        self.canvas.draw()
    
    def on_slider_change(self, value):
        """Handle slider value changes."""
        factor = float(value)
        self.factor_label.config(text=f"{factor:.3f}")
        
        # Update only the Fcomp display
        if self.F is not None and self.Fneu is not None:
            self.axes[2][0].clear()
            Fcomp_normalized = self.calculate_compensation(factor)
            if Fcomp_normalized is not None:
                self.axes[2][0].imshow(Fcomp_normalized, cmap='viridis', aspect='auto')
                self.axes[2][0].set_title("Fcomp (normalized)")
                self.axes[2][0].axis('on')
            
            # Update time series plots
            self.update_timeseries_plots()
            
            self.canvas.draw()


    def update_timeseries_plots(self):
        """Update the middle column time series plots."""
        if self.F is None or self.Fneu is None:
            return
        
        start_row = self.selected_row.get() - 1  # Convert to 0-based indexing
        max_row = self.F.shape[0]
        
        # Ensure we don't go out of bounds
        if start_row + 2 >= max_row:
            return
        
        # Calculate Fcomp for current compensation factor
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
            
            # Create time axis
            time_points = np.arange(len(f_row))
            
            # Plot the three traces
            ax.plot(time_points, f_row, 'g-', label='F', linewidth=1.5)
            ax.plot(time_points, fneu_row, 'r-', label='Fneu', linewidth=1.5)
            ax.plot(time_points, fcomp_row, 'k-', label='Fcomp', linewidth=1.5)
            
            # Configure plot
            ax.set_ylim(0, 1)
            ax.set_title(f"Row {current_row + 1} Time Series")
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


def main():
    """Main function to run the application."""
    root = tk.Tk()
    app = NeuropilCompensationTool(root)
    root.mainloop()


if __name__ == "__main__":
    main()
