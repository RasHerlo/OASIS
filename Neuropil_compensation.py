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
        self.root.geometry("1200x900")
        
        # Data storage
        self.F = None
        self.Fneu = None
        self.F_normalized = None  # Cached normalized F
        self.Fneu_normalized = None  # Cached normalized Fneu
        self.current_folder = None
        
        # Compensation parameters
        self.compensation_factor = tk.DoubleVar(value=0.0)
        
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
        
        # Set titles for placeholder subplots
        for i in range(3):
            for j in range(1, 3):
                self.axes[i][j].set_title(f"Placeholder ({i+1},{j+1})")
                self.axes[i][j].text(0.5, 0.5, "Coming Soon", 
                                   ha='center', va='center', transform=self.axes[i][j].transAxes,
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
            
            # Cache normalized matrices
            self.F_normalized = self.normalize_matrix(self.F)
            self.Fneu_normalized = self.normalize_matrix(self.Fneu)
            
            # Update displays
            self.update_displays()
            
            # Show success message
            messagebox.showinfo("Success", f"Successfully loaded matrices of shape {self.F.shape}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
            self.F = None
            self.Fneu = None
            self.F_normalized = None
            self.Fneu_normalized = None
    
    def normalize_matrix(self, matrix):
        """Normalize each row of the matrix to [0, 1] range."""
        normalized = np.zeros_like(matrix, dtype=np.float64)
        
        for i in range(matrix.shape[0]):
            row = matrix[i, :].astype(np.float64)
            row_min = np.min(row)
            row_max = np.max(row)
            
            # Handle case where max == min (constant row)
            if row_max == row_min:
                normalized[i, :] = 0.0
            else:
                normalized[i, :] = (row - row_min) / (row_max - row_min)
        
        return normalized
    
    def calculate_compensation(self, factor):
        """Calculate compensated matrix: Fcomp = F - factor * Fneu."""
        if self.F is None or self.Fneu is None:
            return None
        
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
            
            self.canvas.draw()


def main():
    """Main function to run the application."""
    root = tk.Tk()
    app = NeuropilCompensationTool(root)
    root.mainloop()


if __name__ == "__main__":
    main()
