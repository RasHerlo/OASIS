#!/usr/bin/env python3
"""
OASIS Batch Deconvolution Script for jRGECO1a Calcium Imaging Data

This script processes 2D numpy arrays where each row represents a calcium trace
from jRGECO1a indicator, performing deconvolution to extract neural activity.

Author: Generated for OASIS calcium imaging analysis
"""

import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
from datetime import datetime
import traceback
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for GUI compatibility

# OASIS imports
from oasis.functions import deconvolve, estimate_parameters
from oasis.oasis_methods import oasisAR1


class OASISBatchProcessor:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("OASIS Batch Deconvolution - jRGECO1a")
        self.root.geometry("600x500")
        
        # jRGECO1a parameters for 2.6Hz sampling
        self.sampling_rate = 2.6  # Hz
        self.jrgeco1a_tau_decay = 1.35  # seconds (typical for jRGECO1a)
        self.jrgeco1a_g_default = np.exp(-1 / (self.jrgeco1a_tau_decay * self.sampling_rate))
        
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the GUI interface"""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="OASIS Batch Deconvolution for jRGECO1a", 
                               font=('Arial', 14, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # File selection
        ttk.Label(main_frame, text="Select calcium imaging data file (.npy):").grid(row=1, column=0, sticky=tk.W, pady=5)
        
        file_frame = ttk.Frame(main_frame)
        file_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.file_path_var = tk.StringVar()
        self.file_entry = ttk.Entry(file_frame, textvariable=self.file_path_var, width=60)
        self.file_entry.grid(row=0, column=0, padx=(0, 10))
        
        ttk.Button(file_frame, text="Browse", command=self.browse_file).grid(row=0, column=1)
        
        # Processing options
        options_frame = ttk.LabelFrame(main_frame, text="Processing Options", padding="10")
        options_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=20)
        
        self.use_jrgeco1a_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Use jRGECO1a defaults (recommended)", 
                       variable=self.use_jrgeco1a_var).grid(row=0, column=0, sticky=tk.W)
        
        ttk.Label(options_frame, text=f"Sampling rate: {self.sampling_rate} Hz").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Label(options_frame, text=f"jRGECO1a decay τ: {self.jrgeco1a_tau_decay}s").grid(row=2, column=0, sticky=tk.W)
        ttk.Label(options_frame, text=f"Default g parameter: {self.jrgeco1a_g_default:.3f}").grid(row=3, column=0, sticky=tk.W)
        
        # Progress section
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="10")
        progress_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=20)
        
        self.progress_var = tk.StringVar(value="Ready to process...")
        self.progress_label = ttk.Label(progress_frame, textvariable=self.progress_var)
        self.progress_label.grid(row=0, column=0, sticky=tk.W)
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate')
        self.progress_bar.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=5, column=0, columnspan=2, pady=20)
        
        self.process_button = ttk.Button(button_frame, text="Start Processing", 
                                       command=self.start_processing)
        self.process_button.grid(row=0, column=0, padx=10)
        
        ttk.Button(button_frame, text="Exit", command=self.root.quit).grid(row=0, column=1, padx=10)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        file_frame.columnconfigure(0, weight=1)
        progress_frame.columnconfigure(0, weight=1)
        
    def browse_file(self):
        """Open file dialog to select .npy file"""
        file_path = filedialog.askopenfilename(
            title="Select calcium imaging data file",
            filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")]
        )
        if file_path:
            self.file_path_var.set(file_path)
    
    def calculate_jrgeco1a_params(self):
        """Calculate jRGECO1a-specific parameters"""
        # For jRGECO1a at 2.6Hz sampling
        g_default = self.jrgeco1a_g_default
        
        # Typical baseline for jRGECO1a (can be auto-estimated)
        baseline_default = None  # Let OASIS estimate
        
        # Noise estimate (will be auto-estimated per trace)
        noise_default = None
        
        return {
            'g': g_default,
            'baseline': baseline_default,
            'noise': noise_default,
            'penalty': 0,  # L0 penalty for sparser results
            'optimize_g': 3  # Use a few isolated events to refine g
        }
    
    def normalize_traces(self, data):
        """Normalize each trace (row) to [0,1] range, handling flat traces with NaN"""
        normalized = np.full_like(data, np.nan)
        
        for i in range(data.shape[0]):
            trace = data[i, :]
            trace_min = np.nanmin(trace)
            trace_max = np.nanmax(trace)
            
            if trace_max > trace_min:  # Not a flat trace
                normalized[i, :] = (trace - trace_min) / (trace_max - trace_min)
            # else: leave as NaN for flat traces
        
        return normalized
    
    def preprocess_data(self, data):
        """Preprocess data by removing artifact columns 725-731 for deconvolution"""
        n_traces, n_timepoints = data.shape
        
        # Check if data has expected number of columns
        if n_timepoints != 1520:
            raise ValueError(f"Method only currently made for arrays with 1520 columns. "
                           f"Your data has {n_timepoints} columns.")
        
        # Remove columns 725-731 (both included) for deconvolution
        clean_data = np.concatenate([data[:, :725], data[:, 732:]], axis=1)
        
        return clean_data
    
    def postprocess_results(self, deconv_result, spike_result, original_shape):
        """Insert NaN columns back into results at positions 725-731"""
        n_traces, n_timepoints = original_shape
        
        # Create output arrays with NaNs at artifact positions
        deconv_with_nans = np.full((n_traces, n_timepoints), np.nan)
        spike_with_nans = np.full((n_traces, n_timepoints), np.nan)
        
        # Fill in the deconvolved data
        deconv_with_nans[:, :725] = deconv_result[:, :725]  # Before artifact
        deconv_with_nans[:, 732:] = deconv_result[:, 725:]  # After artifact (shift indices)
        
        # Fill in the spike data
        spike_with_nans[:, :725] = spike_result[:, :725]   # Before artifact
        spike_with_nans[:, 732:] = spike_result[:, 725:]   # After artifact (shift indices)
        
        # Columns 725-731 remain as NaN (artifact period)
        
        return deconv_with_nans, spike_with_nans

    def create_visualization(self, original_data, denoised_data, spike_data, file_path):
        """Create 3x2 visualization subplot"""
        try:
            # Get folder name for title
            folder_path = os.path.dirname(file_path)
            folder_name = os.path.basename(folder_path)
            
            # Normalize data for visualization
            original_norm = self.normalize_traces(original_data)
            denoised_norm = self.normalize_traces(denoised_data)
            spike_norm = self.normalize_traces(spike_data)
            
            # Create figure
            fig, axes = plt.subplots(3, 2, figsize=(15, 10))
            fig.suptitle(f'OASIS Deconvolution Results - {folder_name}', fontsize=16, fontweight='bold')
            
            # Upper left: Original data
            im1 = axes[0, 0].imshow(original_data, aspect='auto', cmap='viridis', interpolation='nearest')
            axes[0, 0].set_title('Original Data')
            axes[0, 0].set_ylabel('Neurons')
            plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)
            
            # Upper right: Original normalized
            im2 = axes[0, 1].imshow(original_norm, aspect='auto', cmap='viridis', interpolation='nearest')
            axes[0, 1].set_title('Original Data (Normalized [0,1])')
            plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
            
            # Middle left: Deconvolved data
            im3 = axes[1, 0].imshow(denoised_data, aspect='auto', cmap='viridis', interpolation='nearest')
            axes[1, 0].set_title('Deconvolved Data')
            axes[1, 0].set_ylabel('Neurons')
            plt.colorbar(im3, ax=axes[1, 0], shrink=0.8)
            
            # Middle right: Deconvolved normalized
            im4 = axes[1, 1].imshow(denoised_norm, aspect='auto', cmap='viridis', interpolation='nearest')
            axes[1, 1].set_title('Deconvolved Data (Normalized [0,1])')
            plt.colorbar(im4, ax=axes[1, 1], shrink=0.8)
            
            # Lower left: Spike data
            im5 = axes[2, 0].imshow(spike_data, aspect='auto', cmap='gray', interpolation='nearest')
            axes[2, 0].set_title('Inferred Spikes')
            axes[2, 0].set_ylabel('Neurons')
            axes[2, 0].set_xlabel('Time')
            plt.colorbar(im5, ax=axes[2, 0], shrink=0.8)
            
            # Lower right: Spike data normalized
            im6 = axes[2, 1].imshow(spike_norm, aspect='auto', cmap='gray', interpolation='nearest')
            axes[2, 1].set_title('Inferred Spikes (Normalized [0,1])')
            axes[2, 1].set_xlabel('Time')
            plt.colorbar(im6, ax=axes[2, 1], shrink=0.8)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure
            base_path = os.path.splitext(file_path)[0]
            fig_file = base_path + "_visualization.png"
            plt.savefig(fig_file, dpi=300, bbox_inches='tight')
            
            # Show figure
            plt.show()
            
            return fig_file
            
        except Exception as e:
            print(f"Error creating visualization: {str(e)}")
            return None
    
    def process_single_trace(self, trace, trace_idx, params):
        """Process a single calcium trace with error handling"""
        try:
            if self.use_jrgeco1a_var.get():
                # Use jRGECO1a defaults with fallback to auto-estimation
                try:
                    # Try with jRGECO1a defaults first
                    c, s, b, g, lam = deconvolve(
                        trace, 
                        g=(params['g'],),
                        penalty=params['penalty'],
                        optimize_g=params['optimize_g']
                    )
                    method_used = "jRGECO1a_defaults"
                except:
                    # Fallback to full auto-estimation
                    c, s, b, g, lam = deconvolve(trace, penalty=params['penalty'])
                    method_used = "auto_estimation_fallback"
            else:
                # Full auto-estimation
                c, s, b, g, lam = deconvolve(trace, penalty=params['penalty'])
                method_used = "auto_estimation"
            
            # Return results
            return {
                'success': True,
                'denoised': c,
                'spikes': s,
                'baseline': b,
                'g_param': g,
                'lambda': lam,
                'method': method_used,
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
                'method': 'failed',
                'error': str(e)
            }
    
    def process_data(self, file_path):
        """Main processing function"""
        try:
            # Load data
            self.progress_var.set("Loading data...")
            self.root.update()
            
            data = np.load(file_path)
            
            if data.ndim != 2:
                raise ValueError(f"Expected 2D array, got {data.ndim}D array")
            
            original_shape = data.shape
            n_traces, n_timepoints = original_shape
            self.progress_var.set(f"Loaded data: {n_traces} traces × {n_timepoints} timepoints")
            self.root.update()
            
            # Preprocess data (remove artifact columns 725-731)
            self.progress_var.set("Preprocessing data (removing artifact columns 725-731)...")
            self.root.update()
            
            clean_data = self.preprocess_data(data)
            clean_traces, clean_timepoints = clean_data.shape
            
            self.progress_var.set(f"Preprocessed data: {clean_traces} traces × {clean_timepoints} timepoints (removed 7 artifact columns)")
            self.root.update()
            
            # Prepare output arrays for clean data
            denoised_traces_clean = np.zeros_like(clean_data, dtype=np.float64)
            spike_traces_clean = np.zeros_like(clean_data, dtype=np.float64)
            
            # Get processing parameters
            params = self.calculate_jrgeco1a_params()
            
            # Prepare log file
            base_path = os.path.splitext(file_path)[0]
            log_file = base_path + "_deconv_log.txt"
            
            with open(log_file, 'w') as log:
                log.write(f"OASIS Deconvolution Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
                log.write(f"Input file: {file_path}\\n")
                log.write(f"Data shape: {data.shape}\\n")
                log.write(f"Sampling rate: {self.sampling_rate} Hz\\n")
                log.write(f"jRGECO1a defaults: {self.use_jrgeco1a_var.get()}\\n")
                log.write(f"Default g parameter: {params['g']:.6f}\\n")
                log.write("-" * 50 + "\\n\\n")
                
                # Process each trace
                successful_traces = 0
                failed_traces = 0
                
                for i in range(n_traces):
                    # Update progress
                    progress = (i + 1) / n_traces * 100
                    self.progress_bar['value'] = progress
                    self.progress_var.set(f"Processing trace {i+1}/{n_traces} ({progress:.1f}%)")
                    self.root.update()
                    
                    # Process clean trace (without artifact columns)
                    result = self.process_single_trace(clean_data[i, :], i, params)
                    
                    # Store clean results
                    denoised_traces_clean[i, :] = result['denoised']
                    spike_traces_clean[i, :] = result['spikes']
                    
                    # Log results
                    if result['success']:
                        successful_traces += 1
                        log.write(f"Trace {i:4d}: SUCCESS - Method: {result['method']}, "
                                f"g={result['g_param']:.6f}, baseline={result['baseline']:.3f}, "
                                f"lambda={result['lambda']:.6f}\\n")
                    else:
                        failed_traces += 1
                        log.write(f"Trace {i:4d}: FAILED  - Error: {result['error']}\\n")
                
                # Summary
                log.write("\\n" + "-" * 50 + "\\n")
                log.write(f"SUMMARY:\\n")
                log.write(f"Total traces: {n_traces}\\n")
                log.write(f"Successful: {successful_traces}\\n")
                log.write(f"Failed: {failed_traces}\\n")
                log.write(f"Success rate: {successful_traces/n_traces*100:.1f}%\\n")
            
            # Postprocess results (insert NaN columns back)
            self.progress_var.set("Postprocessing results (inserting NaN artifact columns)...")
            self.root.update()
            
            denoised_traces, spike_traces = self.postprocess_results(
                denoised_traces_clean, spike_traces_clean, original_shape)
            
            # Save results
            self.progress_var.set("Saving results...")
            self.root.update()
            
            deconv_file = base_path + "_deconv.npy"
            spikes_file = base_path + "_spikes.npy"
            
            np.save(deconv_file, denoised_traces)
            np.save(spikes_file, spike_traces)
            
            # Create visualization (with original data including NaN columns)
            self.progress_var.set("Creating visualization...")
            self.root.update()
            
            # Create original data with NaN columns for visualization
            original_with_nans = data.copy()
            original_with_nans[:, 725:732] = np.nan  # Mark artifact columns as NaN
            
            viz_file = self.create_visualization(original_with_nans, denoised_traces, spike_traces, file_path)
            
            # Success message
            self.progress_var.set("Processing completed successfully!")
            self.progress_bar['value'] = 100
            
            success_msg = (f"Processing completed!\\n\\n"
                          f"Results saved:\\n"
                          f"• Denoised traces: {os.path.basename(deconv_file)}\\n"
                          f"• Spike inference: {os.path.basename(spikes_file)}\\n"
                          f"• Processing log: {os.path.basename(log_file)}\\n")
            
            if viz_file:
                success_msg += f"• Visualization: {os.path.basename(viz_file)}\\n"
            
            success_msg += f"\\nSuccess rate: {successful_traces}/{n_traces} ({successful_traces/n_traces*100:.1f}%)"
            
            messagebox.showinfo("Success", success_msg)
            
        except Exception as e:
            error_msg = f"Error during processing: {str(e)}"
            self.progress_var.set("Processing failed!")
            messagebox.showerror("Error", error_msg)
            print(f"Full error traceback:\\n{traceback.format_exc()}")
    
    def start_processing(self):
        """Start processing in a separate thread"""
        file_path = self.file_path_var.get().strip()
        
        if not file_path:
            messagebox.showwarning("Warning", "Please select a file first!")
            return
        
        if not os.path.exists(file_path):
            messagebox.showerror("Error", "Selected file does not exist!")
            return
        
        # Disable the process button during processing
        self.process_button.config(state='disabled')
        
        # Start processing in a separate thread
        processing_thread = threading.Thread(target=self._process_wrapper, args=(file_path,))
        processing_thread.daemon = True
        processing_thread.start()
    
    def _process_wrapper(self, file_path):
        """Wrapper to handle threading and re-enable button"""
        try:
            self.process_data(file_path)
        finally:
            # Re-enable the process button
            self.root.after(0, lambda: self.process_button.config(state='normal'))
    
    def run(self):
        """Run the GUI application"""
        self.root.mainloop()


def main():
    """Main function to run the OASIS batch processor"""
    print("Starting OASIS Batch Deconvolution for jRGECO1a...")
    print("=" * 50)
    print("This script processes 2D numpy arrays where each row is a calcium trace.")
    print("Optimized for jRGECO1a indicator at 2.6Hz sampling rate.")
    print("=" * 50)
    
    app = OASISBatchProcessor()
    app.run()


if __name__ == "__main__":
    main()
