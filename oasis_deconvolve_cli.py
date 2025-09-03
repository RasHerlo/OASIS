#!/usr/bin/env python3
"""
OASIS Batch Deconvolution CLI Script for jRGECO1a Calcium Imaging Data

Command-line version for batch processing calcium imaging traces.

Usage:
    python oasis_deconvolve_cli.py input_file.npy [--auto-only] [--sampling-rate 2.6]

Author: Generated for OASIS calcium imaging analysis
"""

import numpy as np
import os
import argparse
import sys
from datetime import datetime
import traceback
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for CLI

# OASIS imports
from oasis.functions import deconvolve


def calculate_jrgeco1a_params(sampling_rate=2.6):
    """Calculate jRGECO1a-specific parameters"""
    tau_decay = 1.35  # seconds (typical for jRGECO1a)
    g_default = np.exp(-1 / (tau_decay * sampling_rate))
    
    return {
        'g': g_default,
        'penalty': 0,  # L0 penalty for sparser results
        'optimize_g': 3  # Use a few isolated events to refine g
    }


def normalize_traces(data):
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


def preprocess_data(data):
    """Preprocess data by removing artifact columns 725-731 for deconvolution"""
    n_traces, n_timepoints = data.shape
    
    # Check if data has expected number of columns
    if n_timepoints != 1520:
        raise ValueError(f"Method only currently made for arrays with 1520 columns. "
                       f"Your data has {n_timepoints} columns.")
    
    # Remove columns 725-731 (both included) for deconvolution
    clean_data = np.concatenate([data[:, :725], data[:, 732:]], axis=1)
    
    return clean_data


def postprocess_results(deconv_result, spike_result, original_shape):
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


def create_visualization(original_data, denoised_data, spike_data, file_path):
    """Create 3x2 visualization subplot"""
    try:
        # Get folder name for title
        folder_path = os.path.dirname(file_path)
        folder_name = os.path.basename(folder_path)
        
        # Normalize data for visualization
        original_norm = normalize_traces(original_data)
        denoised_norm = normalize_traces(denoised_data)
        spike_norm = normalize_traces(spike_data)
        
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
        plt.close(fig)  # Close to free memory
        
        return fig_file
        
    except Exception as e:
        print(f"Error creating visualization: {str(e)}")
        return None


def process_single_trace(trace, params, use_jrgeco1a_defaults=True):
    """Process a single calcium trace with error handling"""
    try:
        if use_jrgeco1a_defaults:
            # Try with jRGECO1a defaults first
            try:
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
            # Full auto-estimation only
            c, s, b, g, lam = deconvolve(trace, penalty=params['penalty'])
            method_used = "auto_estimation"
        
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


def main():
    """Main function for CLI processing"""
    parser = argparse.ArgumentParser(
        description="OASIS batch deconvolution for jRGECO1a calcium imaging data"
    )
    parser.add_argument("input_file", help="Input .npy file containing 2D calcium traces")
    parser.add_argument("--auto-only", action="store_true", 
                       help="Use only auto-estimation (no jRGECO1a defaults)")
    parser.add_argument("--sampling-rate", type=float, default=2.6,
                       help="Sampling rate in Hz (default: 2.6)")
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist!")
        sys.exit(1)
    
    if not args.input_file.endswith('.npy'):
        print("Warning: Input file should be a .npy file")
    
    print("OASIS Batch Deconvolution for jRGECO1a")
    print("=" * 40)
    print(f"Input file: {args.input_file}")
    print(f"Sampling rate: {args.sampling_rate} Hz")
    print(f"Use jRGECO1a defaults: {not args.auto_only}")
    print("=" * 40)
    
    try:
        # Load data
        print("Loading data...")
        data = np.load(args.input_file)
        
        if data.ndim != 2:
            raise ValueError(f"Expected 2D array, got {data.ndim}D array")
        
        original_shape = data.shape
        n_traces, n_timepoints = original_shape
        print(f"Loaded data: {n_traces} traces × {n_timepoints} timepoints")
        
        # Preprocess data (remove artifact columns 725-731)
        print("Preprocessing data (removing artifact columns 725-731)...")
        clean_data = preprocess_data(data)
        clean_traces, clean_timepoints = clean_data.shape
        print(f"Preprocessed data: {clean_traces} traces × {clean_timepoints} timepoints (removed 7 artifact columns)")
        
        # Prepare output arrays for clean data
        denoised_traces_clean = np.zeros_like(clean_data, dtype=np.float64)
        spike_traces_clean = np.zeros_like(clean_data, dtype=np.float64)
        
        # Get processing parameters
        params = calculate_jrgeco1a_params(args.sampling_rate)
        
        # Prepare log file
        base_path = os.path.splitext(args.input_file)[0]
        log_file = base_path + "_deconv_log.txt"
        
        print(f"Processing {n_traces} traces...")
        
        with open(log_file, 'w') as log:
            log.write(f"OASIS Deconvolution Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
            log.write(f"Input file: {args.input_file}\\n")
            log.write(f"Data shape: {data.shape}\\n")
            log.write(f"Sampling rate: {args.sampling_rate} Hz\\n")
            log.write(f"jRGECO1a defaults: {not args.auto_only}\\n")
            log.write(f"Default g parameter: {params['g']:.6f}\\n")
            log.write("-" * 50 + "\\n\\n")
            
            successful_traces = 0
            failed_traces = 0
            
            for i in range(n_traces):
                # Progress indicator
                if (i + 1) % max(1, n_traces // 20) == 0 or i == n_traces - 1:
                    progress = (i + 1) / n_traces * 100
                    print(f"Progress: {i+1}/{n_traces} ({progress:.1f}%)")
                
                # Process clean trace (without artifact columns)
                result = process_single_trace(clean_data[i, :], params, not args.auto_only)
                
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
        print("Postprocessing results (inserting NaN artifact columns)...")
        denoised_traces, spike_traces = postprocess_results(
            denoised_traces_clean, spike_traces_clean, original_shape)
        
        # Save results
        print("Saving results...")
        
        deconv_file = base_path + "_deconv.npy"
        spikes_file = base_path + "_spikes.npy"
        
        np.save(deconv_file, denoised_traces)
        np.save(spikes_file, spike_traces)
        
        # Create visualization (with original data including NaN columns)
        print("Creating visualization...")
        
        # Create original data with NaN columns for visualization
        original_with_nans = data.copy()
        original_with_nans[:, 725:732] = np.nan  # Mark artifact columns as NaN
        
        viz_file = create_visualization(original_with_nans, denoised_traces, spike_traces, args.input_file)
        
        # Success message
        print("\\nProcessing completed successfully!")
        print(f"Results saved:")
        print(f"  • Denoised traces: {os.path.basename(deconv_file)}")
        print(f"  • Spike inference: {os.path.basename(spikes_file)}")
        print(f"  • Processing log: {os.path.basename(log_file)}")
        if viz_file:
            print(f"  • Visualization: {os.path.basename(viz_file)}")
        print(f"\\nSuccess rate: {successful_traces}/{n_traces} ({successful_traces/n_traces*100:.1f}%)")
        
    except Exception as e:
        print(f"\\nError during processing: {str(e)}")
        print(f"Full error traceback:\\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
