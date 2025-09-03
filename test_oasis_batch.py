#!/usr/bin/env python3
"""
Test script for OASIS batch deconvolution

Creates synthetic jRGECO1a-like data and tests the batch processing scripts.
"""

import numpy as np
import os
from oasis.functions import gen_data


def create_test_data(n_traces=10, n_timepoints=1520, sampling_rate=2.6):
    """Create synthetic calcium imaging data mimicking jRGECO1a"""
    
    # jRGECO1a parameters
    tau_decay = 1.35  # seconds
    g_jrgeco1a = np.exp(-1 / (tau_decay * sampling_rate))
    
    print(f"Creating test data with jRGECO1a-like parameters:")
    print(f"  - Decay time constant: {tau_decay}s")
    print(f"  - Sampling rate: {sampling_rate}Hz") 
    print(f"  - g parameter: {g_jrgeco1a:.4f}")
    print(f"  - Data shape: {n_traces} × {n_timepoints}")
    
    # Create multiple traces with varying properties
    all_traces = []
    
    for i in range(n_traces):
        # Vary the firing rate and noise level across traces
        fire_rate = 0.3 + 0.4 * np.random.rand()  # 0.3-0.7 Hz
        noise_level = 0.2 + 0.3 * np.random.rand()  # 0.2-0.5
        baseline = 1.0 + 2.0 * np.random.rand()  # 1-3 baseline
        
        # Generate trace
        y, true_c, true_s = gen_data(
            g=[g_jrgeco1a], 
            sn=noise_level, 
            T=n_timepoints, 
            framerate=sampling_rate,
            firerate=fire_rate,
            b=baseline,
            N=1, 
            seed=i+42
        )
        
        all_traces.append(y.squeeze())
    
    # Stack into 2D array
    data = np.vstack(all_traces)
    
    return data


def main():
    """Create test data and save it"""
    print("OASIS Batch Processing - Test Data Generator")
    print("=" * 50)
    
    # Create test data
    test_data = create_test_data(n_traces=5, n_timepoints=1520)
    
    # Save test data
    test_file = "test_calcium_data.npy"
    np.save(test_file, test_data)
    
    print(f"\\nTest data saved to: {test_file}")
    print(f"Data shape: {test_data.shape}")
    print(f"Data range: {test_data.min():.3f} to {test_data.max():.3f}")
    
    print("\\nYou can now test the batch processing scripts:")
    print("1. GUI version: python oasis_deconvolve_batch.py")
    print(f"2. CLI version: python oasis_deconvolve_cli.py {test_file}")
    
    print("\\nExpected output files:")
    print("- test_calcium_data_deconv.npy (denoised traces)")
    print("- test_calcium_data_spikes.npy (spike inference)")
    print("- test_calcium_data_deconv_log.txt (processing log)")


if __name__ == "__main__":
    main()
