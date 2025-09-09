# OASIS Batch Deconvolution for jRGECO1a Calcium Imaging

This package provides batch processing scripts for deconvolving calcium imaging traces using the OASIS algorithm, specifically optimized for jRGECO1a indicator data.

## 📋 Features

- **Batch Processing**: Process entire 2D arrays where each row is a calcium trace
- **jRGECO1a Optimization**: Pre-configured parameters for jRGECO1a indicator (τ ≈ 1.35s)
- **2.6Hz Sampling**: Optimized for your specific sampling rate
- **Error Handling**: Robust processing with NaN outputs for failed traces
- **Dual Interface**: Both GUI and command-line versions available
- **Comprehensive Logging**: Detailed processing logs with success rates

## 📁 Files Included

- `oasis_deconvolve_batch.py` - GUI version with file picker and progress bar
- `oasis_deconvolve_cli.py` - Command-line version for scripting
- `test_oasis_batch.py` - Test data generator
- `OASIS_BATCH_README.md` - This documentation

## 🚀 Quick Start

### GUI Version (Recommended for first-time users)
```bash
python oasis_deconvolve_batch.py
```
- Click "Browse" to select your `.npy` file
- Choose processing options
- Click "Start Processing"
- Results will be saved automatically

### Command-Line Version
```bash
# Basic usage with jRGECO1a defaults
python oasis_deconvolve_cli.py your_data.npy

# Use only auto-estimation (no jRGECO1a defaults)
python oasis_deconvolve_cli.py your_data.npy --auto-only

# Custom sampling rate
python oasis_deconvolve_cli.py your_data.npy --sampling-rate 5.0
```

## 📊 Input Data Format

Your input data should be:
- **File format**: `.npy` (NumPy array)
- **Array shape**: `(n_traces, n_timepoints)`
- **Data type**: Numeric (will be converted to `np.float64`)
- **Content**: Raw fluorescence traces (one per row)

Example:
```python
import numpy as np

# Your data should look like this:
data = np.array([
    [trace1_timepoint1, trace1_timepoint2, ...],  # ROI 1
    [trace2_timepoint1, trace2_timepoint2, ...],  # ROI 2
    [trace3_timepoint1, trace3_timepoint2, ...],  # ROI 3
    # ... more traces
])

# Shape: (n_ROIs, n_timepoints)
print(data.shape)  # e.g., (100, 3000) for 100 ROIs with 3000 timepoints each
```

## 📤 Output Files

For input file `F.npy`, the following files are created:

1. **`F_deconv.npy`** - Denoised calcium traces (same shape as input)
2. **`F_spikes.npy`** - Inferred spike activity (same shape as input)  
3. **`F_deconv_log.txt`** - Processing log with parameters and success rates

## ⚙️ jRGECO1a Parameters

The script uses optimized parameters for jRGECO1a:
- **Decay time constant (τ)**: 1.35 seconds
- **Sampling rate**: 2.6 Hz (configurable)
- **Default g parameter**: 0.752 (calculated as `exp(-1/(τ×sampling_rate))`)
- **Penalty**: L0 (sparser spike inference)
- **Optimization**: Uses 3 isolated events to refine parameters

## 🔧 Processing Strategy

**Option B Implementation** (as requested):
1. **Primary**: Try jRGECO1a defaults first
2. **Fallback**: If jRGECO1a defaults fail, use full auto-estimation
3. **Error handling**: If both fail, output NaN arrays and log error

## 📈 Example Usage

### Testing with Sample Data
```bash
# Generate test data
python test_oasis_batch.py

# Process with GUI
python oasis_deconvolve_batch.py
# (Select test_calcium_data.npy in the file dialog)

# Or process with CLI
python oasis_deconvolve_cli.py test_calcium_data.npy
```

### Processing Your Real Data
```bash
# Replace with your actual file path
python oasis_deconvolve_cli.py "E:\\Rasmus-Guillermo\\ECF1\\F1_RV_pdf\\New recordings PFC\\LED 2s\\240916_pl100_pc001_LED_min10_ex02\\DATA\\SUPPORT_ChanA\\derippled\\suite2p\\plane0\\F.npy"
```

Expected output files in the same directory:
- `F_deconv.npy`
- `F_spikes.npy` 
- `F_deconv_log.txt`

## 📋 Requirements

- Python 3.6+
- NumPy
- SciPy
- Matplotlib
- Cython
- tkinter (for GUI version)
- OASIS package (installed and compiled)

## 🐛 Troubleshooting

### Common Issues

**1. Import Errors**
```
ModuleNotFoundError: No module named 'oasis'
```
**Solution**: Make sure OASIS is installed and compiled:
```bash
pip install -r requirements.txt
python setup.py build_ext --inplace
```

**2. File Format Issues**
```
ValueError: Expected 2D array, got 1D array
```
**Solution**: Ensure your data is 2D. If you have a 1D trace, reshape it:
```python
data_1d = np.load('single_trace.npy')
data_2d = data_1d.reshape(1, -1)  # Convert to 2D
np.save('reshaped_data.npy', data_2d)
```

**3. Memory Issues with Large Files**
If processing very large files (>1GB), consider:
- Processing subsets of traces
- Using the CLI version (lower memory overhead)
- Increasing system virtual memory

### Error Handling

- **Failed traces**: Output as NaN arrays, logged in detail
- **Processing errors**: Full error traceback in console/log
- **File I/O errors**: Clear error messages with suggested solutions

## 📚 Understanding the Output

### Deconvolved Traces (`*_deconv.npy`)
- **Content**: Denoised calcium fluorescence
- **Interpretation**: Smooth calcium dynamics with noise removed
- **Units**: Same as input (ΔF/F or raw fluorescence)

### Spike Inference (`*_spikes.npy`)
- **Content**: Inferred neural activity (spikes)
- **Interpretation**: Discrete events representing likely action potentials
- **Units**: Relative spike amplitude (higher values = stronger evidence)

### Processing Log (`*_deconv_log.txt`)
Example log entry:
```
Trace    0: SUCCESS - Method: jRGECO1a_defaults, g=0.710, baseline=2.332, lambda=1.021
```
- **g**: Decay parameter (lower = faster decay)
- **baseline**: Estimated fluorescence baseline
- **lambda**: Regularization parameter (higher = sparser spikes)

## 🔬 Method Details

The OASIS algorithm solves:
```
min |s|₀ subject to |c-y|² ≤ σ²T and s = Gc ≥ 0
```

Where:
- `y`: Input fluorescence trace
- `c`: Denoised calcium trace  
- `s`: Spike train
- `G`: Convolution matrix (AR model)
- `σ`: Noise standard deviation
- `T`: Number of timepoints

For jRGECO1a, we use AR(1) model: `c[t] = g·c[t-1] + s[t]`

## 📖 References

- Friedrich J, Paninski L. Fast Active Set Methods for Online Spike Inference from Calcium Imaging. NIPS 2016.
- Friedrich J, Zhou P, Paninski L. Fast Online Deconvolution of Calcium Imaging Data. PLoS Computational Biology 2017.

## 📞 Support

For issues specific to this batch processing implementation, check:
1. Input data format (2D NumPy array)
2. OASIS installation (compiled Cython extensions)
3. File permissions (write access to output directory)
4. Processing log for detailed error messages

For OASIS algorithm questions, refer to the original OASIS documentation and papers.

