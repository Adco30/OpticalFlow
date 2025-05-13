# OpticalFlow Visualization and Optimization System

A tool for optical flow computation, visualization, and parameter optimization using Bayesian optimization.

## Features

- **Lucas-Kanade Optical Flow**: Basic implementation with configurable parameters
- **Hierarchical Lucas-Kanade**: Multi-level pyramid approach for large displacements
- **Parameter Optimization**: Bayesian optimization for finding optimal flow parameters
- **Motion Region Detection**: Automatic detection of moving blocks in image sequences
- **Interactive Visualization**: Real-time parameter adjustment with visual feedback
- **Row-based Parameter Management**: Different parameters for different image pairs

## Architecture

```
OpticalFlow/
├── app.py                     # Main application entry point
├── core/
│   ├── flow_algorithms.py     # Optical flow implementations
│   └── image_operations.py    # Image preprocessing functions
├── analysis/
│   ├── action_detector.py     # Motion pattern detection
│   └── movement_analyzer.py   # Motion metrics calculation
├── optimization/
│   └── parameter_tuner.py     # Bayesian parameter optimization
├── ui/
│   ├── main_window.py         # Main application window
│   └── widgets.py             # Custom UI components
├── utils/
│   ├── config.py              # Parameter configuration
│   └── video_loader.py        # Video loading and visualization
└── input_images/              # Test image directory
```

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place test images in the `input_images/TestSeq` directory

## Usage

Run the application:
```bash
python app.py
```

### Key Features in the UI:

- **Automatic Optimization**: The system automatically runs Bayesian optimization to find optimal parameters
- **Row Selection**: Click on any row to select it and adjust parameters for that specific image pair
- **Real-time Updates**: Parameter changes are immediately applied to the selected row
- **Visualization Options**: Toggle between overlay mode and separate displays
- **Vector Displays**: Text representation of flow vectors with motion region highlighting

### Parameter Categories:

- **Preprocessing**: Gamma, Gaussian blur, bilateral filtering
- **LK Algorithm**: Window size, temporal kernel, determinant threshold
- **Hierarchical**: Pyramid levels for multi-scale processing
- **Visualization**: Grid spacing, vector scaling, motion thresholds

## Algorithm Overview

1. **Image Preprocessing**: Gamma correction and filtering
2. **Motion Region Detection**: Automatic detection of moving blocks
3. **Parameter Optimization**: Bayesian optimization across multiple image pairs
4. **Flow Computation**: Lucas-Kanade or Hierarchical Lucas-Kanade
5. **Visualization**: Flow vectors with motion region highlighting

## Parameters

- `gamma`: Gamma correction factor (0.5-2.5)
- `gaussian_sigma`: Standard deviation for Gaussian blur (0.1-5.0)
- `window_size`: Window size for Lucas-Kanade (5-41)
- `pyramid_levels`: Number of pyramid levels for hierarchical approach (1-6)
- `motion_threshold`: Threshold for significant motion detection (0.05-0.5)

## Optimization Process

The Bayesian optimization process:
1. Evaluates each parameter set across all image pairs
2. Calculates quality scores based on:
   - Minimal outliers outside motion regions
   - Consistent displacement vectors within regions
   - Uniform direction and magnitude within regions
3. Updates the search space based on results
4. Converges to optimal parameters for the test set

## Quality Metrics

The system evaluates flow quality using:
- Outlier penalty (proportion of vectors outside detected regions)
- Coverage score (proportion of significant vectors in motion regions)
- Direction consistency (uniformity of vector directions)
- Magnitude consistency (uniformity of vector magnitudes)
