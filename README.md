# CDK2 Activity Analysis Pipeline

A comprehensive Python-based tool for analyzing CDK2 activity in time-lapse microscopy images, based on the methodologies from Cappell et al., 2016 and the CDK2 activity calculation pipeline.

## Overview

This pipeline combines cell tracking algorithms with CDK2 activity calculation to provide comprehensive analysis of cell cycle dynamics. The tool processes time-lapse microscopy images to:

1. **Track individual cells** across multiple time frames
2. **Calculate CDK2 activity** based on fluorescence intensity ratios
3. **Analyze cell cycle dynamics** and activity patterns
4. **Generate comprehensive visualizations** and statistical reports

## Features

- **Automated cell detection and tracking** using watershed segmentation
- **CDK2 activity calculation** based on fluorescence intensity analysis
- **Statistical analysis** including clustering and transition detection
- **Comprehensive visualizations** with multiple plot types
- **Export capabilities** for further analysis
- **Configurable parameters** for different experimental setups

## Installation

### Prerequisites

- Python 3.7 or higher
- Required packages (install via pip):

```bash
pip install numpy matplotlib pandas opencv-python scipy scikit-image seaborn scikit-learn pillow
```

### Setup

1. Clone or download the pipeline files
2. Ensure your time-lapse images are in the `CDK2 activity` folder
3. Run the main analysis script

## Usage

### Basic Analysis

Run the main CDK2 activity analysis:

```bash
python cdk2_activity_pipeline.py
```

This will:
- Load and sort time-lapse images
- Track cells across frames
- Calculate CDK2 activity
- Generate visualizations
- Save results to `CDK2_Analysis_Results/`

### Advanced Analysis

For additional statistical analysis and clustering:

```bash
python advanced_cdk2_analysis.py
```

### Configuration

Modify `config.py` to adjust analysis parameters:

```python
# Image processing parameters
IMAGE_PROCESSING = {
    'min_cell_area': 100,
    'max_cell_area': 2000,
    'gaussian_blur_sigma': 1.0
}

# CDK2 activity parameters
CDK2_ACTIVITY = {
    'background_threshold': 0.1,
    'signal_threshold': 0.3,
    'normalization_method': 'min_max'
}
```

## Input Data Format

### Image Naming Convention

The pipeline expects time-lapse images with timestamps in the filename:
- Format: `VID211_B4_5_01d23h30m.jpg`
- Pattern: `(\d+)d(\d+)h(\d+)m`
- Example: `01d23h30m` = 1 day, 23 hours, 30 minutes

### Supported Image Formats

- JPG/JPEG
- PNG
- TIFF/TIF

## Output Files

## Web Deployment

This application can be deployed as a web service for easy access and sharing.

### Quick Deployment

1. **Run the deployment helper:**
   ```bash
   ./deploy.sh
   ```

2. **Deploy to Render (Recommended - Free):**
   - Go to [render.com](https://render.com)
   - Sign up and create a new Web Service
   - Connect your Git repository
   - Set Build Command: `pip install -r requirements.txt`
   - Set Start Command: `python app.py`
   - Deploy!

3. **Access your app:**
   - Your app will be available at: `https://your-app-name.onrender.com`
   - Share this link with others to access your cell tracking application

### Alternative Platforms

- **Railway**: [railway.app](https://railway.app) - Free tier available
- **Heroku**: [heroku.com](https://heroku.com) - Paid service
- **PythonAnywhere**: [pythonanywhere.com](https://pythonanywhere.com) - Free tier available

For detailed deployment instructions, see [DEPLOYMENT.md](DEPLOYMENT.md).

### Web App Features

Once deployed, users can:
- Upload their own cell images to two channels (nucleus only and nucleus+cytoplasm)
- View cell images in a web browser
- Select random cells for analysis
- Measure whiteness in selected regions
- Track whiteness changes over time
- Download analysis results as Excel files
- Access the application from any device with a web browser
- Clear uploaded images when needed

### Main Analysis Results

- `cdk2_activity_analysis.png` - Comprehensive visualization plots
- `cdk2_activity_data.csv` - Raw data with cell measurements
- `analysis_summary.txt` - Statistical summary

### Advanced Analysis Results

- `advanced_cdk2_analysis.png` - Advanced statistical plots
- `cdk2_analysis_report.txt` - Detailed analysis report

## Analysis Components

### 1. Cell Detection and Tracking

- **Preprocessing**: Gaussian blur and normalization
- **Segmentation**: Watershed algorithm for cell separation
- **Tracking**: Hungarian algorithm for optimal cell tracking
- **Filtering**: Remove small/large objects and track fragments

### 2. CDK2 Activity Calculation

- **Intensity Analysis**: Extract fluorescence intensity from detected cells
- **Activity Scoring**: Calculate CDK2 activity based on intensity ratios
- **Normalization**: Apply background correction and normalization
- **Categorization**: Classify cells into Low/Medium/High activity groups

### 3. Statistical Analysis

- **Basic Statistics**: Mean, median, standard deviation, percentiles
- **Distribution Analysis**: Histograms, Q-Q plots, cumulative distributions
- **Correlation Analysis**: Activity vs. intensity, area, and other parameters
- **Clustering**: K-means clustering to identify cell populations

### 4. Visualization

- **Activity Distribution**: Histograms and density plots
- **Scatter Plots**: Activity vs. intensity, area, and other parameters
- **Correlation Heatmaps**: Relationship between different parameters
- **Category Analysis**: Box plots and violin plots by activity level
- **Statistical Summaries**: Comprehensive statistical overview

## Methodology

### CDK2 Activity Calculation

The CDK2 activity is calculated based on fluorescence intensity analysis:

1. **Background Correction**: Subtract background fluorescence
2. **Signal Normalization**: Normalize to account for experimental variations
3. **Activity Scoring**: Calculate activity based on intensity ratios
4. **Categorization**: Classify cells into activity categories

### Cell Tracking Algorithm

Based on Cappell et al., 2016 methodology:

1. **Cell Detection**: Watershed segmentation with morphological operations
2. **Feature Extraction**: Area, intensity, centroid, and shape properties
3. **Cost Matrix**: Calculate tracking costs based on distance and intensity changes
4. **Optimal Assignment**: Hungarian algorithm for optimal cell tracking
5. **Track Filtering**: Remove short tracks and outliers

## Configuration Options

### Image Processing

- `min_cell_area`: Minimum cell area for detection
- `max_cell_area`: Maximum cell area for detection
- `gaussian_blur_sigma`: Blur parameter for noise reduction
- `eccentricity_threshold`: Maximum eccentricity for cell filtering

### Tracking Parameters

- `tracking_threshold`: Maximum distance for cell tracking
- `min_track_length`: Minimum number of frames for valid tracks
- `intensity_weight`: Weight for intensity changes in cost calculation

### CDK2 Activity

- `background_threshold`: Background fluorescence threshold
- `signal_threshold`: Signal fluorescence threshold
- `normalization_method`: Method for activity normalization
- `activity_categories`: Thresholds for activity categorization

## Troubleshooting

### Common Issues

1. **No cells detected**: Adjust `min_cell_area` and `max_cell_area` parameters
2. **Poor tracking**: Increase `tracking_threshold` or adjust intensity weights
3. **Low activity scores**: Check `background_threshold` and `signal_threshold`
4. **Memory issues**: Reduce image resolution or process fewer frames

### Performance Optimization

- Use smaller image sizes for faster processing
- Adjust tracking parameters based on cell density
- Process images in batches for large datasets

## References

- Cappell, S.D., et al. (2016). Cell 166, 167-180
- CDK2 Activity Calculation Pipeline Documentation
- Nature Communications CDK2 Analysis Protocol

## License

This pipeline is provided for research purposes. Please cite the original papers when using this tool.

## Support

For questions or issues:
1. Check the configuration parameters
2. Verify input data format
3. Review error messages in the console output
4. Ensure all required packages are installed

## Example Output

The pipeline generates comprehensive analysis including:

- **Cell tracking results**: Individual cell trajectories
- **CDK2 activity measurements**: Normalized activity scores
- **Statistical summaries**: Mean, median, percentiles
- **Visualization plots**: Multiple plot types for analysis
- **Export files**: CSV data and text reports

This enables detailed analysis of cell cycle dynamics and CDK2 activity patterns in your experimental data. 