# 🔬 Cell Image Viewer Web Application

A Flask-based web application that randomly selects cell nucleus images from the `ch00 2` folder and displays the corresponding nucleus+cytoplasm images from the `ch002` folder.

## Features

- **Random Cell Selection**: Pick a random cell nucleus image and view its corresponding nucleus+cytoplasm image
- **Image Pairing**: Automatically matches images between the two channels based on their base names
- **Modern UI**: Beautiful, responsive web interface with gradient backgrounds and smooth animations
- **Image Information**: Displays timepoint and location information for each image
- **Statistics**: Shows total number of available image pairs

## Folder Structure

The application expects the following folder structure:
```
.
├── ch00 2/          # Nucleus only images (e.g., 1_1_1_t471_ch00.tif)
├── ch002/           # Nucleus + cytoplasm images (e.g., 1_1_1_t471_ch02.tif)
├── app.py           # Flask application
├── templates/       # HTML templates
└── requirements.txt # Python dependencies
```

## Image Naming Convention

Images should follow this naming pattern:
- `{row}_{column}_{site}_t{timepoint}_ch{channel}.tif`
- Example: `1_1_1_t471_ch00.tif` (nucleus) and `1_1_1_t471_ch02.tif` (nucleus+cytoplasm)

## Installation

1. **Create a virtual environment**:
   ```bash
   python3 -m venv cell_viewer_env
   source cell_viewer_env/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Start the application**:
   ```bash
   source cell_viewer_env/bin/activate
   python app.py
   ```

2. **Open your web browser** and navigate to:
   ```
   http://localhost:5001
   ```

3. **Use the interface**:
   - Click "🎲 Pick Random Cell" to select a random cell image pair
   - Click "📋 View All Cells" to see statistics and load a random cell from all available pairs
   - The left panel shows the nucleus-only image (ch00)
   - The right panel shows the nucleus+cytoplasm image (ch002)

## API Endpoints

- `GET /` - Main web interface
- `GET /api/random-cell` - Get a random cell image pair
- `GET /api/all-cells` - Get all available cell image pairs
- `GET /images/ch00/<filename>` - Serve nucleus images
- `GET /images/ch002/<filename>` - Serve nucleus+cytoplasm images

## Technical Details

- **Framework**: Flask 2.3.3
- **Port**: 5001 (to avoid conflicts with macOS AirPlay)
- **Image Format**: TIFF files
- **Responsive Design**: Works on desktop and mobile devices

## Troubleshooting

- **Port 5000 in use**: The app uses port 5001 to avoid conflicts with macOS AirPlay
- **No images found**: Ensure both `ch00 2` and `ch002` folders exist and contain matching TIFF files
- **Image loading errors**: Check that image files follow the correct naming convention

## Example Output

When you run the application, you should see:
```
Found 150 image pairs
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://0.0.0.0:5001
```

The web interface will display:
- A beautiful gradient header with the application title
- Two buttons for random selection and viewing all cells
- Two image cards showing the nucleus and nucleus+cytoplasm images
- Information about timepoint and location for each image
