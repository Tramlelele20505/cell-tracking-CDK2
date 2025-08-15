#!/usr/bin/env python3
"""
Cell Image Viewer Web Application
Randomly selects cell nucleus images and shows corresponding nucleus+cytoplasm images
"""

from flask import Flask, render_template, jsonify, request, send_from_directory, Response
import os
import random
import re
from pathlib import Path
from collections import OrderedDict
import cv2
import numpy as np
from PIL import Image
import io
from io import BytesIO
import requests
import openpyxl
from openpyxl import Workbook
from datetime import datetime

app = Flask(__name__)

# Configuration
CH00_FOLDER = "ch00 2"  # Nucleus only images
CH002_FOLDER = "ch002"  # Nucleus + cytoplasm images
STATIC_FOLDER = "static"

# Create static folder if it doesn't exist
os.makedirs(STATIC_FOLDER, exist_ok=True)

def natural_sort_key(s):
    """
    Natural sorting key function that sorts numbers numerically
    """
    import re
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

def get_image_files():
    """
    Get all image files from both folders and organize them by their base name
    Uses natural sorting to ensure proper chronological order
    """
    ch00_files = []
    ch002_files = []
    
    # Get ALL ch00 files (nucleus only) with natural sorting
    if os.path.exists(CH00_FOLDER):
        ch00_files = sorted(
            [file for file in os.listdir(CH00_FOLDER) 
             if file.endswith('.tif') and '_ch00.tif' in file],
            key=natural_sort_key
        )
        print(f"Found {len(ch00_files)} ch00 files")
    
    # Get ALL ch002 files (nucleus + cytoplasm) with natural sorting
    if os.path.exists(CH002_FOLDER):
        ch002_files = sorted(
            [file for file in os.listdir(CH002_FOLDER) 
             if file.endswith('.tif') and '_ch02.tif' in file],
            key=natural_sort_key
        )
        print(f"Found {len(ch002_files)} ch002 files")
    
    # Create a mapping of base names to full file names
    # Use OrderedDict to preserve the natural sorted order
    image_pairs = OrderedDict()
    
    # Process files in the natural sorted order
    for ch00_file in ch00_files:
        # Extract the base name (everything before _ch00.tif)
        base_name = ch00_file.replace('_ch00.tif', '')
        
        # Find corresponding ch002 file
        ch002_file = f"{base_name}_ch02.tif"
        
        if ch002_file in ch002_files:
            image_pairs[base_name] = {
                'nucleus': ch00_file,
                'nucleus_cytoplasm': ch002_file
            }
        else:
            print(f"Warning: No matching ch002 file for {ch00_file}")
    
    print(f"Successfully paired {len(image_pairs)} file pairs")
    return image_pairs

def get_timepoints():
    """
    Get all available timepoints - each file is its own timepoint
    """
    image_pairs = get_image_files()
    timepoints = []
    
    # Create a list of all base names in original folder order
    base_names = list(image_pairs.keys())
    
    # Keep original order - no sorting
    # Each base_name becomes a timepoint index
    for i, base_name in enumerate(base_names):
        timepoints.append(i)
    
    return timepoints

def parse_image_info(filename):
    """
    Parse image filename to extract row, column, site, and timepoint
    Format: {row}_{column}_{site}_t{timepoint}_ch{channel}.tif
    """
    pattern = r'(\d+)_(\d+)_(\d+)_t(\d+)_ch(\d+)\.tif'
    match = re.match(pattern, filename)
    
    if match:
        row = int(match.group(1))
        column = int(match.group(2))
        site = int(match.group(3))
        timepoint = int(match.group(4))
        channel = int(match.group(5))
        
        return {
            'row': row,
            'column': column,
            'site': site,
            'timepoint': timepoint,
            'channel': channel,
            'formatted_time': f"Timepoint {timepoint}",
            'location': f"Row {row}, Column {column}, Site {site}"
        }
    
    return None

@app.route('/')
def index():
    """
    Main page with random cell selection
    """
    return render_template('index.html')

@app.route('/api/random-cell')
def get_random_cell():
    """
    API endpoint to get a random cell image pair
    """
    image_pairs = get_image_files()
    
    if not image_pairs:
        return jsonify({
            'error': 'No image pairs found',
            'message': 'Please check that both ch00 and ch002 folders contain matching images.'
        }), 404
    
    # Select a random base name
    base_name = random.choice(list(image_pairs.keys()))
    pair = image_pairs[base_name]
    
    # Parse image information
    nucleus_info = parse_image_info(pair['nucleus'])
    cytoplasm_info = parse_image_info(pair['nucleus_cytoplasm'])
    
    return jsonify({
        'base_name': base_name,
        'nucleus': {
            'filename': pair['nucleus'],
            'path': f'/images/ch00/{pair["nucleus"]}',
            'info': nucleus_info
        },
        'nucleus_cytoplasm': {
            'filename': pair['nucleus_cytoplasm'],
            'path': f'/images/ch002/{pair["nucleus_cytoplasm"]}',
            'info': cytoplasm_info
        }
    })

@app.route('/api/all-cells')
def get_all_cells():
    """
    API endpoint to get all available cell image pairs
    """
    image_pairs = get_image_files()
    
    cells = []
    for base_name, pair in image_pairs.items():
        nucleus_info = parse_image_info(pair['nucleus'])
        cytoplasm_info = parse_image_info(pair['nucleus_cytoplasm'])
        
        cells.append({
            'base_name': base_name,
            'nucleus': {
                'filename': pair['nucleus'],
                'path': f'/images/ch00/{pair["nucleus"]}',
                'info': nucleus_info
            },
            'nucleus_cytoplasm': {
                'filename': pair['nucleus_cytoplasm'],
                'path': f'/images/ch002/{pair["nucleus_cytoplasm"]}',
                'info': cytoplasm_info
            }
        })
    
    # Sort by timepoint for better organization
    cells.sort(key=lambda x: x['nucleus']['info']['timepoint'] if x['nucleus']['info'] else 0)
    
    return jsonify({
        'total_cells': len(cells),
        'cells': cells
    })

@app.route('/api/timepoints')
def get_timepoints_api():
    """
    API endpoint to get all available timepoints
    """
    timepoints = get_timepoints()
    image_pairs = get_image_files()
    
    # Get base names in original folder order
    base_names = list(image_pairs.keys())
    
    print(f"Found {len(timepoints)} timepoints (files): {len(base_names)} total files")
    print(f"First 10 files: {base_names[:10]}")
    print(f"Last 10 files: {base_names[-10:]}")
    
    # Debug: Show natural sorted order
    print("Natural sorted order verification...")
    ch00_files_natural = sorted(
        [file for file in os.listdir(CH00_FOLDER) 
         if file.endswith('.tif') and '_ch00.tif' in file],
        key=natural_sort_key
    )
    ch00_base_names_natural = [file.replace('_ch00.tif', '') for file in ch00_files_natural]
    print(f"First 10 natural sorted ch00 files: {ch00_base_names_natural[:10]}")
    print(f"First 10 app files: {base_names[:10]}")
    print(f"Natural order matches: {ch00_base_names_natural[:10] == base_names[:10]}")
    
    return jsonify({
        'timepoints': timepoints,
        'total_timepoints': len(timepoints),
        'total_files': len(base_names)
    })

@app.route('/api/cells-by-timepoint/<int:timepoint>')
def get_cells_by_timepoint(timepoint):
    """
    API endpoint to get the specific file pair for a timepoint index
    """
    image_pairs = get_image_files()
    
    # Get base names in original folder order (same order as get_timepoints)
    base_names = list(image_pairs.keys())
    
    # Get the base_name for this timepoint index
    if timepoint < len(base_names):
        base_name = base_names[timepoint]
        pair = image_pairs[base_name]
        
        nucleus_info = parse_image_info(pair['nucleus'])
        cytoplasm_info = parse_image_info(pair['nucleus_cytoplasm'])
        
        cell = {
            'base_name': base_name,
            'nucleus': {
                'filename': pair['nucleus'],
                'path': f'/images/ch00/{pair["nucleus"]}',
                'info': nucleus_info
            },
            'nucleus_cytoplasm': {
                'filename': pair['nucleus_cytoplasm'],
                'path': f'/images/ch002/{pair["nucleus_cytoplasm"]}',
                'info': cytoplasm_info
            }
        }
        
        return jsonify({
            'timepoint': timepoint,
            'total_cells': 1,
            'cells': [cell]
        })
    else:
        return jsonify({
            'timepoint': timepoint,
            'total_cells': 0,
            'cells': []
        })

@app.route('/images/ch00/<filename>')
def serve_ch00_image(filename):
    """
    Serve ch00 (nucleus only) images as JPEG
    """
    try:
        # Load TIFF image
        image_path = os.path.join(CH00_FOLDER, filename)
        if not os.path.exists(image_path):
            return "Image not found", 404
        
        # Read image with OpenCV
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return "Failed to load image", 500
        
        # Normalize and enhance for better visibility
        img_normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        
        # Apply slight contrast enhancement
        img_enhanced = cv2.convertScaleAbs(img_normalized, alpha=1.2, beta=10)
        
        # Convert to JPEG
        _, buffer = cv2.imencode('.jpg', img_enhanced, [cv2.IMWRITE_JPEG_QUALITY, 90])
        
        # Create response
        response = Response(buffer.tobytes(), mimetype='image/jpeg')
        response.headers['Cache-Control'] = 'public, max-age=31536000'
        return response
        
    except Exception as e:
        print(f"Error serving ch00 image {filename}: {e}")
        return "Error processing image", 500

@app.route('/images/ch002/<filename>')
def serve_ch002_image(filename):
    """
    Serve ch002 (nucleus + cytoplasm) images as JPEG
    """
    try:
        # Load TIFF image
        image_path = os.path.join(CH002_FOLDER, filename)
        if not os.path.exists(image_path):
            return "Image not found", 404
        
        # Read image with OpenCV
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return "Failed to load image", 500
        
        # Normalize and enhance for better visibility
        img_normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        
        # Apply slight contrast enhancement
        img_enhanced = cv2.convertScaleAbs(img_normalized, alpha=1.2, beta=10)
        
        # Convert to JPEG
        _, buffer = cv2.imencode('.jpg', img_enhanced, [cv2.IMWRITE_JPEG_QUALITY, 90])
        
        # Create response
        response = Response(buffer.tobytes(), mimetype='image/jpeg')
        response.headers['Cache-Control'] = 'public, max-age=31536000'
        return response
        
    except Exception as e:
        print(f"Error serving ch002 image {filename}: {e}")
        return "Error processing image", 500

@app.route('/debug/test-image/<filename>')
def test_image_loading(filename):
    """
    Debug endpoint to test image loading
    """
    try:
        # Try to load from both folders
        ch00_path = os.path.join(CH00_FOLDER, filename)
        ch002_path = os.path.join(CH002_FOLDER, filename)
        
        result = {
            'filename': filename,
            'ch00_exists': os.path.exists(ch00_path),
            'ch002_exists': os.path.exists(ch002_path),
            'ch00_size': os.path.getsize(ch00_path) if os.path.exists(ch00_path) else 0,
            'ch002_size': os.path.getsize(ch002_path) if os.path.exists(ch002_path) else 0
        }
        
        # Try to load with OpenCV
        if os.path.exists(ch00_path):
            img = cv2.imread(ch00_path, cv2.IMREAD_GRAYSCALE)
            result['ch00_loadable'] = img is not None
            if img is not None:
                result['ch00_shape'] = img.shape
                result['ch00_dtype'] = str(img.dtype)
        
        if os.path.exists(ch002_path):
            img = cv2.imread(ch002_path, cv2.IMREAD_GRAYSCALE)
            result['ch002_loadable'] = img is not None
            if img is not None:
                result['ch002_shape'] = img.shape
                result['ch002_dtype'] = str(img.dtype)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/cell/<base_name>')
def view_specific_cell(base_name):
    """
    View a specific cell by base name
    """
    image_pairs = get_image_files()
    
    if base_name not in image_pairs:
        return jsonify({'error': 'Cell not found'}), 404
    
    pair = image_pairs[base_name]
    nucleus_info = parse_image_info(pair['nucleus'])
    cytoplasm_info = parse_image_info(pair['nucleus_cytoplasm'])
    
    return render_template('cell_view.html', 
                         base_name=base_name,
                         nucleus=pair['nucleus'],
                         nucleus_cytoplasm=pair['nucleus_cytoplasm'],
                         nucleus_info=nucleus_info,
                         cytoplasm_info=cytoplasm_info)


def measure_whiteness(image_path, selection):
    """
    Measures average whiteness of the selected box in the image.
    Whiteness is computed as the mean of grayscale intensities (0–255).
    """
    if not os.path.exists(image_path):
        return None, "Image not found"

    try:
        img = Image.open(image_path).convert("L")  # Convert to grayscale
        width, height = img.size

        # Convert percentage coords to pixels
        left = int((selection["x"] / 100) * width)
        top = int((selection["y"] / 100) * height)
        box_width = int((selection["width"] / 100) * width)
        box_height = int((selection["height"] / 100) * height)
        right = left + box_width
        bottom = top + box_height

        cropped = img.crop((left, top, right, bottom))
        arr = np.array(cropped, dtype=np.float32)

        average_whiteness = float(np.mean(arr))  # 0–255
        return average_whiteness, None
    except Exception as e:
        return None, str(e)
@app.route('/api/measure-both-whiteness', methods=['POST'])
def measure_both_whiteness():
    data = request.json
    nucleus_path = data.get("nucleusPath")
    cytoplasm_path = data.get("cytoplasmPath")
    nucleus_selection = data.get("nucleusSelection")
    cytoplasm_selection = data.get("cytoplasmSelection")

    # Strip domain if URL passed
    if nucleus_path.startswith("http"):
        nucleus_path = nucleus_path.replace(request.host_url, "")
    if cytoplasm_path.startswith("http"):
        cytoplasm_path = cytoplasm_path.replace(request.host_url, "")

    # Extract filename and build absolute path
    nucleus_filename = os.path.basename(nucleus_path)
    cytoplasm_filename = os.path.basename(cytoplasm_path)
    nucleus_abs = os.path.join(CH00_FOLDER, nucleus_filename)
    cytoplasm_abs = os.path.join(CH002_FOLDER, cytoplasm_filename)

    # Check files exist
    if not os.path.exists(nucleus_abs):
        return jsonify({"error": f"Nucleus image not found at {nucleus_abs}"}), 400
    if not os.path.exists(cytoplasm_abs):
        return jsonify({"error": f"Cytoplasm image not found at {cytoplasm_abs}"}), 400

    # Load the right image (ch002) as grayscale - this is the only image we need for whiteness calculation
    cytoplasm_img = cv2.imread(cytoplasm_abs, cv2.IMREAD_GRAYSCALE)
    if cytoplasm_img is None:
        return jsonify({"error": "Failed to read cytoplasm image"}), 400

    results = {}

    # Process nucleus selection (same coordinates applied to right image)
    if nucleus_selection:
        h, w = cytoplasm_img.shape
        x = int(nucleus_selection["x"] / 100 * w)
        y = int(nucleus_selection["y"] / 100 * h)
        width = int(nucleus_selection["width"] / 100 * w)
        height = int(nucleus_selection["height"] / 100 * h)

        # Crop nucleus from the right image (ch002)
        nucleus_crop = cytoplasm_img[y:y+height, x:x+width]

        # Calculate whiteness for nucleus selection
        if nucleus_crop.size > 0:
            # Calculate average pixel intensity for normalization
            avg_pixel_intensity = np.mean(nucleus_crop)
            
            # Whiteness = grayscale / average pixel intensity
            nucleus_whiteness = float(np.mean(nucleus_crop)) / avg_pixel_intensity if avg_pixel_intensity > 0 else 0.0
            
            results["nucleus_selection"] = {
                "whiteness": nucleus_whiteness,
                "avg_pixel_intensity": float(avg_pixel_intensity),
                "mean_grayscale": float(np.mean(nucleus_crop))
            }

    # Process cytoplasm selection (directly on right image)
    if cytoplasm_selection:
        h, w = cytoplasm_img.shape
        x = int(cytoplasm_selection["x"] / 100 * w)
        y = int(cytoplasm_selection["y"] / 100 * h)
        width = int(cytoplasm_selection["width"] / 100 * w)
        height = int(cytoplasm_selection["height"] / 100 * h)

        # Crop cytoplasm from the right image (ch002)
        cytoplasm_crop = cytoplasm_img[y:y+height, x:x+width]

        # Calculate whiteness for cytoplasm selection
        if cytoplasm_crop.size > 0:
            # Calculate average pixel intensity for normalization
            avg_pixel_intensity = np.mean(cytoplasm_crop)
            
            # Whiteness = grayscale / average pixel intensity
            cytoplasm_whiteness = float(np.mean(cytoplasm_crop)) / avg_pixel_intensity if avg_pixel_intensity > 0 else 0.0
            
            results["cytoplasm_selection"] = {
                "whiteness": cytoplasm_whiteness,
                "avg_pixel_intensity": float(avg_pixel_intensity),
                "mean_grayscale": float(np.mean(cytoplasm_crop))
            }

    return jsonify(results)


@app.route('/api/auto-track', methods=['POST'])
def auto_track():
    """
    Automatically track whiteness through a sequence of timepoints
    """
    data = request.json
    start_timepoint = data.get("startTimepoint")
    end_timepoint = data.get("endTimepoint")
    nucleus_selection = data.get("nucleusSelection")
    cytoplasm_selection = data.get("cytoplasmSelection")
    
    if not all([start_timepoint, end_timepoint, nucleus_selection, cytoplasm_selection]):
        return jsonify({"error": "Missing required parameters"}), 400
    
    # Get all timepoints
    image_pairs = get_image_files()
    timepoints = list(image_pairs.keys())
    
    if start_timepoint >= len(timepoints) or end_timepoint >= len(timepoints):
        return jsonify({"error": "Invalid timepoint range"}), 400
    
    tracking_results = []
    
    # Process each timepoint in the range
    for i in range(start_timepoint, end_timepoint + 1):
        try:
            # Get the image pair for this timepoint
            base_name = timepoints[i]
            nucleus_filename = f"{base_name}_ch00.tif"
            cytoplasm_filename = f"{base_name}_ch02.tif"
            
            nucleus_abs = os.path.join(CH00_FOLDER, nucleus_filename)
            cytoplasm_abs = os.path.join(CH002_FOLDER, cytoplasm_filename)
            
            if not os.path.exists(nucleus_abs) or not os.path.exists(cytoplasm_abs):
                print(f"Warning: Missing files for timepoint {i}")
                continue
            
            # Load the right image (ch002) as grayscale
            cytoplasm_img = cv2.imread(cytoplasm_abs, cv2.IMREAD_GRAYSCALE)
            if cytoplasm_img is None:
                print(f"Warning: Failed to read cytoplasm image for timepoint {i}")
                continue
            
            # Calculate whiteness for nucleus selection
            h, w = cytoplasm_img.shape
            x = int(nucleus_selection["x"] / 100 * w)
            y = int(nucleus_selection["y"] / 100 * h)
            width = int(nucleus_selection["width"] / 100 * w)
            height = int(nucleus_selection["height"] / 100 * h)
            
            nucleus_crop = cytoplasm_img[y:y+height, x:x+width]
            nucleus_whiteness = 0.0
            if nucleus_crop.size > 0:
                avg_pixel_intensity = np.mean(nucleus_crop)
                nucleus_whiteness = float(np.mean(nucleus_crop)) / avg_pixel_intensity if avg_pixel_intensity > 0 else 0.0
            
            # Calculate whiteness for cytoplasm selection
            x = int(cytoplasm_selection["x"] / 100 * w)
            y = int(cytoplasm_selection["y"] / 100 * h)
            width = int(cytoplasm_selection["width"] / 100 * w)
            height = int(cytoplasm_selection["height"] / 100 * h)
            
            cytoplasm_crop = cytoplasm_img[y:y+height, x:x+width]
            cytoplasm_whiteness = 0.0
            if cytoplasm_crop.size > 0:
                avg_pixel_intensity = np.mean(cytoplasm_crop)
                cytoplasm_whiteness = float(np.mean(cytoplasm_crop)) / avg_pixel_intensity if avg_pixel_intensity > 0 else 0.0
            
            # Store results
            tracking_results.append({
                "timepoint": i,
                "filename": base_name,
                "nucleus_whiteness": nucleus_whiteness,
                "cytoplasm_whiteness": cytoplasm_whiteness,
                "nucleus_mean_grayscale": float(np.mean(nucleus_crop)) if nucleus_crop.size > 0 else 0.0,
                "cytoplasm_mean_grayscale": float(np.mean(cytoplasm_crop)) if cytoplasm_crop.size > 0 else 0.0
            })
            
            print(f"Processed timepoint {i}: Nucleus={nucleus_whiteness:.4f}, Cytoplasm={cytoplasm_whiteness:.4f}")
            
        except Exception as e:
            print(f"Error processing timepoint {i}: {str(e)}")
            continue
    
    # Create Excel file
    try:
        wb = Workbook()
        ws = wb.active
        ws.title = "Whiteness Tracking Results"
        
        # Add headers
        headers = [
            "Timepoint", "Filename", "Nucleus Whiteness", "Cytoplasm Whiteness",
            "Nucleus Mean Grayscale", "Cytoplasm Mean Grayscale"
        ]
        for col, header in enumerate(headers, 1):
            ws.cell(row=1, column=col, value=header)
        
        # Add data
        for row, result in enumerate(tracking_results, 2):
            ws.cell(row=row, column=1, value=result["timepoint"])
            ws.cell(row=row, column=2, value=result["filename"])
            ws.cell(row=row, column=3, value=result["nucleus_whiteness"])
            ws.cell(row=row, column=4, value=result["cytoplasm_whiteness"])
            ws.cell(row=row, column=5, value=result["nucleus_mean_grayscale"])
            ws.cell(row=row, column=6, value=result["cytoplasm_mean_grayscale"])
        
        # Save Excel file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_filename = f"whiteness_tracking_{start_timepoint}_to_{end_timepoint}_{timestamp}.xlsx"
        excel_path = os.path.join("static", excel_filename)
        
        # Ensure static directory exists
        os.makedirs("static", exist_ok=True)
        
        wb.save(excel_path)
        
        return jsonify({
            "success": True,
            "message": f"Tracking complete! Processed {len(tracking_results)} timepoints.",
            "results": tracking_results,
            "excel_file": excel_filename,
            "download_url": f"/static/{excel_filename}"
        })
        
    except Exception as e:
        return jsonify({"error": f"Failed to create Excel file: {str(e)}"}), 500


if __name__ == '__main__':
    # Check if folders exist
    if not os.path.exists(CH00_FOLDER):
        print(f"Warning: {CH00_FOLDER} folder not found!")
    if not os.path.exists(CH002_FOLDER):
        print(f"Warning: {CH002_FOLDER} folder not found!")
    
    # Get initial image count
    image_pairs = get_image_files()
    print(f"Found {len(image_pairs)} image pairs")
    
    app.run(debug=True, host='0.0.0.0', port=5003)
