#!/usr/bin/env python3
"""
Enhanced Cell Image Viewer with CDK2 Analysis
Integrates Cappell et al. (2016) methodology for comprehensive CDK2 measurement
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
import json
from datetime import datetime

# Import our enhanced CDK2 analyzer
from enhanced_cdk2_measurement import CDK2Analyzer

app = Flask(__name__)

# Configuration
CH00_FOLDER = "ch00 2"  # Nucleus only images
CH002_FOLDER = "ch002"  # Nucleus + cytoplasm images
STATIC_FOLDER = "static"

# Create static folder if it doesn't exist
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Initialize CDK2 analyzer
cdk2_analyzer = CDK2Analyzer()

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
    Main page with enhanced CDK2 analysis capabilities
    """
    return render_template('enhanced_index.html')

@app.route('/api/enhanced-cdk2-measurement', methods=['POST'])
def enhanced_cdk2_measurement():
    """
    Enhanced CDK2 measurement endpoint using Cappell et al. methodology
    """
    data = request.json
    nucleus_path = data.get("nucleusPath")
    cytoplasm_path = data.get("cytoplasmPath")
    selection = data.get("selection")
    segmentation_method = data.get("segmentationMethod", "otsu")
    ring_width = data.get("ringWidth", 5)
    cell_id = data.get("cellId", f"cell_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
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
    
    try:
        # Load images as grayscale
        nucleus_img = cv2.imread(nucleus_abs, cv2.IMREAD_GRAYSCALE)
        cytoplasm_img = cv2.imread(cytoplasm_abs, cv2.IMREAD_GRAYSCALE)
        
        if nucleus_img is None or cytoplasm_img is None:
            return jsonify({"error": "Failed to read one or both images"}), 400
        
        # Convert % selection to pixels
        h, w = nucleus_img.shape
        x = int(selection["x"] / 100 * w)
        y = int(selection["y"] / 100 * h)
        width = int(selection["width"] / 100 * w)
        height = int(selection["height"] / 100 * h)
        
        # Crop images to selection
        nucleus_crop = nucleus_img[y:y+height, x:x+width]
        cytoplasm_crop = cytoplasm_img[y:y+height, x:x+width]
        
        # Perform automated nuclear segmentation
        nucleus_mask = cdk2_analyzer.segment_nucleus(nucleus_crop, method=segmentation_method)
        
        # Create cytoplasmic ring
        cytoplasm_mask = np.ones_like(nucleus_crop, dtype=bool)  # Full selection area
        ring_mask = cdk2_analyzer.create_cytoplasmic_ring(nucleus_mask, ring_width=ring_width)
        
        # Measure CDK2 parameters
        cdk2_results = cdk2_analyzer.measure_cdk2_parameters(
            nucleus_crop, cytoplasm_crop, nucleus_mask, ring_mask
        )
        
        # Add metadata
        cdk2_results.update({
            'cell_id': cell_id,
            'timestamp': datetime.now().isoformat(),
            'segmentation_method': segmentation_method,
            'ring_width': ring_width,
            'selection_area': width * height,
            'nuclear_area_percentage': (np.sum(nucleus_mask) / (width * height)) * 100,
            'ring_area_percentage': (np.sum(ring_mask) / (width * height)) * 100
        })
        
        # Track cell across frames if tracking mode
        if data.get("trackingMode", False):
            cdk2_analyzer.track_cell_across_frames(cell_id, cdk2_results)
        
        # Store in measurement history
        cdk2_analyzer.measurement_history.append(cdk2_results)
        
        return jsonify({
            "success": True,
            "cdk2_measurements": cdk2_results,
            "segmentation_info": {
                "nuclear_pixels": int(np.sum(nucleus_mask)),
                "ring_pixels": int(np.sum(ring_mask)),
                "total_pixels": width * height
            }
        })
        
    except Exception as e:
        print(f"Error in enhanced CDK2 measurement: {e}")
        return jsonify({"error": f"Measurement failed: {str(e)}"}), 500

@app.route('/api/cdk2-dynamics/<cell_id>')
def get_cdk2_dynamics(cell_id):
    """
    Get CDK2 dynamics analysis for a specific cell
    """
    try:
        dynamics = cdk2_analyzer.analyze_cdk2_dynamics(cell_id)
        return jsonify(dynamics)
    except Exception as e:
        return jsonify({"error": f"Dynamics analysis failed: {str(e)}"}), 500

@app.route('/api/export-cdk2-data')
def export_cdk2_data():
    """
    Export all CDK2 measurement data
    """
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"cdk2_measurements_{timestamp}.json"
        filepath = os.path.join(STATIC_FOLDER, filename)
        
        cdk2_analyzer.export_measurements(filepath)
        
        return jsonify({
            "success": True,
            "filename": filename,
            "download_url": f"/static/{filename}",
            "summary": {
                "total_measurements": len(cdk2_analyzer.measurement_history),
                "total_cells": len(cdk2_analyzer.cell_tracks)
            }
        })
    except Exception as e:
        return jsonify({"error": f"Export failed: {str(e)}"}), 500

@app.route('/api/plot-cdk2-trajectory/<cell_id>')
def plot_cdk2_trajectory(cell_id):
    """
    Generate and serve CDK2 trajectory plot
    """
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"cdk2_trajectory_{cell_id}_{timestamp}.png"
        filepath = os.path.join(STATIC_FOLDER, filename)
        
        cdk2_analyzer.plot_cdk2_trajectory(cell_id, save_path=filepath)
        
        return jsonify({
            "success": True,
            "filename": filename,
            "plot_url": f"/static/{filename}"
        })
    except Exception as e:
        return jsonify({"error": f"Plot generation failed: {str(e)}"}), 500

@app.route('/api/cell-tracking-summary')
def get_cell_tracking_summary():
    """
    Get summary of all tracked cells
    """
    try:
        summary = {
            "total_cells": len(cdk2_analyzer.cell_tracks),
            "total_measurements": len(cdk2_analyzer.measurement_history),
            "cells": []
        }
        
        for cell_id, track in cdk2_analyzer.cell_tracks.items():
            cell_summary = {
                "cell_id": cell_id,
                "birth_frame": track['birth_frame'],
                "total_frames": len(track['measurements']),
                "mother_id": track['mother_id'],
                "daughter_count": len(track['daughter_ids'])
            }
            
            # Add CDK2 dynamics if available
            if len(track['measurements']) > 1:
                dynamics = cdk2_analyzer.analyze_cdk2_dynamics(cell_id)
                if 'error' not in dynamics:
                    cell_summary.update({
                        "peak_cdk2": dynamics['peak_cdk2'],
                        "peak_time": dynamics['peak_time'],
                        "mean_cdk2_ratio": dynamics['mean_cdk2_ratio'],
                        "oscillations": dynamics['oscillations_detected']
                    })
            
            summary["cells"].append(cell_summary)
        
        return jsonify(summary)
    except Exception as e:
        return jsonify({"error": f"Summary generation failed: {str(e)}"}), 500

# Keep existing routes for compatibility
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
    image_pairs = get_image_files()
    base_names = list(image_pairs.keys())
    timepoints = list(range(len(base_names)))
    
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
    base_names = list(image_pairs.keys())
    
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
        image_path = os.path.join(CH00_FOLDER, filename)
        if not os.path.exists(image_path):
            return "Image not found", 404
        
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return "Failed to load image", 500
        
        img_normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img_enhanced = cv2.convertScaleAbs(img_normalized, alpha=1.2, beta=10)
        
        _, buffer = cv2.imencode('.jpg', img_enhanced, [cv2.IMWRITE_JPEG_QUALITY, 90])
        
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
        image_path = os.path.join(CH002_FOLDER, filename)
        if not os.path.exists(image_path):
            return "Image not found", 404
        
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return "Failed to load image", 500
        
        img_normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img_enhanced = cv2.convertScaleAbs(img_normalized, alpha=1.2, beta=10)
        
        _, buffer = cv2.imencode('.jpg', img_enhanced, [cv2.IMWRITE_JPEG_QUALITY, 90])
        
        response = Response(buffer.tobytes(), mimetype='image/jpeg')
        response.headers['Cache-Control'] = 'public, max-age=31536000'
        return response
        
    except Exception as e:
        print(f"Error serving ch002 image {filename}: {e}")
        return "Error processing image", 500

if __name__ == '__main__':
    # Check if folders exist
    if not os.path.exists(CH00_FOLDER):
        print(f"Warning: {CH00_FOLDER} folder not found!")
    if not os.path.exists(CH002_FOLDER):
        print(f"Warning: {CH002_FOLDER} folder not found!")
    
    # Get initial image count
    image_pairs = get_image_files()
    print(f"Found {len(image_pairs)} image pairs")
    print("Enhanced CDK2 Analyzer initialized with Cappell et al. methodology")
    
    app.run(debug=True, host='0.0.0.0', port=5003)

