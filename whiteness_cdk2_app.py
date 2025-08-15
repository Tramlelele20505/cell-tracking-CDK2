#!/usr/bin/env python3
"""
CDK2 Activity Measurement App using Whiteness Analysis
Based on Cappell et al. (2016) methodology using grayscale intensity measurements
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
import pandas as pd

# Import our whiteness-based CDK2 analyzer
from cdk2_whiteness_analyzer import CDK2WhitenessAnalyzer

app = Flask(__name__)

# Configuration
CH00_FOLDER = "ch00 2"  # Nucleus only images
CH002_FOLDER = "ch002"  # Nucleus + cytoplasm images
STATIC_FOLDER = "static"

# Create static folder if it doesn't exist
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Initialize CDK2 whiteness analyzer
cdk2_analyzer = CDK2WhitenessAnalyzer()

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
    Main page with CDK2 whiteness analysis capabilities
    """
    return render_template('whiteness_cdk2_index.html')

@app.route('/api/cdk2-whiteness-measurement', methods=['POST'])
def cdk2_whiteness_measurement():
    """
    CORRECTED CDK2 activity measurement endpoint using whiteness analysis
    Following the proper methodology:
    1. Segment nucleus from ch00 (nucleus only)
    2. Align nucleus with ch002 (nucleus + cytoplasm)
    3. Measure nuclear whiteness in ch002
    4. Measure cytoplasmic whiteness in ch002
    5. Calculate CDK2 activity based on ch002 measurements
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
        ch00_img = cv2.imread(nucleus_abs, cv2.IMREAD_GRAYSCALE)  # ch00 - nucleus only
        ch002_img = cv2.imread(cytoplasm_abs, cv2.IMREAD_GRAYSCALE)  # ch002 - nucleus + cytoplasm
        
        if ch00_img is None or ch002_img is None:
            return jsonify({"error": "Failed to read one or both images"}), 400
        
        # Use the CORRECTED CDK2 measurement method
        cdk2_results = cdk2_analyzer.measure_cdk2_activity(
            ch00_img, ch002_img, selection, segmentation_method, 
            cytoplasm_mode='whole_cell', ring_width=ring_width, 
            background_threshold=10
        )
        
        # Add metadata
        cdk2_results.update({
            'cell_id': cell_id,
            'timestamp': datetime.now().isoformat(),
            'filename_ch00': nucleus_filename,
            'filename_ch002': cytoplasm_filename
        })
        
        # Track cell across frames if tracking mode
        if data.get("trackingMode", False):
            cdk2_analyzer.track_cell_whiteness_across_frames(cell_id, cdk2_results)
        
        # Store in measurement history
        cdk2_analyzer.measurement_history.append(cdk2_results)
        
        return jsonify({
            "success": True,
            "cdk2_measurements": cdk2_results,
            "methodology": "Corrected CDK2 Analysis: Nucleus segmented from ch00, whiteness measured in ch002 (excluding black background)",
            "reference": "Cappell et al. (2016) Cell 166, 167-180"
        })
        
    except Exception as e:
        print(f"Error in corrected CDK2 whiteness measurement: {e}")
        return jsonify({"error": f"Measurement failed: {str(e)}"}), 500

@app.route('/api/cdk2-whiteness-dynamics/<cell_id>')
def get_cdk2_whiteness_dynamics(cell_id):
    """
    Get CDK2 whiteness dynamics analysis for a specific cell
    """
    try:
        dynamics = cdk2_analyzer.analyze_cdk2_whiteness_dynamics(cell_id)
        return jsonify(dynamics)
    except Exception as e:
        return jsonify({"error": f"Dynamics analysis failed: {str(e)}"}), 500

@app.route('/api/export-cdk2-whiteness-data')
def export_cdk2_whiteness_data():
    """
    Export all CDK2 whiteness measurement data
    """
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"cdk2_whiteness_measurements_{timestamp}.json"
        filepath = os.path.join(STATIC_FOLDER, filename)
        
        cdk2_analyzer.export_whiteness_measurements(filepath)
        
        return jsonify({
            "success": True,
            "filename": filename,
            "download_url": f"/static/{filename}",
            "summary": {
                "total_measurements": len(cdk2_analyzer.measurement_history),
                "total_cells": len(cdk2_analyzer.cell_tracks),
                "methodology": "Corrected CDK2 Activity Analysis: Nucleus segmented from ch00, whiteness measured in ch002"
            }
        })
    except Exception as e:
        return jsonify({"error": f"Export failed: {str(e)}"}), 500

@app.route('/api/export-cdk2-whiteness-excel')
def export_cdk2_whiteness_excel():
    """
    Export all CDK2 whiteness measurement data to Excel with separated columns
    """
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"cdk2_whiteness_measurements_{timestamp}.xlsx"
        filepath = os.path.join(STATIC_FOLDER, filename)
        
        # Prepare data for Excel export
        excel_data = []
        
        for i, measurement in enumerate(cdk2_analyzer.measurement_history):
            row = {
                # Basic measurement info
                'Measurement_ID': i + 1,
                'Timestamp': measurement.get('timestamp', ''),
                'Cell_ID': measurement.get('cell_id', ''),
                'Filename_ch00': measurement.get('filename_ch00', ''),
                'Filename_ch002': measurement.get('filename_ch002', ''),
                
                # Selection coordinates
                'Selection_X': measurement.get('selection_xywh', {}).get('x', ''),
                'Selection_Y': measurement.get('selection_xywh', {}).get('y', ''),
                'Selection_Width': measurement.get('selection_xywh', {}).get('w', ''),
                'Selection_Height': measurement.get('selection_xywh', {}).get('h', ''),
                
                # Nuclear measurements (ch002)
                'Nuclear_Mean_Intensity': measurement.get('nuclear_whiteness_ch002', {}).get('mean_intensity', 0),
                'Nuclear_Median_Intensity': measurement.get('nuclear_whiteness_ch002', {}).get('median_intensity', 0),
                'Nuclear_Std_Intensity': measurement.get('nuclear_whiteness_ch002', {}).get('std_intensity', 0),
                'Nuclear_Min_Intensity': measurement.get('nuclear_whiteness_ch002', {}).get('min_intensity', 0),
                'Nuclear_Max_Intensity': measurement.get('nuclear_whiteness_ch002', {}).get('max_intensity', 0),
                'Nuclear_Pixel_Count': measurement.get('nuclear_whiteness_ch002', {}).get('pixel_count', 0),
                'Nuclear_Foreground_Pixels': measurement.get('nuclear_whiteness_ch002', {}).get('foreground_pixel_count', 0),
                'Nuclear_Background_Pixels': measurement.get('nuclear_whiteness_ch002', {}).get('background_pixel_count', 0),
                'Nuclear_Area': measurement.get('nuclear_whiteness_ch002', {}).get('area', 0),
                'Nuclear_Eccentricity': measurement.get('nuclear_whiteness_ch002', {}).get('eccentricity', 0),
                'Nuclear_Solidity': measurement.get('nuclear_whiteness_ch002', {}).get('solidity', 0),
                'Nuclear_Extent': measurement.get('nuclear_whiteness_ch002', {}).get('extent', 0),
                'Nuclear_Perimeter': measurement.get('nuclear_whiteness_ch002', {}).get('perimeter', 0),
                'Nuclear_Centroid_X': measurement.get('nuclear_whiteness_ch002', {}).get('centroid', (0, 0))[0] if measurement.get('nuclear_whiteness_ch002', {}).get('centroid') else 0,
                'Nuclear_Centroid_Y': measurement.get('nuclear_whiteness_ch002', {}).get('centroid', (0, 0))[1] if measurement.get('nuclear_whiteness_ch002', {}).get('centroid') else 0,
                
                # Cytoplasmic measurements (ch002)
                'Cytoplasmic_Mean_Intensity': measurement.get('cytoplasmic_whiteness_ch002', {}).get('mean_intensity', 0),
                'Cytoplasmic_Median_Intensity': measurement.get('cytoplasmic_whiteness_ch002', {}).get('median_intensity', 0),
                'Cytoplasmic_Std_Intensity': measurement.get('cytoplasmic_whiteness_ch002', {}).get('std_intensity', 0),
                'Cytoplasmic_Min_Intensity': measurement.get('cytoplasmic_whiteness_ch002', {}).get('min_intensity', 0),
                'Cytoplasmic_Max_Intensity': measurement.get('cytoplasmic_whiteness_ch002', {}).get('max_intensity', 0),
                'Cytoplasmic_Pixel_Count': measurement.get('cytoplasmic_whiteness_ch002', {}).get('pixel_count', 0),
                'Cytoplasmic_Foreground_Pixels': measurement.get('cytoplasmic_whiteness_ch002', {}).get('foreground_pixel_count', 0),
                'Cytoplasmic_Background_Pixels': measurement.get('cytoplasmic_whiteness_ch002', {}).get('background_pixel_count', 0),
                'Cytoplasmic_Area': measurement.get('cytoplasmic_whiteness_ch002', {}).get('area', 0),
                'Cytoplasmic_Eccentricity': measurement.get('cytoplasmic_whiteness_ch002', {}).get('eccentricity', 0),
                'Cytoplasmic_Solidity': measurement.get('cytoplasmic_whiteness_ch002', {}).get('solidity', 0),
                'Cytoplasmic_Extent': measurement.get('cytoplasmic_whiteness_ch002', {}).get('extent', 0),
                'Cytoplasmic_Perimeter': measurement.get('cytoplasmic_whiteness_ch002', {}).get('perimeter', 0),
                'Cytoplasmic_Centroid_X': measurement.get('cytoplasmic_whiteness_ch002', {}).get('centroid', (0, 0))[0] if measurement.get('cytoplasmic_whiteness_ch002', {}).get('centroid') else 0,
                'Cytoplasmic_Centroid_Y': measurement.get('cytoplasmic_whiteness_ch002', {}).get('centroid', (0, 0))[1] if measurement.get('cytoplasmic_whiteness_ch002', {}).get('centroid') else 0,
                
                # CDK2 activity metrics
                'CDK2_Ratio_Cyto_over_Nuc': measurement.get('cdk2_ratio_cyto_over_nuc', 0),
                'CDK2_Ratio_Nuc_over_Cyto': measurement.get('cdk2_ratio_nuc_over_cyto', 0),
                'CDK2_Activity_Index': measurement.get('cdk2_activity_index', 0),
                
                # Area measurements
                'Nucleus_Pixel_Count': measurement.get('nucleus_pixel_count', 0),
                'Cytoplasm_Pixel_Count': measurement.get('cytoplasm_pixel_count', 0),
                'Nuclear_Area_Percentage': (100 * measurement.get('nucleus_pixel_count', 0) / max(1, (measurement.get('nucleus_pixel_count', 0) + measurement.get('cytoplasm_pixel_count', 0)))),
                'Cytoplasmic_Area_Percentage': (100 * measurement.get('cytoplasm_pixel_count', 0) / max(1, (measurement.get('nucleus_pixel_count', 0) + measurement.get('cytoplasm_pixel_count', 0)))),
                
                # Analysis parameters
                'Segmentation_Method': measurement.get('segmentation_method', ''),
                'Cytoplasm_Mode': measurement.get('cytoplasm_mode', ''),
                'Ring_Width': measurement.get('ring_width', 0),
                'Inner_Gap': measurement.get('inner_gap', 0),
                'Background_Threshold': measurement.get('background_threshold', 0),
                
                # Cell cycle phase (if available)
                'Cell_Cycle_Phase': measurement.get('cell_cycle_phase', '')
            }
            excel_data.append(row)
        
        # Create DataFrame and export to Excel
        df = pd.DataFrame(excel_data)
        
        # Create Excel writer with multiple sheets
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Main measurements sheet
            df.to_excel(writer, sheet_name='CDK2_Measurements', index=False)
            
            # Summary statistics sheet
            if len(df) > 0:
                summary_data = {
                    'Metric': [
                        'Total Measurements',
                        'Total Cells',
                        'Mean CDK2 Ratio (C/N)',
                        'Std CDK2 Ratio (C/N)',
                        'Mean CDK2 Activity Index',
                        'Std CDK2 Activity Index',
                        'Mean Nuclear Intensity',
                        'Mean Cytoplasmic Intensity',
                        'Mean Nuclear Area %',
                        'Mean Cytoplasmic Area %'
                    ],
                    'Value': [
                        len(df),
                        len(df['Cell_ID'].unique()),
                        df['CDK2_Ratio_Cyto_over_Nuc'].mean(),
                        df['CDK2_Ratio_Cyto_over_Nuc'].std(),
                        df['CDK2_Activity_Index'].mean(),
                        df['CDK2_Activity_Index'].std(),
                        df['Nuclear_Mean_Intensity'].mean(),
                        df['Cytoplasmic_Mean_Intensity'].mean(),
                        df['Nuclear_Area_Percentage'].mean(),
                        df['Cytoplasmic_Area_Percentage'].mean()
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)
            
            # Methodology sheet
            methodology_data = {
                'Parameter': [
                    'Analysis Method',
                    'Reference',
                    'Nuclear Segmentation',
                    'Cytoplasm Detection',
                    'Background Exclusion',
                    'Intensity Measurement',
                    'CDK2 Ratio Calculation'
                ],
                'Description': [
                    'CDK2 Activity Analysis using Whiteness Measurement',
                    'Cappell et al. (2016) Cell 166, 167-180',
                    'Nucleus segmented from ch00 (nucleus only channel)',
                    'Cytoplasm detected in ch002 (nucleus + cytoplasm channel)',
                    'Black background pixels excluded (threshold: 10)',
                    'Mean grayscale intensity measured in ch002 only',
                    'Cytoplasmic intensity / Nuclear intensity'
                ]
            }
            methodology_df = pd.DataFrame(methodology_data)
            methodology_df.to_excel(writer, sheet_name='Methodology', index=False)
        
        return jsonify({
            "success": True,
            "filename": filename,
            "download_url": f"/static/{filename}",
            "summary": {
                "total_measurements": len(cdk2_analyzer.measurement_history),
                "total_cells": len(cdk2_analyzer.cell_tracks),
                "excel_sheets": ["CDK2_Measurements", "Summary_Statistics", "Methodology"],
                "methodology": "Corrected CDK2 Activity Analysis: Nucleus segmented from ch00, whiteness measured in ch002 (excluding black background)"
            }
        })
    except Exception as e:
        print(f"Error in Excel export: {e}")
        return jsonify({"error": f"Excel export failed: {str(e)}"}), 500

@app.route('/api/plot-cdk2-whiteness-trajectory/<cell_id>')
def plot_cdk2_whiteness_trajectory(cell_id):
    """
    Generate and serve CDK2 whiteness trajectory plot
    """
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"cdk2_whiteness_trajectory_{cell_id}_{timestamp}.png"
        filepath = os.path.join(STATIC_FOLDER, filename)
        
        cdk2_analyzer.plot_cdk2_whiteness_trajectory(cell_id, save_path=filepath)
        
        return jsonify({
            "success": True,
            "filename": filename,
            "plot_url": f"/static/{filename}"
        })
    except Exception as e:
        return jsonify({"error": f"Plot generation failed: {str(e)}"}), 500

@app.route('/api/cell-whiteness-tracking-summary')
def get_cell_whiteness_tracking_summary():
    """
    Get summary of all tracked cells with whiteness analysis
    """
    try:
        summary = {
            "total_cells": len(cdk2_analyzer.cell_tracks),
            "total_measurements": len(cdk2_analyzer.measurement_history),
            "methodology": "Corrected CDK2 Activity Analysis: Nucleus segmented from ch00, whiteness measured in ch002",
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
            
            # Add CDK2 whiteness dynamics if available
            if len(track['measurements']) > 1:
                dynamics = cdk2_analyzer.analyze_cdk2_whiteness_dynamics(cell_id)
                if 'error' not in dynamics:
                    cell_summary.update({
                        "peak_cdk2_ratio": dynamics['peak_cdk2_ratio'],
                        "peak_time": dynamics['peak_time'],
                        "mean_cdk2_ratio": dynamics['mean_cdk2_ratio'],
                        "oscillations": dynamics['oscillations_detected'],
                        "mean_nuclear_whiteness": dynamics['mean_nuclear_whiteness'],
                        "mean_cytoplasmic_whiteness": dynamics['mean_cytoplasmic_whiteness']
                    })
            
            summary["cells"].append(cell_summary)
        
        return jsonify(summary)
    except Exception as e:
        return jsonify({"error": f"Summary generation failed: {str(e)}"}), 500

@app.route('/api/simple-whiteness-measurement', methods=['POST'])
def simple_whiteness_measurement():
    """
    Simple whiteness measurement endpoint (compatible with existing interface)
    Now uses the corrected methodology
    """
    data = request.json
    nucleus_path = data.get("nucleusPath")
    cytoplasm_path = data.get("cytoplasmPath")
    selection = data.get("selection")
    
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
        ch00_img = cv2.imread(nucleus_abs, cv2.IMREAD_GRAYSCALE)  # ch00 - nucleus only
        ch002_img = cv2.imread(cytoplasm_abs, cv2.IMREAD_GRAYSCALE)  # ch002 - nucleus + cytoplasm
        
        if ch00_img is None or ch002_img is None:
            return jsonify({"error": "Failed to read one or both images"}), 400
        
        # Use the CORRECTED CDK2 measurement method
        cdk2_results = cdk2_analyzer.measure_cdk2_activity(
            ch00_img, ch002_img, selection, "otsu", 
            cytoplasm_mode='whole_cell', ring_width=5, 
            background_threshold=10
        )
        
        # Return raw whiteness values (not normalized) for accurate comparison
        nuclear_whiteness = cdk2_results['nuclear_whiteness_ch002']['mean_intensity']
        cytoplasmic_whiteness = cdk2_results['cytoplasmic_whiteness_ch002']['mean_intensity']
        
        # Also provide normalized values for compatibility
        nuclear_whiteness_normalized = nuclear_whiteness / 255.0
        cytoplasmic_whiteness_normalized = cytoplasmic_whiteness / 255.0
        
        return jsonify({
            "success": True,
            "nucleus_whiteness": nuclear_whiteness_normalized,  # Normalize to 0-1 for display
            "cytoplasm_whiteness": cytoplasmic_whiteness_normalized,  # Normalize to 0-1 for display
            "nucleus_whiteness_raw": nuclear_whiteness,  # Raw intensity value
            "cytoplasm_whiteness_raw": cytoplasmic_whiteness,  # Raw intensity value
            "cdk2_ratio": cdk2_results['cdk2_ratio_cyto_over_nuc'],
            "cdk2_activity": cdk2_results['cdk2_activity_index'],
            "max_possible": 1.0,
            "methodology": "Corrected CDK2 Analysis: Nucleus segmented from ch00, whiteness measured in ch002 (excluding black background)",
            "background_threshold": cdk2_results['background_threshold'],
            "nuclear_foreground_pixels": cdk2_results['nuclear_whiteness_ch002']['foreground_pixel_count'],
            "cytoplasmic_foreground_pixels": cdk2_results['cytoplasmic_whiteness_ch002']['foreground_pixel_count']
        })
        
    except Exception as e:
        print(f"Error in simple whiteness measurement: {e}")
        return jsonify({"error": f"Measurement failed: {str(e)}"}), 500

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
    print("CDK2 Whiteness Analyzer initialized with CORRECTED methodology")
    print("Methodology: Nucleus segmented from ch00, whiteness measured in ch002")
    
    app.run(debug=True, host='0.0.0.0', port=5003)
