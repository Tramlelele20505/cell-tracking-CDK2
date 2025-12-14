#!/usr/bin/env python3
"""
Cell Image Viewer Web Application
 
 Supports uploading cell nucleus images and corresponding nucleus+cytoplasm images
"""

from flask import Flask, render_template, jsonify, request,send_file, send_from_directory, Response, flash, redirect, url_for

from werkzeug.utils import secure_filename
import os
import random
import re
from pathlib import Path
from collections import OrderedDict
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import openpyxl
from openpyxl import Workbook
from datetime import datetime
import base64
from cellpose.models import CellposeModel #version: 2.3.2
from scipy import ndimage as ndi
from skimage.morphology import disk
from skimage.filters import median as sk_median
from scipy.ndimage import binary_dilation, binary_erosion
from skimage import exposure
import btrack
import pandas as pd
import json



os.environ['OMP_NUM_THREADS'] = '6'
os.environ['MKL_NUM_THREADS'] = '6'
#=======Configuration & Globals========
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

UPLOAD_FOLDER = "uploads"
CACHE_DIR = "seg_cache"
CH00_FOLDER = os.path.join(UPLOAD_FOLDER, "ch00")  
CH002_FOLDER = os.path.join(UPLOAD_FOLDER, "ch002")  
STATIC_FOLDER = "static"
ALLOWED_EXTENSIONS = {'tif', 'tiff', 'jpg', 'jpeg', 'png', 'bmp'}
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2 GB max upload size

_cyto_model_cache = None

# Create folders if they don't exist
for d in [UPLOAD_FOLDER, CH00_FOLDER, CH002_FOLDER, STATIC_FOLDER, CACHE_DIR]:
    os.makedirs(d, exist_ok=True)
    
def get_cyto_model():
    """
    Loads and caches the Cellpose 'cyto2' model for segmentation using CPU.

    This function uses a global variable to store the model, ensuring it is
    loaded only once per application run to speeds up program.
    """
    global _cyto_model_cache

    if _cyto_model_cache is None:
            _cyto_model_cache = CellposeModel(model_type="cyto2", gpu=False)
      
    return _cyto_model_cache

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def natural_sort_key(s):
    """
    This splits the string into text and number parts.
    Example:
        natural_sort_key("file10.tif")
        → ["file", 10, ".tif"]
    """
    import re
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

def get_image_files():
    """
    Scan the upload folders, match nucleus (ch00) and nucleus+cytoplasm (ch002)
    images, and return them as paired entries.

    Pairing is done in two steps:

    1. **Primary matching: Same base filename**
       Example:
           ch00:  "img1.tif"
           ch002: "img1.tif"
       → These are paired together.

    2. **Fallback matching: Index-based pairing**
       If no filenames match between the folders, the function pairs files
       simply by their index after natural sorting.
       Example:
           ch00_files[0] ↔ ch002_files[0]
           ch00_files[1] ↔ ch002_files[1]
       Useful when filenames differ but ordering corresponds.
    """
    ch00_files = []
    ch002_files = []
    
    # Scan ch00 (nucleus) folder for allowed image files and sort by natural key
    if os.path.exists(CH00_FOLDER):
        ch00_files = sorted(
            [file for file in os.listdir(CH00_FOLDER) 
             if allowed_file(file)],
            key=natural_sort_key
        )
        print(f"Found {len(ch00_files)} ch00 files")
    
    # Scan ch02 (nucleus) folder for allowed image files and sort by natural key
    if os.path.exists(CH002_FOLDER):
        ch002_files = sorted(
            [file for file in os.listdir(CH002_FOLDER) 
             if allowed_file(file)],
            key=natural_sort_key
        )
        print(f"Found {len(ch002_files)} ch002 files")
    
    # Create a lookup for ch002 files by their base name for quick matching (ignore extension)
    ch002_by_base = {os.path.splitext(f)[0]: f for f in ch002_files}
    image_pairs = OrderedDict()
    
    # --- Pairing Strategy 1: Match by identical base name ---
    for ch00_file in ch00_files:
        base_name = os.path.splitext(ch00_file)[0]
        ch002_file = ch002_by_base.get(base_name)
        if ch002_file:
            image_pairs[base_name] = {
                'nucleus': ch00_file,
                'nucleus_cytoplasm': ch002_file
            }

    # --- Pairing Strategy 2: Fallback to index-based pairing ---
    if len(image_pairs) == 0 and ch00_files and ch002_files:
        print("No filename matches across channels. Falling back to index-based pairing.")
        num_pairs = min(len(ch00_files), len(ch002_files))
        for i in range(num_pairs):
            ch00_file = ch00_files[i]
            ch002_file = ch002_files[i]
            base_name = f"pair_{i:05d}"
            image_pairs[base_name] = {
                'nucleus': ch00_file,
                'nucleus_cytoplasm': ch002_file
            }
    
    print(f"Successfully paired {len(image_pairs)} file pairs")
    return image_pairs

def get_timepoints():
    """
    Generates a list of timepoint indices based on the number of paired images.
    It assigns a time index (0, 1, 2, …) to each image pair in the same order they appear in `image_pairs`
    Simply, it assign the index in range of len of image_pairs to its order in list.
    """
    image_pairs = get_image_files()
    timepoints = []
    
    # Create a list of all base names in original folder order
    base_names = list(image_pairs.keys())
    
    for i, base_name in enumerate(base_names):
        timepoints.append(i)
    
    return timepoints

def parse_image_info(filename):
    """
    Extracts metadata (row, column, site, timepoint) from a filename.

    Assumes a specific filename format: "row_col_site_t..._ch....tif".
    If the format does not match, it returns default "Uploaded Image" info.

    Args:
        filename (str): The filename to parse.

    Returns:
        dict: A dictionary containing the parsed metadata.
    """
    # Regex to capture information from a structured filename
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
    
    # If filename doesn't match the pattern, return basic info
    return {
        'row': 0,
        'column': 0,
        'site': 0,
        'timepoint': 0,
        'channel': 0,
        'formatted_time': "Custom Image",
        'location': "Uploaded Image"
    }

def serve_processed_image(folder, filename):
    """
    Generic function to load a grayscale image, convert it to JPEG, and serve it.
    
    This function is used for serve_ch00_image and serve_ch002_image
    """
    try:
        image_path = os.path.join(folder, filename)
        if not os.path.exists(image_path):
            return "Image not found", 404
        
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return "Failed to load image", 500
        
        # Encode the image to JPEG format
        _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        
        # Create a response with caching headers
        response = Response(buffer.tobytes(), mimetype='image/jpeg')
        response.headers['Cache-Control'] = 'public, max-age=31536000'
        return response
        
    except Exception as e:
        print(f"Error serving image {filename}: {e}")
        return "Error processing image", 500
    
@app.route('/')
def index():
    """
    Main page with upload interface and random cell selection
    """
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_files():
    """
    Handles uploading of images for both channels:
      - ch00  = nucleus only images
      - ch002 = nucleus + cytoplasm images

    The function:
        • Accepts uploaded files from two channels
        • Validates file types
        • Saves them into the correct folders
        • Cleans up partially saved files if an error occurs
        • Shows the user a confirmation (flash message)
    """
    #Handle POST (actual upload)
    if request.method == 'POST':
        # Retrieve file lists from the form upload.
        # If a channel is not provided, it becomes an empty list.
        ch00_files = request.files.getlist('ch00_files') if 'ch00_files' in request.files else []
        ch002_files = request.files.getlist('ch002_files') if 'ch002_files' in request.files else []
        
        # If submitted the form but uploaded nothing, return warning
        if not ch00_files and not ch002_files:
            flash('No folders selected. Please choose at least one channel folder to upload.')
            return redirect(request.url)
        
        #Counters
        uploaded_count = 0
        ch00_image_count = 0
        ch002_image_count = 0
        
        # Track files saved this request for cleanup if needed
        saved_file_paths = []

        # Process ch00 files (nucleus only) - filter for image files only
        try:
            for file in ch00_files:
                
                # Ignore empty file inputs
                if not file or not file.filename:
                    continue
                
                # Clean filename and remove directory paths
                original_name = os.path.basename(file.filename)
                safe_name = secure_filename(original_name)
                
                # Check the valid extensions
                if not allowed_file(safe_name):
                    continue
                
                 # Build storage path inside ch00 folder
                filepath = os.path.join(CH00_FOLDER, safe_name)
                # Ensure upload directories exist
                os.makedirs(CH00_FOLDER, exist_ok=True)
                # Save file to disk
                file.save(filepath)
                
                
                saved_file_paths.append(filepath)
                uploaded_count += 1
                ch00_image_count += 1
        except OSError as e:
            # Cleanup partial files on disk space error
            for p in saved_file_paths:
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass
            flash('Upload failed: not enough disk space. Please free space or upload fewer images.')
            return redirect(request.url)
        
        # The same with second channel
        try:
            for file in ch002_files:
                if not file or not file.filename:
                    continue
                original_name = os.path.basename(file.filename)
                safe_name = secure_filename(original_name)
                if not allowed_file(safe_name):
                    continue
                filepath = os.path.join(CH002_FOLDER, safe_name)
                os.makedirs(CH002_FOLDER, exist_ok=True)
                file.save(filepath)
                saved_file_paths.append(filepath)
                uploaded_count += 1
                ch002_image_count += 1
        except OSError as e:
            for p in saved_file_paths:
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass
            flash('Upload failed: not enough disk space. Please free space or upload fewer images.')
            return redirect(request.url)
        
        # If all done, show informations:
        if uploaded_count > 0:
            parts = []
            if ch00_image_count:
                parts.append(f"{ch00_image_count} nucleus (ch00)")
            if ch002_image_count:
                parts.append(f"{ch002_image_count} nucleus+cytoplasm (ch002)")
            flash('Successfully uploaded ' + ', '.join(parts))
        else:
            flash('No valid image files found. Please ensure folders contain TIF, TIFF, JPG, JPEG, PNG, or BMP files.')
        
        return redirect(url_for('index'))
    
    return render_template('upload.html')

@app.route('/clear-uploads', methods=['POST'])
def clear_uploads():
    """
    Deletes all uploaded images and cached segmentation masks.
    """
    try:
        # Remove all files from upload folders
        for folder in [CH00_FOLDER, CH002_FOLDER, CACHE_DIR]:
            if os.path.exists(folder):
                for filename in os.listdir(folder):
                    file_path = os.path.join(folder, filename)
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
        
        flash('All uploaded files have been cleared')
    except Exception as e:
        flash(f'Error clearing files: {str(e)}')
    
    return redirect(url_for('index'))

@app.route('/api/random-cell')
def get_random_cell():
    """
    API endpoint to get a random cell image pair
    """
    image_pairs = get_image_files()
    
    if not image_pairs:
        return jsonify({
            'error': 'No image pairs found',
            'message': 'Please upload folders containing images for both channels first.'
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
    Returns a sorted list of all paired cell images with their metadata.
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

@app.route('/api/upload-stats')
def upload_stats():
    """
    Return raw counts of uploaded files per channel to confirm uploads succeeded
    even if pairing hasn't been established yet.
    """
    ch00_count = 0
    ch002_count = 0
    if os.path.exists(CH00_FOLDER):
        ch00_count = len([f for f in os.listdir(CH00_FOLDER) if allowed_file(f)])
    if os.path.exists(CH002_FOLDER):
        ch002_count = len([f for f in os.listdir(CH002_FOLDER) if allowed_file(f)])
    return jsonify({
        'ch00_count': ch00_count,
        'ch002_count': ch002_count,
        'pairs': len(get_image_files())
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

def segment_cytoplasm_from_nucleus(
    cytoplasm_img_path, 
    nucleus_mask,
    denoise_h=10,
    dilate_px=3,          # dilate nucleus 5–10 px
    bright_thresh=30,
    dark_thresh=20,
    dark_region_ratio=0.1
):
    """
    It uses region growing and morphological operations to define the cytoplasm area around a given nucleus mask.
    """
    import cv2
    import numpy as np

    #  Read image & denoise
    img = cv2.imread(cytoplasm_img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    blur = cv2.fastNlMeansDenoising(img, None, h=denoise_h,
                                    templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    blur = clahe.apply(blur)
    h, w = img.shape

    #  Dilate the nucleus 5–10 px so that the seed is greater than the nucleus
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated_nucleus = cv2.dilate(nucleus_mask, kernel, iterations=dilate_px)
    blur[dilated_nucleus > 0] = 255

    #  Region growing
    visited = np.zeros_like(img, dtype=np.uint8)
    mask = np.zeros_like(img, dtype=np.uint8)
    seed_pts = np.column_stack(np.where(dilated_nucleus > 0))
    stack = [tuple(pt[::-1]) for pt in seed_pts]

    while stack:
        x, y = stack.pop()
        if not (0 <= x < w and 0 <= y < h):
            continue
        if visited[y, x]:
            continue
        visited[y, x] = 1
        val = blur[y, x]

        if val >= bright_thresh:  # bright cytoplasm
            mask[y, x] = 255
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0: 
                        continue
                    stack.append((x + dx, y + dy))
        elif val <= dark_thresh:
            neigh = blur[max(0, y-2):min(h, y+3), max(0, x-2):min(w, x+3)]
            dark_ratio = np.mean(neigh < dark_thresh)
            if dark_ratio < dark_region_ratio:
                mask[y, x] = 255
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0: 
                            continue
                        stack.append((x + dx, y + dy))

    #  Post-Processing & taking the largest contour 
    clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    max_cnt = max(contours, key=cv2.contourArea)
    cytoplasm_mask = np.zeros_like(img, dtype=np.uint8)
    cv2.drawContours(cytoplasm_mask, [max_cnt], -1, 255, -1)

    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cytoplasm_mask = cv2.erode(cytoplasm_mask, erode_kernel, iterations=3)

    #  Expand from nucleus based on area, based on 1/4 of the image area
    nucleus_area = np.sum(nucleus_mask > 0)
    img_area = img.shape[0] * img.shape[1]

    min_expand_px = 5
    max_expand_px = 22


    scale_factor = np.sqrt(nucleus_area / float(img_area))
    expand_px = int(min_expand_px + (max_expand_px - min_expand_px) * scale_factor)
    expand_px = max(min_expand_px, min(expand_px, max_expand_px))

    dist_map = cv2.distanceTransform(255 - nucleus_mask, cv2.DIST_L2, 5)
    expanded_mask = (dist_map <= expand_px).astype(np.uint8) * 255

    #  take the intersection with the original cytoplasm mask
    final_mask = cv2.bitwise_and(cytoplasm_mask, expanded_mask)

    #  soften the cytoplasm
    blurred = cv2.GaussianBlur(final_mask, (5, 5), 0)
    _, final_mask = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    #  merge with the nucleus
    final_mask = cv2.bitwise_or(final_mask, nucleus_mask)

    return final_mask

@app.route('/api/select-cytoplasm', methods=['POST'])
def select_cytoplasm():
    
    data = request.json
    image_filename = data.get("image_filename")
    nucleus_contour = data.get("nucleus_contour")

    if not image_filename:
        return jsonify({"error": "Missing image filename"}), 400
    if not nucleus_contour:
        return jsonify({"error": "Missing nucleus contour"}), 400

    image_path = os.path.join(CH002_FOLDER, image_filename)
    if not os.path.exists(image_path):
        return jsonify({"error": f"Image not found: {image_path}"}), 404

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return jsonify({"error": "Failed to read image"}), 500

    # build mask from nucleus contour
    nucleus_mask = np.zeros_like(img, dtype=np.uint8)
    pts = np.array(nucleus_contour, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(nucleus_mask, [pts], 255)

    # run cytoplasm segmentation
    cytoplasm_mask = segment_cytoplasm_from_nucleus(image_path, nucleus_mask)
    if cytoplasm_mask is None:
        return jsonify({"error": "Failed to segment cytoplasm"}), 500

    # convert to contour polygon
    contours, _ = cv2.findContours(cytoplasm_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return jsonify({"error": "No cytoplasm contour found"}), 500

    max_cnt = max(contours, key=cv2.contourArea)
    eps = 0.01 * cv2.arcLength(max_cnt, True)
    approx = cv2.approxPolyDP(max_cnt, eps, True)
    cyto_contour = [(int(pt[0][0]), int(pt[0][1])) for pt in approx]

    return jsonify({"cytoplasm_contour": cyto_contour})

def preprocess_image(gray, bg_sigma=25, median_radius=2, denoise_sigma=0.8, clahe_clip=0.03):
    """
    Applies a series of filters to preprocess an image for segmentation.
    
    The pipeline includes background subtraction, median filtering, Gaussian smoothing,
    and Contrast Limited Adaptive Histogram Equalization (CLAHE).
    
    Args:
        gray (np.array): The input grayscale image (float, 0-1 range).
        bg_sigma (int): Sigma for Gaussian filter for background subtraction.
        median_radius (int): Radius for the median filter disk.
        denoise_sigma (float): Sigma for the Gaussian denoising filter.
        clahe_clip (float): Clip limit for CLAHE.

    Returns:
        np.array: The processed image.
    """
    # Background subtraction using a large-sigma Gaussian filter
    bg = ndi.gaussian_filter(gray, sigma=bg_sigma)
    sub = gray - bg
    sub = np.clip(sub, 0, None)
    if sub.max() > 0:
        sub = sub / sub.max()
    
    # Denoising using median and Gaussian filters
    den = sk_median(sub, disk(median_radius))
    den = ndi.gaussian_filter(den, sigma=denoise_sigma)
    
    # Contrast enhancement
    deneq = exposure.equalize_adapthist(den, clip_limit=clahe_clip)
    return deneq

def shrink_masks_aggressive(masks, erosion_iterations=4):
    """
    Aggressively shrinks segmented masks by applying binary erosion.
    This is used to ensure masks fit tightly around bright nuclei centers.
    
    Args:
        masks (np.array): The labeled mask output from Cellpose.
        erosion_iterations (int): The number of erosion iterations.

    Returns:
        np.array: The shrunken labeled mask.
    """
    masks_shrunk = np.zeros_like(masks)
    
    # Erode each individual nucleus mask
    for nucleus_id in np.unique(masks):
        if nucleus_id == 0:
            continue
        
        nucleus_mask = (masks == nucleus_id).astype(np.uint8)
        # Apply erosion multiple times
        shrunk = nucleus_mask.astype(bool)
        for _ in range(erosion_iterations):
            shrunk = binary_erosion(shrunk)
            
        # Only keep the shrunk mask if it's not empty
        if shrunk.sum() > 0:
            masks_shrunk[shrunk] = nucleus_id
    n_nuclei_kept = masks_shrunk.max()
    return masks_shrunk

def segment_nuclei(image_path, min_area=50):
    """
    Segments nuclei in an image using a pre-trained Cellpose model.
    Includes preprocessing and shrink masks to refine masks.
    
    Args:
        image_path (str): The path to the image file.
        min_area (int): The minimum area (in pixels) for a valid nucleus.

    Returns:
        tuple: A tuple containing:
            - np.array: The final labeled mask.
            - list: A list of valid contour arrays.
            - list: A list of (x, y) centroids for each valid nucleus.
    """
    
    #Load pre-trained Cellpose model - cyto2
    model = get_cyto_model()
    
    # Evaluation parameters for Cellpose
    eval_kwargs = {
        'diameter': 35,
        'channels': [0, 0],
        'flow_threshold': 0.55,
        'cellprob_threshold': -3.5,
        'min_size': 200
    }

    # Load image as grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, [], []
    
    # Convert image to float32 and normalize to [0, 1] if needed
    #Cellpose expect floating-point numbers instead of integers.
    img_float = img.astype(np.float32)
    if img_float.max() > 1:
        img_float = img_float / 255.0
    
    #Preprocess the image
    img_processed = preprocess_image(img_float, bg_sigma=8, median_radius=2, denoise_sigma=0.8, clahe_clip=0.03)

    # Run Cellpose segmentation
    result = model.eval(img_processed, **eval_kwargs)
    mask = result[0].astype(np.uint8) # convert predicted mask to uint8
    
    #Refine masks by aggressive shrinking
    mask = shrink_masks_aggressive(mask, erosion_iterations=6)

    #Extract valid contours and centroids
    valid_contours = []
    centroids = []

    for label in np.unique(mask):
        if label == 0:  # Skip background
            continue
        
        # Create a binary mask for this nucleus    
        binary_mask = (mask == label).astype(np.uint8) * 255
        
        # Find contours for this nucleus
        cnts, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in cnts:
            # Ignore small objects below min_area, avoid noise
            if cv2.contourArea(cnt) < min_area:
                continue
            valid_contours.append(cnt)

            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                centroids.append((cx, cy))

    return mask, valid_contours, centroids

def select_nucleus_at_point(image_path, click_x, click_y):
    """
    Selects the nucleus that contains a given click point (click_x, click_y) by:
        Loads or computes the segmentation mask and nuclei contours from the image.
        Finds which contour contains the clicked point.
        Returns a mask of the selected nucleus, its contour, and its centroid.
    """
    #Load or segment nuclei (cached)
    mask, contours, centroids = segment_nuclei_cached(image_path)

    selected_mask = np.zeros_like(mask)
    selected_contour = None
    selected_centroid = None

    #Find the nucleus containing the click
    for i, cnt in enumerate(contours):
        if cv2.pointPolygonTest(cnt, (click_x, click_y), False) >= 0:
            selected_contour = cnt
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                selected_centroid = (cx, cy)
            cv2.drawContours(selected_mask, [cnt], -1, 255, -1)
            break

    if selected_contour is None:
        return None, None, None

    return selected_mask, selected_contour, selected_centroid

def segment_nuclei_cached(image_path):
    """
    A wrapper for `segment_nuclei` that uses file-based caching.
    
    If a segmentation mask for the given image already exists in the cache,
    it loads it. Otherwise, it runs segmentation and saves the result to the cache.
    """
    
    #Construct cache file path
    base = os.path.basename(image_path)
    
    # Extract filename without extension
    name = os.path.splitext(base)[0]
    cache_mask_path = os.path.join(CACHE_DIR, f"{name}_mask.npy")

    #CASE 1: Cached mask exists, load it
    if os.path.exists(cache_mask_path):
        mask = np.load(cache_mask_path).astype(np.uint8)

        # Find contours and centroids like in segment_nuclei function (below)
        contours = []
        centroids = []

        for label in np.unique(mask):
            if label == 0:
                continue

            binary = (mask == label).astype(np.uint8) * 255
            cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in cnts:
                contours.append(c)
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cx = int(M["m10"]/M["m00"])
                    cy = int(M["m01"]/M["m00"])
                    centroids.append((cx, cy))

        return mask, contours, centroids

    # CASE 2: Cached mask does NOT exist, run segmentation
    mask, contours, centroids = segment_nuclei(image_path)
    np.save(cache_mask_path, mask)
    return mask, contours, centroids

@app.route('/api/select-nucleus', methods=['POST'])
def select_nucleus():
    """
    Given an (x, y) click coordinate, segments the image and returns the
    contour, centroid, and area of the nucleus at that point.
    """
    # Get data
    data = request.json
    image_filename = data.get("image_filename")
    channel = data.get("channel", "ch002")
    click_x = data.get("click_x")
    click_y = data.get("click_y")

    # Validate the parameter above (is not missing)
    if not all([image_filename, click_x is not None, click_y is not None]):
        return jsonify({"error": "Missing parameters"}), 400

    # Determine folder based on channel
    if channel == "ch002":
        image_path = os.path.join(CH002_FOLDER, image_filename)
    else:
        image_path = os.path.join(CH00_FOLDER, image_filename)

    # Check if file exists
    if not os.path.exists(image_path):
        return jsonify({"error": f"Image not found: {image_path}"}), 404

    # Select nucleus at clicked point by using select_nucleus_at_point function
    mask, contour, centroid = select_nucleus_at_point(image_path, click_x, click_y)
    if mask is None or contour is None:
        return jsonify({"error": "No nucleus found at this point"}), 404

    # Compute nucleus area
    area = int(cv2.contourArea(contour))

    # OpenCV contours are nested arrays: [[x, y]], so flatten to [x, y]
    contour_list = [(int(pt[0][0]), int(pt[0][1])) for pt in contour]

    return jsonify({
        "centroid": {"x": centroid[0], "y": centroid[1]},
        "area": area,
        "contour": contour_list
    })
@app.route('/images/ch00/<filename>')
def serve_ch00_image(filename):
    """
    Serve ch00 (nucleus only) images as JPEG
    """
    return serve_processed_image(CH00_FOLDER, filename)

@app.route('/images/ch002/<filename>')
def serve_ch002_image(filename):
    """
    Serve ch002 (nucleus + cytoplasm) images as JPEG
    """
    return serve_processed_image(CH002_FOLDER, filename)

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

def measure_both_whiteness(image_path, nucleus_mask, cell_mask, white_ratio_factor=0.5, debug=False):
    """
    Measures brightness and CDK2 activity in nucleus and cytoplasm regions.
    """
    import cv2, numpy as np, matplotlib.pyplot as plt

    # Read image
    I = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if I is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # Scale image to 0-255 if needed
    if np.issubdtype(I.dtype, np.floating):
        max_val = I.max()
        if max_val > 0:
            I = (I / max_val * 255).astype(np.float32)
        else:
            I = I.astype(np.float32)
    elif I.dtype == np.uint16:
        I = (I / 65535 * 255).astype(np.float32)
    else:
        I = I.astype(np.float32)

    # Convert masks to boolean
    if nucleus_mask is None or cell_mask is None:
        raise ValueError("nucleus_mask or cell_mask is None")
    nuc_mask = (np.array(nucleus_mask) > 0)
    cell_mask_bin = (np.array(cell_mask) > 0)

    # Cytoplasm mask = cell - nucleus
    cyto_mask = cell_mask_bin & (~nuc_mask)

    # Extract pixel values within masks
    nuc_pixels = I[nuc_mask]
    cyto_pixels = I[cyto_mask]

    if nuc_pixels.size == 0 or cyto_pixels.size == 0:
        print("[WARN] Empty mask detected.")

    # Calculate mean intensity and standard deviation
    mean_nuc = float(np.mean(nuc_pixels)) if nuc_pixels.size > 0 else 0.0
    mean_cyto = float(np.mean(cyto_pixels)) if cyto_pixels.size > 0 else 0.0
    combined_pixels = np.concatenate([nuc_pixels, cyto_pixels])
    std_combined = float(np.std(combined_pixels)) if combined_pixels.size > 0 else 0.0

    # Compute "white threshold" (unused now, for future updating)
    white_threshold = (mean_nuc + mean_cyto) / 2 + white_ratio_factor * std_combined


    # Compute CDK2 activity = mean_cytoplasm / mean_nucleus
    if mean_nuc > 0:
        cdk2_activity = mean_cyto / mean_nuc
    else:
        cdk2_activity = 0.0

    # Show images and masks (for debug)
    if debug:
        print(f"[DEBUG] mean_nuc={mean_nuc:.2f}, mean_cyto={mean_cyto:.2f}")
        print(f"[DEBUG] threshold={white_threshold:.2f}")
        print(f"[DEBUG] CDK2 activity={cdk2_activity:.3f}")

        fig, axes = plt.subplots(1, 3, figsize=(12,4))
        axes[0].imshow(I, cmap='gray'); axes[0].set_title('Original Image (scaled)')
        axes[1].imshow(nuc_mask, cmap='gray'); axes[1].set_title('Nucleus Mask')
        axes[2].imshow(cyto_mask, cmap='gray'); axes[2].set_title('Cytoplasm Mask')
        plt.show()

    return {
        'mean_nucleus': mean_nuc,
        'mean_cytoplasm': mean_cyto,
        'cdk2_activity': cdk2_activity
    }


@app.route('/api/track-cell-brightness', methods=['POST'])
def track_cell_brightness():
    """
    Automatically measures mean brightness & CDK2 activity of nucleus and cytoplasm
    across multiple timepoints. Supports both binary masks or contour polygons as input.
    """
    try:
        data = request.json

        # Get input masks or contours from JSON
        nucleus_mask_input = data.get("nucleusMask")
        cell_mask_input = data.get("cellMask")
        nucleus_contour = data.get("nucleusContour")
        cytoplasm_contour = data.get("cytoplasmContour")
        output_filename = data.get("outputFilename", "tracking_results.xlsx")

        # Convert masks if provided
        if nucleus_mask_input is not None and cell_mask_input is not None:
            # Convert to np.array (uint8) and scale 0/1 to 0/255
            nucleus_mask = np.array(nucleus_mask_input, dtype=np.uint8)
            cell_mask = np.array(cell_mask_input, dtype=np.uint8)
            if np.max(nucleus_mask) <= 1: nucleus_mask *= 255
            if np.max(cell_mask) <= 1: cell_mask *= 255

        # Convert contours to masks if provided
        elif nucleus_contour is not None and cytoplasm_contour is not None:
            # Load first image in CH002_FOLDER to get shape
            cytoplasm_files = sorted([f for f in os.listdir(CH002_FOLDER)
                                      if f.lower().endswith((".png", ".tif", ".jpg"))])
            if not cytoplasm_files:
                return jsonify({"error": "No images found in cytoplasm folder"}), 400

            first_image = cv2.imread(os.path.join(CH002_FOLDER, cytoplasm_files[0]), cv2.IMREAD_GRAYSCALE)
            if first_image is None:
                return jsonify({"error": "Failed to read first cytoplasm image"}), 500
            shape = first_image.shape

            # Helper: convert contour polygon to binary mask
            def contour_to_mask(contour, shape):
                mask = np.zeros(shape, dtype=np.uint8)
                pts = np.array(contour, dtype=np.int32).reshape((-1,1,2))
                cv2.fillPoly(mask, [pts], 255)
                return mask

            nucleus_mask = contour_to_mask(nucleus_contour, shape)
            cell_mask = contour_to_mask(cytoplasm_contour, shape)

        else:
            return jsonify({"error": "Missing mask or contour input"}), 400

        # Get list of cytoplasm images
        cytoplasm_files = sorted([f for f in os.listdir(CH002_FOLDER)
                                  if f.lower().endswith((".png", ".tif", ".jpg"))])
        if not cytoplasm_files:
            return jsonify({"error": "No images found in cytoplasm folder"}), 400

        results = []


        # Loop through images (timepoints)
        for time_idx, filename in enumerate(cytoplasm_files, start=1):
            image_path = os.path.join(CH002_FOLDER, filename)
            if not os.path.exists(image_path):
                print(f"[WARN] Image not found: {filename}, skipping.")
                continue

            # Measure brightness & CDK2 activity using existing function
            stats = measure_both_whiteness(
                image_path=image_path,
                nucleus_mask=nucleus_mask,
                cell_mask=cell_mask,
                white_ratio_factor=0.5,
                debug=False
            )

            results.append([
                time_idx, filename,
                stats['mean_nucleus'],       # Nucleus mean intensity
                stats['mean_cytoplasm'],     # Cytoplasm mean intensity
                stats['cdk2_activity']       # CDK2 activity
            ])

        # Export results to Excel
        os.makedirs(STATIC_FOLDER, exist_ok=True)
        output_path = os.path.join(STATIC_FOLDER, output_filename)
        wb = Workbook()
        ws = wb.active
        ws.title = "Tracking Results"

        headers = [
            "Timepoint", "Filename",
            "Nucleus Mean Brightness",   # could add "Nucleus White Ratio"
            "Cytoplasm Mean Brightness", # could add "Cytoplasm White Ratio"
            "CDK2 Activity"
        ]
        ws.append(headers)

        for row in results:
            ws.append(row)

        wb.save(output_path)


        # Return success response
        return jsonify({"message": "Tracking completed", "file": output_filename})

    except Exception as e:
        print(f"[ERROR] track_cell_brightness failed: {e}")
        return jsonify({"error": str(e)}), 500




@app.route('/api/run-btrack', methods=['POST'])
def run_btrack_analysis():
    """
    Performs BTrack cell tracking and measures nucleus/cytoplasm brightness (CDK2 activity)
    along a tracked path starting from a user click point.
    """
    try:

        # Get input JSON parameters
        data = request.json or {}
        click_x = data.get("click_x")             # X coordinate of user click
        click_y = data.get("click_y")             # Y coordinate of user click
        image_filename = data.get("image_filename")  # Reference image to start tracking
        start = int(data.get("start_timepoint", 0))  # Start frame index
        end = data.get("end_timepoint", None)        # End frame index

        # Validate required parameters
        if click_x is None or click_y is None or not image_filename:
            return {"error": "Missing click_x, click_y or image_filename"}, 400

        # Load and sort CH00 (nucleus) images
        nucleus_files = sorted(
            [f for f in os.listdir(CH00_FOLDER) if allowed_file(f)],
            key=natural_sort_key
        )
        if not nucleus_files:
            return {"error": "No CH00 images found"}, 404

        # Find index of reference image
        try:
            first_frame_index = nucleus_files.index(image_filename)
        except ValueError:
            return {"error": "image_filename not found"}, 404

        # Set end frame if not provided
        if end is None:
            end = len(nucleus_files) - 1

        used_files = nucleus_files[start:end + 1]

        # Segment nuclei for all frames
        all_masks = []
        for fname in used_files:
            imgpath = os.path.join(CH00_FOLDER, fname)
            mask, _, _ = segment_nuclei_cached(imgpath)

            if mask is None:
                # If segmentation fails, create empty mask
                h, w = all_masks[-1].shape if all_masks else (512, 512)
                all_masks.append(np.zeros((h, w), dtype=np.uint16))
                continue

            all_masks.append(mask.astype(np.uint16))

        segmentation_stack = np.stack(all_masks, axis=0)  # 3D stack: (time, height, width)

        # Convert segmentation stack to BTrack objects
        all_objects = btrack.utils.segmentation_to_objects(
            segmentation_stack, properties=("area", "major_axis_length")
        )

        # Run Bayesian tracking (BTrack)
        with btrack.BayesianTracker() as tracker:
            tracker.configure_from_file("config.json")
            tracker.append(all_objects)

            # Set volume bounds for tracking (x, y, z ranges)
            first_img = cv2.imread(os.path.join(CH00_FOLDER, used_files[0]))
            tracker.volume = (
                (0, first_img.shape[1]),  # x-range
                (0, first_img.shape[0]),  # y-range
                (-1e5, 1e5)               # z-range
            )

            tracker.track()      # Perform tracking
            tracker.optimize()   # Refine tracks

            # Identify track corresponding to click
            click_offset = first_frame_index - start
            clicked_label = int(segmentation_stack[click_offset][int(click_y), int(click_x)])
            if clicked_label == 0:
                clicked_label = None  # Clicked background, ignore

            # Find closest track to click point
            best_track = None
            best_dist = 1e20
            for tr in tracker.tracks:
                for i in range(len(tr.x)):
                    d = (tr.x[i] - click_x) ** 2 + (tr.y[i] - click_y) ** 2
                    if d < best_dist:
                        best_dist = d
                        best_track = tr

            # If no track found, create a dummy track
            if best_track is None:
                best_track = type("FakeTrack", (), {})()
                best_track.x = [0] * len(used_files)
                best_track.y = [0] * len(used_files)
                best_track.t = list(range(len(used_files)))
                best_track.ID = 0

        # Segment cytoplasm & measure brightness along the track
        ch002_files = sorted(os.listdir(CH002_FOLDER))
        excel_data = []

        for i in range(len(best_track.x)):
            t = int(best_track.t[i])
            x = int(best_track.x[i])
            y = int(best_track.y[i])

            mask_t = segmentation_stack[t]
            label = int(mask_t[int(round(y)), int(round(x))])

            # If no nucleus detected, record zeros
            if label == 0:
                excel_data.append({
                    "Timepoint": t,
                    "Mean_whiteness_nucleus": 0,
                    "Mean_whiteness_cytoplasm": 0,
                    "CDK2_Activity": 0
                })
                continue

            # Extract nucleus mask
            nucleus_mask = (mask_t == label).astype(np.uint8) * 255
            img_path = os.path.join(CH002_FOLDER, ch002_files[t])

            # Segment cytoplasm from nucleus mask
            cyto_mask = segment_cytoplasm_from_nucleus(img_path, nucleus_mask)

            # If cytoplasm segmentation fails, record zeros
            if cyto_mask is None:
                excel_data.append({
                    "Timepoint": t,
                    "Mean_whiteness_nucleus": 0,
                    "Mean_whiteness_cytoplasm": 0,
                    "CDK2_Activity": 0
                })
                continue

            # Measure brightness & CDK2 activity
            stats = measure_both_whiteness(img_path, nucleus_mask, cyto_mask)

            excel_data.append({
                "Timepoint": t,
                "Mean_whiteness_nucleus": stats["mean_nucleus"],
                "Mean_whiteness_cytoplasm": stats["mean_cytoplasm"],
                "CDK2_Activity": stats["cdk2_activity"]
            })


        # Export results to Excel
        df = pd.DataFrame(excel_data)
        output = BytesIO()
        df.to_excel(output, index=False)
        output.seek(0)

        return send_file(
            output,
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            download_name=f"btrack_results_track{best_track.ID}.xlsx",
            as_attachment=True
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}, 500

@app.route('/download/<path:filename>')
def download_file(filename):
    return send_from_directory(STATIC_FOLDER, filename, as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)