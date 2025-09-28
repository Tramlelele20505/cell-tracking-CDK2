#!/usr/bin/env python3
"""
Cell Image Viewer Web Application
 
 Supports uploading cell nucleus images and corresponding nucleus+cytoplasm images
"""

from flask import Flask, render_template, jsonify, request, send_from_directory, Response, flash, redirect, url_for
from werkzeug.utils import secure_filename
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
import uuid
import shutil
import base64 # For encoding/decoding base64 images
from flask import send_from_directory
import os






app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production

# Configuration
UPLOAD_FOLDER = "uploads"
CH00_FOLDER = os.path.join(UPLOAD_FOLDER, "ch00")  # Nucleus only images
CH002_FOLDER = os.path.join(UPLOAD_FOLDER, "ch002")  # Nucleus + cytoplasm images
STATIC_FOLDER = "static"
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2 GB max upload size

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CH00_FOLDER, exist_ok=True)
os.makedirs(CH002_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'tif', 'tiff', 'jpg', 'jpeg', 'png', 'bmp'}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
             if allowed_file(file)],
            key=natural_sort_key
        )
        print(f"Found {len(ch00_files)} ch00 files")
    
    # Get ALL ch002 files (nucleus + cytoplasm) with natural sorting
    if os.path.exists(CH002_FOLDER):
        ch002_files = sorted(
            [file for file in os.listdir(CH002_FOLDER) 
             if allowed_file(file)],
            key=natural_sort_key
        )
        print(f"Found {len(ch002_files)} ch002 files")
    
    # Build lookup by base name for ch002 (ignore extension)
    ch002_by_base = {os.path.splitext(f)[0]: f for f in ch002_files}

    # Attempt to pair by exact base name first
    image_pairs = OrderedDict()
    for ch00_file in ch00_files:
        base_name = os.path.splitext(ch00_file)[0]
        ch002_file = ch002_by_base.get(base_name)
        if ch002_file:
            image_pairs[base_name] = {
                'nucleus': ch00_file,
                'nucleus_cytoplasm': ch002_file
            }

    # If no pairs (or very few) were found, fall back to index-based pairing
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

@app.route('/')
def index():
    """
    Main page with upload interface and random cell selection
    """
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_files():
    """
    Handle folder uploads for both channels
    """
    if request.method == 'POST':
        # Gather uploaded lists (may be missing or empty for one channel)
        ch00_files = request.files.getlist('ch00_files') if 'ch00_files' in request.files else []
        ch002_files = request.files.getlist('ch002_files') if 'ch002_files' in request.files else []
        
        if not ch00_files and not ch002_files:
            flash('No folders selected. Please choose at least one channel folder to upload.')
            return redirect(request.url)
        
        uploaded_count = 0
        ch00_image_count = 0
        ch002_image_count = 0
        
        # Track files saved this request for cleanup if needed
        saved_file_paths = []

        # Process ch00 files (nucleus only) - filter for image files only
        try:
            for file in ch00_files:
                if not file or not file.filename:
                    continue
                # Strip any provided path and sanitize the name
                original_name = os.path.basename(file.filename)
                safe_name = secure_filename(original_name)
                if not allowed_file(safe_name):
                    continue
                filepath = os.path.join(CH00_FOLDER, safe_name)
                # Ensure upload directories exist
                os.makedirs(CH00_FOLDER, exist_ok=True)
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
        
        # Process ch002 files (nucleus + cytoplasm) - filter for image files only
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
    Clear all uploaded files
    """
    try:
        # Remove all files from upload folders
        for folder in [CH00_FOLDER, CH002_FOLDER]:
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

def segment_cytoplasm_from_nucleus(cytoplasm_img_path, nucleus_mask,
                                   denoise_h=10,
                                   dilate_px=3,        # dilate nhân 5-10 px
                                   bright_thresh=30,
                                   dark_thresh=20,
                                   dark_region_ratio=0.1):
    """
    Segment cytoplasm quanh nucleus bằng region growing có kiểm tra hố tối.
    Nhân được dilate và coi là vùng sáng để region growing luôn bao quanh nhân.
    Trả về final_mask (uint8 0/255).
    """
    import cv2
    import numpy as np

    # 1️⃣ Đọc ảnh & khử nhiễu
    img = cv2.imread(cytoplasm_img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    blur = cv2.fastNlMeansDenoising(img, None, h=denoise_h,
                                    templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    blur = clahe.apply(blur)

    h, w = img.shape

    # 2️⃣ Dilate nhân 5–10 px để seed lớn hơn nhân
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    dilated_nucleus = cv2.dilate(nucleus_mask, kernel, iterations=dilate_px)

    # 3️⃣ Gán toàn bộ vùng nhân + dilate là sáng (255)
    blur[dilated_nucleus > 0] = 255

    # 4️⃣ Region growing
    visited = np.zeros_like(img, dtype=np.uint8)
    mask = np.zeros_like(img, dtype=np.uint8)

    seed_pts = np.column_stack(np.where(dilated_nucleus > 0))
    stack = [tuple(pt[::-1]) for pt in seed_pts]  # (x, y)

    while stack:
        x, y = stack.pop()
        if not (0 <= x < w and 0 <= y < h):
            continue
        if visited[y, x]:
            continue
        visited[y, x] = 1

        val = blur[y, x]

        if val >= bright_thresh:  # cytoplasm sáng
            mask[y, x] = 255
            for dx in [-1,0,1]:
                for dy in [-1,0,1]:
                    if dx == 0 and dy == 0:
                        continue
                    stack.append((x+dx, y+dy))

        elif val <= dark_thresh:
            # kiểm tra hố tối
            neigh = blur[max(0,y-2):min(h,y+3), max(0,x-2):min(w,x+3)]
            dark_ratio = np.mean(neigh < dark_thresh)
            if dark_ratio < dark_region_ratio:  # chỉ là nhiễu
                mask[y, x] = 255
                for dx in [-1,0,1]:
                    for dy in [-1,0,1]:
                        if dx == 0 and dy == 0:
                            continue
                        stack.append((x+dx, y+dy))
            # else: hố tối thật → dừng lại

    # 5️⃣ Hậu xử lý
    clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel, iterations=1)

    # Lấy contour lớn nhất
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    max_cnt = max(contours, key=cv2.contourArea)
    final_mask = np.zeros_like(img, dtype=np.uint8)
    cv2.drawContours(final_mask, [max_cnt], -1, 255, -1)

    # Làm mềm biên
    blurred = cv2.GaussianBlur(final_mask, (5,5), 0)
    _, final_mask = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

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

    # build mask từ nucleus contour
    nucleus_mask = np.zeros_like(img, dtype=np.uint8)
    pts = np.array(nucleus_contour, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(nucleus_mask, [pts], 255)

    # chạy cytoplasm segmentation
    cytoplasm_mask = segment_cytoplasm_from_nucleus(image_path, nucleus_mask)
    if cytoplasm_mask is None:
        return jsonify({"error": "Failed to segment cytoplasm"}), 500

    # convert sang contour polygon
    contours, _ = cv2.findContours(cytoplasm_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return jsonify({"error": "No cytoplasm contour found"}), 500

    max_cnt = max(contours, key=cv2.contourArea)
    eps = 0.01 * cv2.arcLength(max_cnt, True)
    approx = cv2.approxPolyDP(max_cnt, eps, True)
    cyto_contour = [(int(pt[0][0]), int(pt[0][1])) for pt in approx]

    return jsonify({"cytoplasm_contour": cyto_contour})



def fill_inner_holes(mask, gray_img, circularity_thresh=0.6, area_ratio_thresh=0.3, elongation_thresh=3.0):
    """
    Lấp lỗ bên trong nhân nhưng bỏ qua rãnh dài giữa các nhân.
    Dựa trên độ tròn + tỷ lệ diện tích + độ đậm bên trong + elongation.
    """
    mask_filled = mask.copy()
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    if hierarchy is not None:
        for i, h in enumerate(hierarchy[0]):
            parent_idx = h[3]
            if parent_idx != -1:  # contour con
                area = cv2.contourArea(contours[i])
                perimeter = cv2.arcLength(contours[i], True)
                circularity = 4 * np.pi * area / (perimeter**2 + 1e-6)

                # contour cha (nhân chứa nó)
                parent_area = cv2.contourArea(contours[parent_idx]) if parent_idx >= 0 else 1e9
                area_ratio = area / (parent_area + 1e-6)

                # elongation = chiều dài lớn nhất / chiều dài nhỏ nhất
                x, y, w, h_rect = cv2.boundingRect(contours[i])
                elongation = max(w, h_rect) / (min(w, h_rect) + 1e-6)

                # trung bình mức xám bên trong lỗ
                mask_hole = np.zeros_like(mask)
                cv2.drawContours(mask_hole, [contours[i]], -1, 255, -1)
                mean_val = cv2.mean(gray_img, mask=mask_hole)[0]

                # Điều kiện: hình tròn + nhỏ so với cha + không quá tối + không dài mảnh
                if circularity > circularity_thresh and area_ratio < area_ratio_thresh \
                   and mean_val > 30 and elongation < elongation_thresh:
                    cv2.drawContours(mask_filled, [contours[i]], -1, 255, -1)

    return mask_filled


def segment_nuclei(image_path, min_area=50):
    import cv2
    import numpy as np

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, [], []

    # 1️⃣ Tăng tương phản
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(img)

    # 2️⃣ Ngưỡng hóa (Otsu)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3️⃣ Morphology nhẹ để loại nhiễu
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # 4️⃣ Lấp lỗ trong nhân (truyền enhanced để phân biệt lỗ thật / rãnh)
    mask_filled = fill_inner_holes(mask, enhanced)

    # 5️⃣ Lấy contours & centroid
    contours, _ = cv2.findContours(mask_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours, centroids = [], []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        valid_contours.append(cnt)
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])
            centroids.append((cx, cy))

    return mask_filled, valid_contours, centroids








def select_nucleus_at_point(image_path, click_x, click_y):
    """
    Chọn nhân chứa điểm click (click_x, click_y)
    Trả về: mask của nhân, contour, centroid
    """
    mask, contours, centroids = segment_nuclei(image_path)
    selected_mask = np.zeros_like(mask)
    selected_contour = None
    selected_centroid = None

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
@app.route('/api/select-nucleus', methods=['POST'])
def select_nucleus():
    """
    Nhận click (x, y) từ client và trả về contour + centroid + diện tích của nhân chứa điểm đó
    """
    data = request.json
    image_filename = data.get("image_filename")
    channel = data.get("channel", "ch002")
    click_x = data.get("click_x")
    click_y = data.get("click_y")

    if not all([image_filename, click_x is not None, click_y is not None]):
        return jsonify({"error": "Missing parameters"}), 400

    # Chọn folder theo channel
    if channel == "ch002":
        image_path = os.path.join(CH002_FOLDER, image_filename)
    else:
        image_path = os.path.join(CH00_FOLDER, image_filename)

    if not os.path.exists(image_path):
        return jsonify({"error": f"Image not found: {image_path}"}), 404

    mask, contour, centroid = select_nucleus_at_point(image_path, click_x, click_y)
    if mask is None or contour is None:
        return jsonify({"error": "No nucleus found at this point"}), 404

    area = int(cv2.contourArea(contour))

    # Convert contour sang list (danh sách [x, y])
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
    try:
        # Load image
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
        # Load image
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
    """
    Đo độ sáng trung bình và tỉ lệ trắng của vùng nhân & bào tương.
    Hỗ trợ vùng chọn hình chữ nhật hoặc polygon.
    """
    data = request.json
    nucleus_path = data.get("nucleusPath")
    cytoplasm_path = data.get("cytoplasmPath")
    nucleus_selection = data.get("nucleusSelection")
    cytoplasm_selection = data.get("cytoplasmSelection")

    # Build absolute paths
    nucleus_filename = os.path.basename(nucleus_path)
    cytoplasm_filename = os.path.basename(cytoplasm_path)
    nucleus_abs = os.path.join(CH00_FOLDER, nucleus_filename)
    cytoplasm_abs = os.path.join(CH002_FOLDER, cytoplasm_filename)

    if not os.path.exists(nucleus_abs):
        return jsonify({"error": f"Nucleus image not found at {nucleus_abs}"}), 400
    if not os.path.exists(cytoplasm_abs):
        return jsonify({"error": f"Cytoplasm image not found at {cytoplasm_abs}"}), 400

    # Đọc ảnh bào tương để đo độ sáng (dùng cùng ảnh cho cả nhân & bào tương)
    img = cv2.imread(cytoplasm_abs, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return jsonify({"error": "Failed to read cytoplasm image"}), 400

    h, w = img.shape

    def polygon_to_mask(points):
        """Convert polygon (percent coords) thành mask binary"""
        if not points:
            return None
        pts = []
        for p in points:
            px = int(np.clip(p.get("x", 0), 0, 100) / 100 * w)
            py = int(np.clip(p.get("y", 0), 0, 100) / 100 * h)
            pts.append([px, py])
        if len(pts) < 3:
            return None
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(pts, np.int32)], 255)
        return mask

    def rect_to_mask(sel):
        """Convert rectangle (percent coords) thành mask binary"""
        x1 = int(sel["x"] / 100 * w)
        y1 = int(sel["y"] / 100 * h)
        x2 = int((sel["x"] + sel["width"]) / 100 * w)
        y2 = int((sel["y"] + sel["height"]) / 100 * h)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        return mask

    def measure_region(mask):
        """Trả về trung bình độ sáng và tỉ lệ trắng trong mask"""
        values = img[mask > 0]
        if values.size == 0:
            return None
        mean_brightness = float(np.mean(values))
        white_ratio = float(np.sum(values > 200)) / float(values.size)
        return {
            "mean_brightness": mean_brightness,
            "white_ratio": white_ratio
        }

    results = {}

    # Nucleus
    if nucleus_selection:
        if "points" in nucleus_selection:
            mask = polygon_to_mask(nucleus_selection["points"])
        else:
            mask = rect_to_mask(nucleus_selection)
        if mask is not None:
            s = measure_region(mask)
            if s:
                results["nucleus"] = s

    # Cytoplasm
    if cytoplasm_selection:
        if "points" in cytoplasm_selection:
            mask = polygon_to_mask(cytoplasm_selection["points"])
        else:
            mask = rect_to_mask(cytoplasm_selection)
        if mask is not None:
            s = measure_region(mask)
            if s:
                results["cytoplasm"] = s

    return jsonify(results)
@app.route('/api/track-cell-brightness', methods=['POST'])
def track_cell_brightness():
    """
    Track mean & whiteness của 1 tế bào qua nhiều timepoint, export Excel.
    """
    import datetime
    data = request.json
    nucleus_contour = data.get("nucleus_contour")
    cytoplasm_contour = data.get("cytoplasm_contour")
    start_tp = int(data.get("start_timepoint"))
    end_tp = int(data.get("end_timepoint"))

    if not nucleus_contour or not cytoplasm_contour:
        return jsonify({"error": "Missing contours"}), 400

    cytoplasm_files = sorted(
        [f for f in os.listdir(CH002_FOLDER) if f.lower().endswith((".png", ".tif", ".jpg"))]
    )
    total_tp = len(cytoplasm_files)

    if start_tp < 1 or end_tp > total_tp:
        return jsonify({"error": f"Timepoint range must be 1–{total_tp}"}), 400

    def contour_to_mask(contour, shape):
        mask = np.zeros(shape, dtype=np.uint8)
        pts = np.array(contour, np.int32)
        cv2.fillPoly(mask, [pts], 255)
        return mask

    results = []
    for t in range(start_tp-1, end_tp):
        filename = cytoplasm_files[t]
        img_path = os.path.join(CH002_FOLDER, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        nucleus_mask = contour_to_mask(nucleus_contour, img.shape)
        cytoplasm_mask = contour_to_mask(cytoplasm_contour, img.shape)

        nucleus_vals = img[nucleus_mask > 0]
        cyto_vals = img[cytoplasm_mask > 0]

        def stats(values):
            if values.size == 0:
                return 0.0, 0.0
            mean = float(np.mean(values))
            white_ratio = float(np.sum(values > 200)) / float(values.size)
            return mean, white_ratio

        n_mean, n_white = stats(nucleus_vals)
        c_mean, c_white = stats(cyto_vals)

        results.append([t+1, filename, n_mean, n_white, c_mean, c_white])

    # Export Excel
    wb = Workbook()
    ws = wb.active
    ws.title = "Brightness Tracking"
    ws.append([
        "Timepoint", "Filename",
        "Nucleus Mean Brightness", "Nucleus White Ratio",
        "Cytoplasm Mean Brightness", "Cytoplasm White Ratio"
    ])
    for row in results:
        ws.append(row)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"cell_tracking_{timestamp}.xlsx"
    filepath = os.path.join(STATIC_FOLDER, filename)
    os.makedirs(STATIC_FOLDER, exist_ok=True)
    wb.save(filepath)

    return jsonify({"message": "done", "filename": filename})



@app.route('/api/auto-track', methods=['POST'])
def auto_track():
    """
    Tự động đo độ sáng trung bình & tỉ lệ trắng của nucleus & cytoplasm qua các timepoint.
    """
    try:
        data = request.json
        nucleus_selection = data.get("nucleusSelection")
        cytoplasm_selection = data.get("cytoplasmSelection")
        output_filename = data.get("outputFilename", "tracking_results.xlsx")

        # Kiểm tra vùng chọn
        if not nucleus_selection or not cytoplasm_selection:
            return jsonify({"error": "Missing nucleus or cytoplasm selection"}), 400

        # Lấy danh sách ảnh timepoint (sắp xếp theo tên)
        cytoplasm_files = sorted(
            [f for f in os.listdir(CH002_FOLDER) if f.lower().endswith((".png", ".tif", ".jpg"))]
        )
        if not cytoplasm_files:
            return jsonify({"error": "No images found in cytoplasm folder"}), 400

        results = []

        def measure_region_stats(crop):
            """Tính trung bình & tỉ lệ trắng trong một vùng ảnh"""
            if crop.size == 0:
                return 0.0, 0.0
            mean_brightness = float(np.mean(crop))
            white_ratio = float(np.sum(crop > 200)) / float(crop.size)
            return mean_brightness, white_ratio

        # Lặp qua các timepoint
        for time_idx, filename in enumerate(cytoplasm_files, start=1):
            cytoplasm_path = os.path.join(CH002_FOLDER, filename)
            cytoplasm_img = cv2.imread(cytoplasm_path, cv2.IMREAD_GRAYSCALE)
            if cytoplasm_img is None:
                print(f"[WARN] Could not read {filename}, skipping.")
                continue

            h, w = cytoplasm_img.shape

            # --- Nucleus ---
            x1 = int(nucleus_selection["x"] / 100 * w)
            y1 = int(nucleus_selection["y"] / 100 * h)
            x2 = x1 + int(nucleus_selection["width"] / 100 * w)
            y2 = y1 + int(nucleus_selection["height"] / 100 * h)
            nucleus_crop = cytoplasm_img[y1:y2, x1:x2]
            nucleus_mean, nucleus_white_ratio = measure_region_stats(nucleus_crop)

            # --- Cytoplasm ---
            x1 = int(cytoplasm_selection["x"] / 100 * w)
            y1 = int(cytoplasm_selection["y"] / 100 * h)
            x2 = x1 + int(cytoplasm_selection["width"] / 100 * w)
            y2 = y1 + int(cytoplasm_selection["height"] / 100 * h)
            cytoplasm_crop = cytoplasm_img[y1:y2, x1:x2]
            cytoplasm_mean, cytoplasm_white_ratio = measure_region_stats(cytoplasm_crop)

            results.append([
                time_idx, filename,
                nucleus_mean, nucleus_white_ratio,
                cytoplasm_mean, cytoplasm_white_ratio
            ])

        # --- Xuất ra Excel ---
        output_path = os.path.join(STATIC_FOLDER, output_filename)
        wb = Workbook()
        ws = wb.active
        ws.title = "Tracking Results"

        headers = [
            "Timepoint", "Filename",
            "Nucleus Mean Brightness", "Nucleus White Ratio",
            "Cytoplasm Mean Brightness", "Cytoplasm White Ratio"
        ]
        ws.append(headers)

        for row in results:
            ws.append(row)

        wb.save(output_path)
        print(f"[INFO] Tracking results saved to {output_path}")

        return jsonify({"message": "Tracking completed", "file": output_filename})

    except Exception as e:
        print(f"[ERROR] auto_track failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/download/<path:filename>')
def download_file(filename):
    return send_from_directory(STATIC_FOLDER, filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
