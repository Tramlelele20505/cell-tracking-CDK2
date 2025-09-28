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
    Sau khi tìm mask, lùi vào khoảng 3 pixel để tránh lan ra nền tối.
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

    # 6️⃣ Lùi mask vào khoảng 3 pixel để tránh viền lan ra nền
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    final_mask = cv2.erode(final_mask, erode_kernel, iterations=1)

    # 7️⃣ Làm mềm biên
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




def measure_both_whiteness(image_path, nucleus_mask, cell_mask, white_ratio_factor=0.5, debug=False):
    """
    Đo độ sáng trung bình & tỉ lệ pixel sáng trong nhân và bào tương.
    - Tự động scale ảnh về 8-bit nếu ảnh gốc là 16-bit hoặc float với giá trị >255.
    - Bào tương = mask cytoplasm - mask nhân.
    - White ratio tính theo ngưỡng chung cho cả nhân và bào tương:
        threshold = trung bình (mean_nuc + mean_cyto) + white_ratio_factor * std_combined
    """
    import cv2, numpy as np, matplotlib.pyplot as plt

    # 1️⃣ Đọc ảnh
    I = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if I is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # 2️⃣ Scale ảnh về 0-255 nếu cần
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

    # 3️⃣ Convert mask sang boolean
    if nucleus_mask is None or cell_mask is None:
        raise ValueError("nucleus_mask or cell_mask is None")
    nuc_mask = (np.array(nucleus_mask) > 0)
    cell_mask_bin = (np.array(cell_mask) > 0)

    # 4️⃣ Mask bào tương = cell - nucleus
    cyto_mask = cell_mask_bin & (~nuc_mask)

    # 5️⃣ Lấy pixel trong mask
    nuc_pixels = I[nuc_mask]
    cyto_pixels = I[cyto_mask]

    if nuc_pixels.size == 0 or cyto_pixels.size == 0:
        print("[WARN] Empty mask detected.")

    # 6️⃣ Tính mean & std
    mean_nuc = float(np.mean(nuc_pixels)) if nuc_pixels.size > 0 else 0.0
    mean_cyto = float(np.mean(cyto_pixels)) if cyto_pixels.size > 0 else 0.0
    combined_pixels = np.concatenate([nuc_pixels, cyto_pixels])
    std_combined = float(np.std(combined_pixels)) if combined_pixels.size > 0 else 0.0

    # 7️⃣ Tính ngưỡng chung & ratio
    white_threshold = (mean_nuc + mean_cyto) / 2 + white_ratio_factor * std_combined
    white_ratio_nuc = float(np.sum(nuc_pixels > white_threshold)) / max(nuc_pixels.size, 1)
    white_ratio_cyto = float(np.sum(cyto_pixels > white_threshold)) / max(cyto_pixels.size, 1)

    # 8️⃣ Debug
    if debug:
        print(f"[DEBUG] Image dtype: {I.dtype}, min={I.min()}, max={I.max()}")
        print(f"[DEBUG] mean_nuc={mean_nuc:.2f}, mean_cyto={mean_cyto:.2f}, threshold={white_threshold:.2f}")
        print(f"[DEBUG] white_ratio_nuc={white_ratio_nuc:.3f}, white_ratio_cyto={white_ratio_cyto:.3f}")

        fig, axes = plt.subplots(1, 3, figsize=(12,4))
        axes[0].imshow(I, cmap='gray'); axes[0].set_title('Original Image (scaled)')
        axes[1].imshow(nuc_mask, cmap='gray'); axes[1].set_title('Nucleus Mask')
        axes[2].imshow(cyto_mask, cmap='gray'); axes[2].set_title('Cytoplasm Mask')
        plt.show()

    return {
        'mean_nucleus': mean_nuc,
        'mean_cytoplasm': mean_cyto,
        'white_ratio_nucleus': white_ratio_nuc,
        'white_ratio_cytoplasm': white_ratio_cyto
    }




@app.route('/api/track-cell-brightness', methods=['POST'])
def track_cell_brightness():
    """
    Tự động đo độ sáng trung bình & tỉ lệ trắng của nucleus & cytoplasm qua các timepoint,
    hỗ trợ cả mask nhị phân hoặc contour polygon.
    """
    try:
        data = request.json

        # --- Lấy mask hoặc contour từ JSON ---
        nucleus_mask_input = data.get("nucleusMask")
        cell_mask_input = data.get("cellMask")
        nucleus_contour = data.get("nucleusContour")
        cytoplasm_contour = data.get("cytoplasmContour")
        output_filename = data.get("outputFilename", "tracking_results.xlsx")

        # --- Xử lý mask ---
        if nucleus_mask_input is not None and cell_mask_input is not None:
            # Convert sang np.array uint8, nhân 255 nếu JSON gửi 0/1
            nucleus_mask = np.array(nucleus_mask_input, dtype=np.uint8)
            cell_mask = np.array(cell_mask_input, dtype=np.uint8)
            if np.max(nucleus_mask) <= 1: nucleus_mask *= 255
            if np.max(cell_mask) <= 1: cell_mask *= 255
        elif nucleus_contour is not None and cytoplasm_contour is not None:
            # Lấy kích thước từ ảnh đầu tiên trong CH002_FOLDER
            cytoplasm_files = sorted([f for f in os.listdir(CH002_FOLDER)
                                      if f.lower().endswith((".png", ".tif", ".jpg"))])
            if not cytoplasm_files:
                return jsonify({"error": "No images found in cytoplasm folder"}), 400
            first_image = cv2.imread(os.path.join(CH002_FOLDER, cytoplasm_files[0]), cv2.IMREAD_GRAYSCALE)
            if first_image is None:
                return jsonify({"error": "Failed to read first cytoplasm image"}), 500
            shape = first_image.shape

            # Convert contour sang mask
            def contour_to_mask(contour, shape):
                mask = np.zeros(shape, dtype=np.uint8)
                pts = np.array(contour, dtype=np.int32).reshape((-1,1,2))
                cv2.fillPoly(mask, [pts], 255)
                return mask

            nucleus_mask = contour_to_mask(nucleus_contour, shape)
            cell_mask = contour_to_mask(cytoplasm_contour, shape)
        else:
            return jsonify({"error": "Missing mask or contour input"}), 400

        # --- Lấy danh sách ảnh ---
        cytoplasm_files = sorted([f for f in os.listdir(CH002_FOLDER)
                                  if f.lower().endswith((".png", ".tif", ".jpg"))])
        if not cytoplasm_files:
            return jsonify({"error": "No images found in cytoplasm folder"}), 400

        results = []

        # --- Lặp qua từng timepoint ---
        for time_idx, filename in enumerate(cytoplasm_files, start=1):
            image_path = os.path.join(CH002_FOLDER, filename)
            if not os.path.exists(image_path):
                print(f"[WARN] Image not found: {filename}, skipping.")
                continue

            stats = measure_both_whiteness(
                image_path=image_path,
                nucleus_mask=nucleus_mask,
                cell_mask=cell_mask,
                white_ratio_factor=0.5,
                debug=False
            )

            results.append([
                time_idx, filename,
                stats['mean_nucleus'], stats['white_ratio_nucleus'],
                stats['mean_cytoplasm'], stats['white_ratio_cytoplasm']
            ])

        # --- Xuất Excel ---
        os.makedirs(STATIC_FOLDER, exist_ok=True)
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

        return jsonify({"message": "Tracking completed", "file": output_filename})

    except Exception as e:
        print(f"[ERROR] track_cell_brightness failed: {e}")
        return jsonify({"error": str(e)}), 500




@app.route('/download/<path:filename>')
def download_file(filename):
    return send_from_directory(STATIC_FOLDER, filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
