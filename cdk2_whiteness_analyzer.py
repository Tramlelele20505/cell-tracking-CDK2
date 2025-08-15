#!/usr/bin/env python3
"""
CDK2 Activity Analyzer using Whiteness Measurement with skimage.measure.regionprops
Based on Cappell et al. (2016) methodology using grayscale intensity measurements
"""

import cv2
import numpy as np
from scipy import ndimage
from typing import Dict, Tuple, List, Optional
import json
from datetime import datetime
import matplotlib.pyplot as plt
from skimage.measure import regionprops, label
from skimage.filters import threshold_otsu, threshold_adaptive
from skimage.morphology import remove_small_objects, binary_closing, binary_opening
from skimage.segmentation import watershed
from skimage.feature import peak_local_maxima

class CDK2WhitenessAnalyzer:
    """
    CDK2 activity analyzer using nucleus mask from ch00 and intensity measurements on ch002.
    Implements two cytoplasm strategies:
      - "ring": annulus outside nucleus (as in Cappell pipeline)
      - "whole_cell": watershed-based whole-cell mask on ch002, seeded by the ch00 nucleus
    """

    def __init__(self):
        self.measurement_history = []
        self.cell_tracks = {}

    # -------------------- generic helpers --------------------

    @staticmethod
    def _to_pixels(sel: Dict, shape_hw: Tuple[int,int]) -> Tuple[int,int,int,int]:
        h, w = shape_hw
        x = int(sel["x"] / 100.0 * w)
        y = int(sel["y"] / 100.0 * h)
        ww = int(sel["width"] / 100.0 * w)
        hh = int(sel["height"] / 100.0 * h)
        x = max(0, min(w-1, x)); y = max(0, min(h-1, y))
        return x, y, max(1, ww), max(1, hh)

    @staticmethod
    def _remove_small_objects(mask: np.ndarray, min_size: int = 20) -> np.ndarray:
        lbl, n = ndimage.label(mask.astype(np.uint8))
        if n == 0:
            return mask.astype(bool)
        sizes = np.bincount(lbl.ravel())
        remove = sizes < min_size
        remove[0] = False
        cleaned = mask.copy()
        cleaned[remove[lbl]] = 0
        return cleaned.astype(bool)

    @staticmethod
    def _largest_component(mask: np.ndarray) -> np.ndarray:
        lbl, n = ndimage.label(mask.astype(np.uint8))
        if n == 0:
            return np.zeros_like(mask, dtype=bool)
        areas = ndimage.sum(mask, lbl, index=np.arange(1, n+1))
        k = int(np.argmax(areas)) + 1
        return (lbl == k)

    def measure_whiteness_regionprops(self, image: np.ndarray, mask: np.ndarray = None,
                                     background_threshold: int = 10) -> Dict:
        """
        Measure whiteness using skimage.measure.regionprops for robust statistical analysis.
        This provides more accurate mean, median, and other statistical measures.
        
        Args:
            image: Input grayscale image
            mask: Optional boolean mask for region of interest
            background_threshold: Minimum intensity to consider (excludes black background)
            
        Returns:
            Dictionary with comprehensive whiteness statistics using regionprops
        """
        if mask is not None:
            # Apply mask to get only the region of interest
            masked_image = image.copy()
            masked_image[~mask] = 0  # Set non-mask regions to 0
        else:
            # Use entire image
            masked_image = image.copy()
        
        # Create a binary mask for regionprops (excluding background)
        binary_mask = masked_image > background_threshold
        
        if not np.any(binary_mask):
            # If no foreground pixels found, return zeros
            return {
                'mean_intensity': 0.0,
                'median_intensity': 0.0,
                'std_intensity': 0.0,
                'min_intensity': 0.0,
                'max_intensity': 0.0,
                'pixel_count': 0,
                'foreground_pixel_count': 0,
                'background_pixel_count': np.sum(mask) if mask is not None else image.size,
                'area': 0,
                'eccentricity': 0.0,
                'solidity': 0.0,
                'extent': 0.0,
                'perimeter': 0.0,
                'centroid': (0.0, 0.0)
            }
        
        # Label connected components
        labeled_mask = label(binary_mask)
        
        # Get region properties
        regions = regionprops(labeled_mask, intensity_image=masked_image)
        
        if not regions:
            # Fallback to basic measurement if regionprops fails
            foreground_pixels = masked_image[masked_image > background_threshold]
            if len(foreground_pixels) == 0:
                return {
                    'mean_intensity': 0.0,
                    'median_intensity': 0.0,
                    'std_intensity': 0.0,
                    'min_intensity': 0.0,
                    'max_intensity': 0.0,
                    'pixel_count': np.sum(mask) if mask is not None else image.size,
                    'foreground_pixel_count': 0,
                    'background_pixel_count': np.sum(mask) if mask is not None else image.size,
                    'area': 0,
                    'eccentricity': 0.0,
                    'solidity': 0.0,
                    'extent': 0.0,
                    'perimeter': 0.0,
                    'centroid': (0.0, 0.0)
                }
            
            return {
                'mean_intensity': float(np.mean(foreground_pixels)),
                'median_intensity': float(np.median(foreground_pixels)),
                'std_intensity': float(np.std(foreground_pixels)),
                'min_intensity': float(np.min(foreground_pixels)),
                'max_intensity': float(np.max(foreground_pixels)),
                'pixel_count': np.sum(mask) if mask is not None else image.size,
                'foreground_pixel_count': len(foreground_pixels),
                'background_pixel_count': (np.sum(mask) if mask is not None else image.size) - len(foreground_pixels),
                'area': len(foreground_pixels),
                'eccentricity': 0.0,
                'solidity': 0.0,
                'extent': 0.0,
                'perimeter': 0.0,
                'centroid': (0.0, 0.0)
            }
        
        # Use the largest region (most significant)
        largest_region = max(regions, key=lambda r: r.area)
        
        # Extract intensity statistics from regionprops
        mean_intensity = float(largest_region.mean_intensity)
        median_intensity = float(np.median(largest_region.intensity_image[largest_region.intensity_image > 0]))
        std_intensity = float(largest_region.std_intensity)
        min_intensity = float(np.min(largest_region.intensity_image[largest_region.intensity_image > 0]))
        max_intensity = float(np.max(largest_region.intensity_image[largest_region.intensity_image > 0]))
        
        return {
            'mean_intensity': mean_intensity,
            'median_intensity': median_intensity,
            'std_intensity': std_intensity,
            'min_intensity': min_intensity,
            'max_intensity': max_intensity,
            'pixel_count': np.sum(mask) if mask is not None else image.size,
            'foreground_pixel_count': int(largest_region.area),
            'background_pixel_count': (np.sum(mask) if mask is not None else image.size) - int(largest_region.area),
            'area': int(largest_region.area),
            'eccentricity': float(largest_region.eccentricity),
            'solidity': float(largest_region.solidity),
            'extent': float(largest_region.extent),
            'perimeter': float(largest_region.perimeter),
            'centroid': tuple(largest_region.centroid)
        }

    def measure_whiteness(self, image: np.ndarray, mask: np.ndarray = None,
                         background_threshold: int = 10) -> Dict:
        """
        Backward compatibility method - now uses regionprops for better accuracy
        """
        return self.measure_whiteness_regionprops(image, mask, background_threshold)

    # -------------------- segmentation --------------------

    def segment_nucleus_from_ch00(self, ch00_image: np.ndarray, selection: Dict,
                                  method: str = 'otsu') -> Tuple[np.ndarray, Tuple[int,int,int,int]]:
        """
        Segment nucleus from ch00 (nucleus only) image using skimage functions
        Returns nucleus boolean mask cropped to the selection box (same size as crop).
        """
        x, y, w, h = self._to_pixels(selection, ch00_image.shape)
        crop = ch00_image[y:y+h, x:x+w]

        if method == 'otsu':
            # Use skimage Otsu thresholding
            threshold = threshold_otsu(crop)
            mask = crop > threshold
        elif method == 'adaptive':
            # Use skimage adaptive thresholding
            mask = threshold_adaptive(crop, block_size=11, offset=2)
        elif method == 'watershed':
            # Use skimage watershed segmentation
            # Preprocessing
            from skimage.filters import gaussian
            blur = gaussian(crop, sigma=1)
            threshold = threshold_otsu(blur)
            binary = blur > threshold
            
            # Clean up binary image
            binary = remove_small_objects(binary, min_size=20)
            binary = binary_closing(binary)
            binary = binary_opening(binary)
            
            # Watershed segmentation
            from skimage.feature import peak_local_maxima
            from scipy import ndimage as ndi
            
            # Distance transform
            distance = ndi.distance_transform_edt(binary)
            
            # Find peaks
            coords = peak_local_maxima(distance, footprint=np.ones((3, 3)), labels=binary)
            mask = np.zeros(distance.shape, dtype=bool)
            mask[tuple(coords.T)] = True
            
            # Watershed
            markers = label(mask)
            labels = watershed(-distance, markers, mask=binary)
            mask = labels > 0
        else:
            raise ValueError(f"Unknown segmentation method: {method}")

        # Clean up mask using skimage functions
        mask = remove_small_objects(mask, min_size=20)
        mask = binary_closing(mask)
        mask = binary_opening(mask)
        
        # Keep largest component
        mask = self._largest_component(mask)
        return mask.astype(bool), (x, y, w, h)

    def _cyto_ring_mask(self, nucleus_mask: np.ndarray, ring_width: int = 5,
                        inner_gap: int = 0) -> np.ndarray:
        """Annulus outside nucleus: dilate then subtract (optional small gap to avoid bleed)."""
        if inner_gap > 0:
            inner = cv2.dilate(nucleus_mask.astype(np.uint8),
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*inner_gap+1, 2*inner_gap+1)),
                               iterations=1).astype(bool)
        else:
            inner = nucleus_mask.astype(bool)

        outer = cv2.dilate(nucleus_mask.astype(np.uint8),
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*ring_width+1, 2*ring_width+1)),
                           iterations=1).astype(bool)
        ring = outer & (~inner)
        ring = self._remove_small_objects(ring, min_size=10)
        return ring

    def _whole_cell_mask_from_ch002(self, ch002_crop: np.ndarray, nucleus_mask: np.ndarray) -> np.ndarray:
        """
        Watershed segmentation on ch002, seeded by the (one) nucleus in the crop.
        Returns the basin corresponding to the nucleus (whole cell mask).
        """
        # 1) Threshold ch002 to get likely foreground (cell)
        blur = cv2.GaussianBlur(ch002_crop, (5,5), 0)
        thr_val, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        fg_prob = thr > 0

        # clean
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(fg_prob.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=1)
        opening = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2).astype(bool)
        opening = self._remove_small_objects(opening, min_size=30)

        # 2) seeds: use nucleus as sure foreground
        sure_fg = cv2.dilate(nucleus_mask.astype(np.uint8), kernel, iterations=1)
        sure_fg = (sure_fg > 0)

        # 3) background
        sure_bg = cv2.dilate((~opening).astype(np.uint8), kernel, iterations=2) > 0

        # 4) unknown region
        unknown = (opening & (~sure_fg))

        # 5) markers
        _cc, markers = cv2.connectedComponents(sure_fg.astype(np.uint8))
        if _cc < 2:
            # ensure at least 1 marker label for nucleus
            markers = (sure_fg.astype(np.uint8))*1
            _cc = 2
        markers = markers + 1
        markers[unknown] = 0

        # 6) watershed on gradient
        grad = cv2.Laplacian(blur, cv2.CV_32F)
        grad = cv2.convertScaleAbs(grad)
        m = cv2.watershed(cv2.cvtColor(grad, cv2.COLOR_GRAY2BGR), markers.copy())

        # pick the region that contains the nucleus label (original label '2')
        nucleus_label = 2  # because connectedComponents start at 1, then +1 above
        whole_cell = (m == nucleus_label)

        # fallback: if watershed failed, at least return foreground around nucleus
        if not np.any(whole_cell):
            whole_cell = opening.copy()
            # keep largest component that overlaps nucleus
            lbl, n = ndimage.label(whole_cell.astype(np.uint8))
            if n > 0:
                overlaps = [np.sum(nucleus_mask & (lbl == i+1)) for i in range(n)]
                k = int(np.argmax(overlaps)) + 1
                whole_cell = (lbl == k)

        # ensure nucleus ⊂ whole_cell
        whole_cell = (whole_cell | nucleus_mask).astype(bool)
        return whole_cell

    # -------------------- main measurement --------------------

    def measure_cdk2_activity(self, ch00_img: np.ndarray, ch002_img: np.ndarray,
                              selection: Dict, segmentation_method: str = 'otsu',
                              cytoplasm_mode: str = 'whole_cell',  # 'ring' or 'whole_cell'
                              ring_width: int = 5, inner_gap: int = 1,
                              background_threshold: int = 10) -> Dict:
        """
        Pipeline:
          1) segment nucleus on ch00 inside selection
          2) crop ch002 to the same box and reuse aligned nucleus mask
          3) measure nuclear intensity on ch002
          4) cytoplasm:
             - ring: annulus outside nucleus (Cappell style)
             - whole_cell: watershed cell mask on ch002 (seeded by nucleus) minus nucleus
          5) compute CDK2 ratios
        """
        # 1) nucleus on ch00
        nuc_mask_ch00, (x, y, w, h) = self.segment_nucleus_from_ch00(ch00_img, selection, method=segmentation_method)

        # 2) crop ch002 & align mask 1:1 (images assumed co-registered)
        ch002_crop = ch002_img[y:y+h, x:x+w]
        nucleus_mask = nuc_mask_ch00  # aligned by design

        # 3) nuclear intensity (measured on ch002)
        nuclear_stats = self.measure_whiteness_regionprops(ch002_crop, nucleus_mask, background_threshold)

        # 4) cytoplasm mask
        if cytoplasm_mode == 'ring':
            cyto_mask = self._cyto_ring_mask(nucleus_mask, ring_width=ring_width, inner_gap=inner_gap)
        elif cytoplasm_mode == 'whole_cell':
            whole_cell = self._whole_cell_mask_from_ch002(ch002_crop, nucleus_mask)
            cyto_mask = (whole_cell & (~nucleus_mask))
        else:
            raise ValueError("cytoplasm_mode must be 'ring' or 'whole_cell'")

        # 5) cytoplasm intensity (on ch002), excluding background
        cytoplasm_stats = self.measure_whiteness_regionprops(ch002_crop, cyto_mask, background_threshold)

        # Robustness: if cytoplasm mask empty (tiny cell, dim, etc.), fall back to ring
        if cytoplasm_stats['foreground_pixel_count'] == 0 and cytoplasm_mode == 'whole_cell':
            backup_mask = self._cyto_ring_mask(nucleus_mask, ring_width=max(3, ring_width), inner_gap=inner_gap)
            cytoplasm_stats = self.measure_whiteness_regionprops(ch002_crop, backup_mask, background_threshold)
            cyto_mask = backup_mask
            cytoplasm_mode = 'ring_fallback'

        # 6) ratios (C/N & N/C)
        n_mean = nuclear_stats['mean_intensity'] if nuclear_stats['foreground_pixel_count'] > 0 else 0.0
        c_mean = cytoplasm_stats['mean_intensity'] if cytoplasm_stats['foreground_pixel_count'] > 0 else 0.0
        cn_ratio = (c_mean / n_mean) if n_mean > 0 else 0.0
        nc_ratio = (n_mean / c_mean) if c_mean > 0 else 0.0
        activity_index = ((n_mean - c_mean) / (n_mean + c_mean)) if (n_mean + c_mean) > 0 else 0.0

        out = {
            'nuclear_whiteness_ch002': nuclear_stats,
            'cytoplasmic_whiteness_ch002': cytoplasm_stats,
            'cdk2_ratio_cyto_over_nuc': float(cn_ratio),
            'cdk2_ratio_nuc_over_cyto': float(nc_ratio),
            'cdk2_activity_index': float(activity_index),
            'nucleus_pixel_count': int(np.sum(nucleus_mask)),
            'cytoplasm_pixel_count': int(np.sum(cyto_mask)),
            'selection_xywh': {'x': x, 'y': y, 'w': w, 'h': h},
            'segmentation_method': segmentation_method,
            'cytoplasm_mode': cytoplasm_mode,
            'ring_width': ring_width,
            'inner_gap': inner_gap,
            'background_threshold': background_threshold
        }
        self.measurement_history.append(out)
        return out

    # -------------------- tracking & plotting (unchanged except key names) --------------------

    def analyze_cell_cycle_phase_whiteness(self, cdk2_ratio: float, nuclear_whiteness: float) -> str:
        if cdk2_ratio < 0.8:
            return "G1"
        elif cdk2_ratio < 1.2:
            return "G1/S"
        elif cdk2_ratio < 1.8:
            return "S"
        elif cdk2_ratio < 2.5:
            return "G2"
        else:
            return "M"

    def track_cell_whiteness_across_frames(self, cell_id: str, frame_data: Dict) -> None:
        if cell_id not in self.cell_tracks:
            self.cell_tracks[cell_id] = {'birth_frame': len(self.measurement_history),
                                         'mother_id': None, 'daughter_ids': [], 'measurements': []}
        self.cell_tracks[cell_id]['measurements'].append(frame_data)
        frame_data['cell_cycle_phase'] = self.analyze_cell_cycle_phase_whiteness(
            frame_data['cdk2_ratio_cyto_over_nuc'],
            frame_data['nuclear_whiteness_ch002']['mean_intensity']
        )

    def analyze_cdk2_whiteness_dynamics(self, cell_id: str) -> Dict:
        if cell_id not in self.cell_tracks:
            return {}
        m = self.cell_tracks[cell_id]['measurements']
        if len(m) < 2:
            return {'error': 'Insufficient data for dynamics analysis'}
        cn = [x['cdk2_ratio_cyto_over_nuc'] for x in m]
        act = [x['cdk2_activity_index'] for x in m]
        n = [x['nuclear_whiteness_ch002']['mean_intensity'] for x in m]
        c = [x['cytoplasmic_whiteness_ch002']['mean_intensity'] for x in m]
        peak = float(np.max(cn)); pidx = int(np.argmax(cn))
        valley = float(np.min(cn)); vidx = int(np.argmin(cn))
        inc = (peak - cn[0]) / (pidx+1) if pidx > 0 else 0.0
        dec = (peak - cn[-1]) / (len(cn)-pidx) if pidx < len(cn)-1 else 0.0
        return {
            'cell_id': cell_id,
            'total_frames': len(m),
            'peak_cdk2_ratio': peak, 'peak_time': pidx,
            'valley_cdk2_ratio': valley, 'valley_time': vidx,
            'cdk2_increase_rate': float(inc), 'cdk2_decrease_rate': float(dec),
            'mean_cdk2_ratio': float(np.mean(cn)), 'std_cdk2_ratio': float(np.std(cn)),
            'mean_nuclear_whiteness': float(np.mean(n)), 'mean_cytoplasmic_whiteness': float(np.mean(c))
        }

    def export_whiteness_measurements(self, filename: str) -> None:
        export = {
            'measurement_history': self.measurement_history,
            'cell_tracks': self.cell_tracks,
            'summary': {
                'total_measurements': len(self.measurement_history),
                'total_cells': len(self.cell_tracks),
                'analysis_timestamp': str(np.datetime64('now')),
                'methodology': 'Nucleus from ch00; nuclear & cytoplasmic intensities measured on ch002; cytoplasm via ring or watershed.',
                'reference': 'Cappell et al., Cell (2016) and follow-ups'
            }
        }
        with open(filename, 'w') as f:
            json.dump(export, f, indent=2)

    def plot_cdk2_whiteness_trajectory(self, cell_id: str, save_path: Optional[str] = None) -> None:
        if cell_id not in self.cell_tracks or len(self.cell_tracks[cell_id]['measurements']) < 2:
            print("Insufficient data")
            return
        ms = self.cell_tracks[cell_id]['measurements']
        frames = np.arange(len(ms))
        cn = [m['cdk2_ratio_cyto_over_nuc'] for m in ms]
        act = [m['cdk2_activity_index'] for m in ms]
        n = [m['nuclear_whiteness_ch002']['mean_intensity'] for m in ms]
        c = [m['cytoplasmic_whiteness_ch002']['mean_intensity'] for m in ms]

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        axs[0,0].plot(frames, cn, 'o-'); axs[0,0].set_ylabel('C/N ratio'); axs[0,0].grid(True, alpha=0.3)
        axs[0,1].plot(frames, act, 'o-'); axs[0,1].set_ylabel('Activity index'); axs[0,1].grid(True, alpha=0.3)
        axs[1,0].plot(frames, n, 'o-', label='Nuclear'); axs[1,0].plot(frames, c, 'o-', label='Cytoplasm')
        axs[1,0].legend(); axs[1,0].set_ylabel('Mean intensity'); axs[1,0].set_xlabel('Frame'); axs[1,0].grid(True, alpha=0.3)
        axs[1,1].plot(frames, [100*mm['nucleus_pixel_count']/max(1,(mm['nucleus_pixel_count']+mm['cytoplasm_pixel_count'])) for mm in ms], 'o-')
        axs[1,1].set_ylabel('Nuclear area (%)'); axs[1,1].set_xlabel('Frame'); axs[1,1].grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else: plt.show()
        plt.close()
