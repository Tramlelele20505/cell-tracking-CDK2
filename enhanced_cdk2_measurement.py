#!/usr/bin/env python3
"""
Enhanced CDK2 Measurement Module
Implements Cappell et al. (2016) methodology for CDK2 fluorescence analysis
"""

import cv2
import numpy as np
from scipy import ndimage
from scipy.stats import percentileofscore
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional
import json

class CDK2Analyzer:
    """
    Enhanced CDK2 measurement analyzer following Cappell et al. methodology
    """
    
    def __init__(self):
        self.measurement_history = []
        self.cell_tracks = {}
        
    def segment_nucleus(self, image: np.ndarray, method: str = 'otsu') -> np.ndarray:
        """
        Automated nuclear segmentation using multiple methods
        
        Args:
            image: Grayscale image array
            method: Segmentation method ('otsu', 'watershed', 'adaptive')
            
        Returns:
            Boolean mask of nuclear regions
        """
        if method == 'otsu':
            # Otsu thresholding for bimodal histograms
            threshold = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
            mask = image > threshold
            
        elif method == 'adaptive':
            # Adaptive thresholding for varying illumination
            mask = cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            mask = mask > 0
            
        elif method == 'watershed':
            # Watershed segmentation for complex boundaries
            # Preprocessing
            blur = cv2.GaussianBlur(image, (5, 5), 0)
            ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Noise removal
            kernel = np.ones((3, 3), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
            
            # Sure background area
            sure_bg = cv2.dilate(opening, kernel, iterations=3)
            
            # Finding sure foreground area
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
            sure_fg = np.uint8(sure_fg)
            
            # Finding unknown region
            unknown = cv2.subtract(sure_bg, sure_fg)
            
            # Marker labelling
            ret, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            markers[unknown == 255] = 0
            
            # Watershed
            markers = cv2.watershed(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), markers)
            mask = markers > 1
            
        else:
            raise ValueError(f"Unknown segmentation method: {method}")
        
        # Morphological operations to clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Remove small objects
        mask = self._remove_small_objects(mask, min_size=50)
        
        return mask.astype(bool)
    
    def _remove_small_objects(self, mask: np.ndarray, min_size: int = 50) -> np.ndarray:
        """Remove small objects from binary mask"""
        labeled, num_features = ndimage.label(mask)
        sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
        mask = sizes >= min_size
        remove_pixel = mask[labeled - 1]
        mask[remove_pixel] = 0
        return mask
    
    def create_cytoplasmic_ring(self, nucleus_mask: np.ndarray, ring_width: int = 5) -> np.ndarray:
        """
        Create cytoplasmic ring around nucleus
        
        Args:
            nucleus_mask: Boolean mask of nuclear regions
            ring_width: Width of cytoplasmic ring in pixels
            
        Returns:
            Boolean mask of cytoplasmic ring
        """
        # Dilate nucleus to create ring
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_width*2+1, ring_width*2+1))
        dilated = cv2.dilate(nucleus_mask.astype(np.uint8), kernel, iterations=1)
        
        # Create ring by subtracting nucleus from dilated region
        ring_mask = dilated.astype(bool) & ~nucleus_mask
        
        return ring_mask
    
    def measure_cdk2_parameters(self, nucleus_img: np.ndarray, cytoplasm_img: np.ndarray, 
                              nucleus_mask: np.ndarray, cytoplasm_mask: np.ndarray) -> Dict:
        """
        Enhanced CDK2 measurement following Cappell et al. methodology
        
        Args:
            nucleus_img: Nuclear channel image
            cytoplasm_img: Cytoplasmic channel image  
            nucleus_mask: Nuclear segmentation mask
            cytoplasm_mask: Cytoplasmic segmentation mask
            
        Returns:
            Dictionary containing all CDK2 measurement parameters
        """
        # Nuclear measurements
        nuclear_pixels = nucleus_img[nucleus_mask]
        nuclear_area = np.sum(nucleus_mask)
        nuclear_mass = np.sum(nuclear_pixels)
        nuclear_median = np.median(nuclear_pixels)
        nuclear_mean = np.mean(nuclear_pixels)
        nuclear_std = np.std(nuclear_pixels)
        
        # Cytoplasmic ring measurements
        ring_mask = cytoplasm_mask & ~nucleus_mask  # Cytoplasm excluding nucleus
        ring_pixels = cytoplasm_img[ring_mask]
        
        if len(ring_pixels) > 0:
            ring_median = np.median(ring_pixels)
            ring_75th = np.percentile(ring_pixels, 75)
            ring_mean = np.mean(ring_pixels)
            ring_std = np.std(ring_pixels)
        else:
            ring_median = ring_75th = ring_mean = ring_std = 0
        
        # CDK2 calculations
        cdk2_ratio = nuclear_median / ring_median if ring_median > 0 else 0
        cdk2_normalized = nuclear_median / nuclear_area if nuclear_area > 0 else 0
        cdk2_activity = (nuclear_median - ring_median) / (nuclear_median + ring_median) if (nuclear_median + ring_median) > 0 else 0
        
        # Additional metrics
        nuclear_cytoplasmic_ratio = nuclear_mean / ring_mean if ring_mean > 0 else 0
        signal_to_noise = nuclear_median / nuclear_std if nuclear_std > 0 else 0
        
        return {
            # Basic measurements
            'nuclear_median': float(nuclear_median),
            'nuclear_mean': float(nuclear_mean),
            'nuclear_mass': float(nuclear_mass),
            'nuclear_area': int(nuclear_area),
            'nuclear_std': float(nuclear_std),
            
            # Cytoplasmic measurements
            'ring_median': float(ring_median),
            'ring_75th': float(ring_75th),
            'ring_mean': float(ring_mean),
            'ring_std': float(ring_std),
            
            # CDK2 ratios and indices
            'cdk2_ratio': float(cdk2_ratio),
            'cdk2_normalized': float(cdk2_normalized),
            'cdk2_activity': float(cdk2_activity),
            'nuclear_cytoplasmic_ratio': float(nuclear_cytoplasmic_ratio),
            'signal_to_noise': float(signal_to_noise),
            
            # Quality metrics
            'nuclear_pixel_count': len(nuclear_pixels),
            'ring_pixel_count': len(ring_pixels)
        }
    
    def analyze_cell_cycle_phase(self, cdk2_ratio: float, nuclear_area: int) -> str:
        """
        Determine cell cycle phase based on CDK2 levels and nuclear size
        
        Args:
            cdk2_ratio: Nuclear-to-cytoplasmic CDK2 ratio
            nuclear_area: Nuclear area in pixels
            
        Returns:
            Predicted cell cycle phase
        """
        # These thresholds would need to be calibrated for your specific system
        if cdk2_ratio < 0.5:
            return "G1"
        elif cdk2_ratio < 1.0:
            return "G1/S"
        elif cdk2_ratio < 2.0:
            return "S"
        elif cdk2_ratio < 3.0:
            return "G2"
        else:
            return "M"
    
    def track_cell_across_frames(self, cell_id: str, frame_data: Dict) -> None:
        """
        Track individual cell across timepoints
        
        Args:
            cell_id: Unique cell identifier
            frame_data: CDK2 measurement data for current frame
        """
        if cell_id not in self.cell_tracks:
            self.cell_tracks[cell_id] = {
                'birth_frame': len(self.measurement_history),
                'mother_id': None,
                'daughter_ids': [],
                'measurements': []
            }
        
        # Add current measurement
        self.cell_tracks[cell_id]['measurements'].append(frame_data)
        
        # Determine cell cycle phase
        frame_data['cell_cycle_phase'] = self.analyze_cell_cycle_phase(
            frame_data['cdk2_ratio'], frame_data['nuclear_area']
        )
    
    def analyze_cdk2_dynamics(self, cell_id: str) -> Dict:
        """
        Analyze CDK2 dynamics for a specific cell across timepoints
        
        Args:
            cell_id: Cell identifier
            
        Returns:
            Dictionary containing CDK2 dynamics analysis
        """
        if cell_id not in self.cell_tracks:
            return {}
        
        track = self.cell_tracks[cell_id]
        measurements = track['measurements']
        
        if len(measurements) < 2:
            return {'error': 'Insufficient data for dynamics analysis'}
        
        # Extract time series
        cdk2_ratios = [m['cdk2_ratio'] for m in measurements]
        cdk2_activity = [m['cdk2_activity'] for m in measurements]
        nuclear_areas = [m['nuclear_area'] for m in measurements]
        
        # Calculate dynamics metrics
        peak_cdk2 = max(cdk2_ratios)
        peak_time = np.argmax(cdk2_ratios)
        valley_cdk2 = min(cdk2_ratios)
        valley_time = np.argmin(cdk2_ratios)
        
        # Calculate rates of change
        if len(cdk2_ratios) > 1:
            cdk2_increase_rate = (peak_cdk2 - cdk2_ratios[0]) / (peak_time + 1) if peak_time > 0 else 0
            cdk2_decrease_rate = (peak_cdk2 - cdk2_ratios[-1]) / (len(cdk2_ratios) - peak_time) if peak_time < len(cdk2_ratios) - 1 else 0
        else:
            cdk2_increase_rate = cdk2_decrease_rate = 0
        
        # Detect oscillations (simplified)
        oscillations = self._detect_oscillations(cdk2_ratios)
        
        return {
            'cell_id': cell_id,
            'total_frames': len(measurements),
            'peak_cdk2': float(peak_cdk2),
            'peak_time': int(peak_time),
            'valley_cdk2': float(valley_cdk2),
            'valley_time': int(valley_time),
            'cdk2_increase_rate': float(cdk2_increase_rate),
            'cdk2_decrease_rate': float(cdk2_decrease_rate),
            'oscillations_detected': oscillations,
            'mean_cdk2_ratio': float(np.mean(cdk2_ratios)),
            'std_cdk2_ratio': float(np.std(cdk2_ratios)),
            'mean_nuclear_area': float(np.mean(nuclear_areas))
        }
    
    def _detect_oscillations(self, time_series: List[float], threshold: float = 0.1) -> bool:
        """
        Simple oscillation detection based on direction changes
        
        Args:
            time_series: List of CDK2 values over time
            threshold: Minimum change threshold for oscillation detection
            
        Returns:
            True if oscillations detected
        """
        if len(time_series) < 3:
            return False
        
        direction_changes = 0
        for i in range(1, len(time_series) - 1):
            diff1 = time_series[i] - time_series[i-1]
            diff2 = time_series[i+1] - time_series[i]
            
            if abs(diff1) > threshold and abs(diff2) > threshold:
                if (diff1 > 0 and diff2 < 0) or (diff1 < 0 and diff2 > 0):
                    direction_changes += 1
        
        return direction_changes >= 2
    
    def export_measurements(self, filename: str) -> None:
        """
        Export all measurements to JSON file
        
        Args:
            filename: Output filename
        """
        export_data = {
            'measurement_history': self.measurement_history,
            'cell_tracks': self.cell_tracks,
            'summary': {
                'total_measurements': len(self.measurement_history),
                'total_cells': len(self.cell_tracks),
                'analysis_timestamp': str(np.datetime64('now'))
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def plot_cdk2_trajectory(self, cell_id: str, save_path: Optional[str] = None) -> None:
        """
        Plot CDK2 trajectory for a specific cell
        
        Args:
            cell_id: Cell identifier
            save_path: Optional path to save plot
        """
        if cell_id not in self.cell_tracks:
            print(f"Cell {cell_id} not found in tracks")
            return
        
        track = self.cell_tracks[cell_id]
        measurements = track['measurements']
        
        if len(measurements) < 2:
            print(f"Insufficient data for cell {cell_id}")
            return
        
        # Extract data
        frames = range(len(measurements))
        cdk2_ratios = [m['cdk2_ratio'] for m in measurements]
        cdk2_activity = [m['cdk2_activity'] for m in measurements]
        nuclear_areas = [m['nuclear_area'] for m in measurements]
        
        # Create plot
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
        
        # CDK2 ratio
        ax1.plot(frames, cdk2_ratios, 'b-o', linewidth=2, markersize=4)
        ax1.set_ylabel('CDK2 Ratio (Nuclear/Cytoplasmic)')
        ax1.set_title(f'CDK2 Dynamics - Cell {cell_id}')
        ax1.grid(True, alpha=0.3)
        
        # CDK2 activity
        ax2.plot(frames, cdk2_activity, 'r-o', linewidth=2, markersize=4)
        ax2.set_ylabel('CDK2 Activity Index')
        ax2.grid(True, alpha=0.3)
        
        # Nuclear area
        ax3.plot(frames, nuclear_areas, 'g-o', linewidth=2, markersize=4)
        ax3.set_xlabel('Frame')
        ax3.set_ylabel('Nuclear Area (pixels)')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()

# Example usage and testing
if __name__ == "__main__":
    # Create analyzer instance
    analyzer = CDK2Analyzer()
    
    # Example with synthetic data
    print("CDK2 Analyzer initialized successfully!")
    print("Use this module to enhance your cell tracking application with:")
    print("- Automated nuclear segmentation")
    print("- Cytoplasmic ring analysis") 
    print("- CDK2 ratio calculations")
    print("- Cell cycle phase analysis")
    print("- Time-series tracking")

