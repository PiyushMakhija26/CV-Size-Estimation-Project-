"""
Vision-Fit Processing Module
Handles all computer vision and measurement logic separately from UI.
"""

import cv2
import numpy as np
import math
import os
import urllib.request
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp


class VisionFitProcessor:
    """
    Encapsulates all Vision-Fit measurement and processing logic.
    Can be used by any UI (Tkinter, Streamlit, etc.)
    """
    
    def __init__(self):
        self.shoulder_buffer = deque(maxlen=20)
        self.model_path = self._get_model_path()
        
        # Constants
        self.SHOULDER_OFFSET = 2.0  # cm
        self.DISTANCE_THRESHOLD = 80  # % of frame height
        
        # Size thresholds (realistic values for shoulder width including +2cm offset)
        self.SIZE_THRESHOLDS = {
            'S': (0, 40),
            'M': (40, 44),
            'L': (44, 48),
            'XL': (48, 52),
            'XXL': (52, 300)
        }
    
    def _get_model_path(self):
        """Download or retrieve cached pose model."""
        model_path = 'pose_landmarker_lite.task'
        if not os.path.exists(model_path):
            print("Downloading pose model...")
            url = (
                'https://storage.googleapis.com/mediapipe-models/'
                'pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'
            )
            try:
                urllib.request.urlretrieve(url, model_path)
                print("Model downloaded successfully.")
            except Exception as e:
                print(f"Error downloading model: {e}")
        return model_path
    
    def validate_user_distance(self, image, landmarks):
        """Check if user is too close to camera."""
        nose = landmarks[0]
        left_heel = landmarks[29]
        right_heel = landmarks[30]
        
        heel_y = min(left_heel.y, right_heel.y)
        frame_height = image.shape[0]
        pixel_distance = abs(nose.y - heel_y) * frame_height
        distance_percentage = (pixel_distance / frame_height) * 100
        
        is_valid = distance_percentage <= self.DISTANCE_THRESHOLD
        warning = (
            "⚠️ Too close to camera! Move back 1-2 steps."
            if not is_valid else None
        )
        
        return is_valid, distance_percentage, warning
    
    def detect_a4_paper(self, image):
        """Detect A4 paper for calibration."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)
        
        contours, _ = cv2.findContours(
            edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            if len(approx) == 4:
                rect = cv2.minAreaRect(contour)
                width, height = rect[1]
                
                if width > height:
                    aspect = height / width
                else:
                    aspect = width / height
                
                if 0.6 < aspect < 0.8:
                    return min(width, height)
        
        return None
    
    def get_pixels_per_cm(self, image, user_height_cm=None):
        """Get calibration factor (pixels per cm)."""
        # Try paper calibration first
        paper_width_pixels = self.detect_a4_paper(image)
        if paper_width_pixels:
            pixels_per_cm = paper_width_pixels / 21.0
            return pixels_per_cm, "paper"
        
        # Fall back to height calibration
        if user_height_cm:
            options = vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(
                    model_asset_path=self.model_path
                ),
                running_mode=vision.RunningMode.IMAGE
            )
            with vision.PoseLandmarker.create_from_options(options) as landmarker:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=rgb_image
                )
                result = landmarker.detect(mp_image)
                
                if result.pose_landmarks:
                    landmarks = result.pose_landmarks[0]
                    nose = landmarks[0]
                    left_heel = landmarks[29]
                    right_heel = landmarks[30]
                    
                    heel_y = min(left_heel.y, right_heel.y)
                    frame_height = image.shape[0]
                    pixel_distance = abs(nose.y - heel_y) * frame_height
                    pixels_per_cm = pixel_distance / user_height_cm
                    
                    return pixels_per_cm, "height"
        
        return None, None
    
    def calculate_bmi(self, weight_kg, height_cm):
        """Calculate BMI."""
        if height_cm:
            height_m = height_cm / 100.0
            bmi = weight_kg / (height_m ** 2)
            return bmi
        return None
    
    def calculate_shoulder_width(self, image, pixels_per_cm):
        """Calculate shoulder width with distance validation."""
        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=self.model_path
            ),
            running_mode=vision.RunningMode.IMAGE
        )
        
        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            result = landmarker.detect(mp_image)
            
            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                
                # Validate distance
                is_valid, dist_pct, warning = self.validate_user_distance(
                    image, landmarks
                )
                
                # Get shoulder landmarks
                left_shoulder = landmarks[11]
                right_shoulder = landmarks[12]
                
                h, w, _ = image.shape
                left_x = int(left_shoulder.x * w)
                right_x = int(right_shoulder.x * w)
                
                pixel_distance = abs(left_x - right_x)
                cm_distance = pixel_distance / pixels_per_cm
                cm_distance_adjusted = cm_distance + self.SHOULDER_OFFSET
                
                return cm_distance_adjusted, result, is_valid, warning
        
        return None, None, None, None
    
    def apply_temporal_smoothing(self, shoulder_width_cm):
        """Apply temporal smoothing to reduce jitter."""
        self.shoulder_buffer.append(shoulder_width_cm)
        smoothed_width = np.mean(list(self.shoulder_buffer))
        return smoothed_width
    
    def recommend_size(self, shoulder_width_cm, bmi=None):
        """Recommend T-shirt size."""
        base_size = "XXL"
        for size, (min_val, max_val) in self.SIZE_THRESHOLDS.items():
            if min_val <= shoulder_width_cm < max_val:
                base_size = size
                break
        
        fit_note = ""
        if bmi is not None:
            if bmi > 26:
                if base_size == "S":
                    fit_note = " (Consider M for relaxed fit)"
                elif base_size == "M":
                    fit_note = " (Consider L for relaxed fit)"
                elif base_size == "L":
                    fit_note = " (Consider XL for relaxed fit)"
                elif base_size == "XL":
                    fit_note = " (Consider XXL for relaxed fit)"
                else:
                    fit_note = " (Oversized fit recommended)"
            elif bmi < 18.5:
                fit_note = " (Slim fit recommended)"
        
        return base_size, fit_note
    
    def process_image(self, image_path, height_cm=None, weight_kg=None):
        """
        Main processing pipeline.
        
        Args:
            image_path: Path to input image
            height_cm: User's height in cm (optional)
            weight_kg: User's weight in kg (optional)
        
        Returns:
            dict with results or error message
        """
        try:
            # Read image
            if not os.path.exists(image_path):
                return {'error': 'Image file not found'}
            
            image = cv2.imread(image_path)
            if image is None:
                return {'error': 'Could not read image. Ensure it is a valid image file.'}
            
            # Get calibration
            if height_cm is None:
                return {'error': 'Height is required for calibration.'}
            
            pixels_per_cm, calib_method = self.get_pixels_per_cm(image, height_cm)
            if pixels_per_cm is None:
                return {'error': 'Could not calibrate. Ensure full body is visible.'}
            
            # Calculate shoulder width
            shoulder_cm, pose_results, is_valid_dist, distance_warning = (
                self.calculate_shoulder_width(image, pixels_per_cm)
            )
            
            if shoulder_cm is None:
                return {'error': 'No person detected. Please try again with a clear full-body photo.'}
            
            if not is_valid_dist:
                return {'error': distance_warning or 'User too close to camera'}
            
            # Calculate BMI if weight provided
            bmi = None
            if weight_kg:
                bmi = self.calculate_bmi(weight_kg, height_cm)
            
            # Apply smoothing
            smoothed_shoulder = self.apply_temporal_smoothing(shoulder_cm)
            
            # Recommend size
            size, fit_note = self.recommend_size(smoothed_shoulder, bmi)
            
            # Prepare results
            results = {
                'success': True,
                'shoulder_width_raw': shoulder_cm,
                'shoulder_width_smoothed': smoothed_shoulder,
                'bmi': bmi,
                'recommended_size': size,
                'fit_note': fit_note,
                'calibration_method': calib_method,
                'pose_results': pose_results,
                'image': image
            }
            
            return results
        
        except Exception as e:
            return {'error': f'Processing error: {str(e)}'}
    
    def annotate_image(self, image, results):
        """
        Annotate image with measurement results.
        
        Args:
            image: Input image
            results: Processing results dict
        
        Returns:
            Annotated image
        """
        if 'error' in results:
            return image
        
        annotated = image.copy()
        h, w, _ = annotated.shape
        
        size = results['recommended_size']
        shoulder = results['shoulder_width_smoothed']
        bmi = results['bmi']
        fit_note = results['fit_note']
        
        # Draw text
        cv2.putText(
            annotated, f"SIZE: {size}{fit_note}",
            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3
        )
        cv2.putText(
            annotated, f"Shoulder: {shoulder:.1f} cm",
            (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2
        )
        if bmi:
            cv2.putText(
                annotated, f"BMI: {bmi:.1f}",
                (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2
            )
        
        return annotated
