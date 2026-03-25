"""
Vision-Fit Processing Module
Handles all computer vision and measurement logic separately from UI.
"""

import cv2
import numpy as np
import os
from collections import deque
import mediapipe as mp


class VisionFitProcessor:
    """Core body measurement logic."""

    def __init__(self):
        self.shoulder_buffer = deque(maxlen=20)
        self.SHOULDER_OFFSET = 2.0
        self.DISTANCE_THRESHOLD = 80
        self.SIZE_THRESHOLDS = {
            'S': (0, 40),
            'M': (40, 44),
            'L': (44, 48),
            'XL': (48, 52),
            'XXL': (52, 300)
        }

    def _get_pose_landmarks(self, image):
        with mp.solutions.pose.Pose(
            static_image_mode=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose:
            result = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if result.pose_landmarks:
                return result.pose_landmarks.landmark
        return None

    def validate_user_distance(self, image, landmarks):
        nose = landmarks[0]
        left_heel = landmarks[29]
        right_heel = landmarks[30]
        heel_y = min(left_heel.y, right_heel.y)
        frame_height = image.shape[0]
        pixel_distance = abs(nose.y - heel_y) * frame_height
        distance_percentage = (pixel_distance / frame_height) * 100

        if distance_percentage > self.DISTANCE_THRESHOLD:
            return False, distance_percentage, "⚠️ Too close to camera! Move back 1-2 steps."
        return True, distance_percentage, None

    def detect_a4_paper(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                rect = cv2.minAreaRect(contour)
                width, height = rect[1]
                aspect = min(width, height) / max(width, height)
                if 0.6 < aspect < 0.8:
                    return min(width, height)
        return None

    def get_pixels_per_cm(self, image, user_height_cm=None):
        paper_width_pixels = self.detect_a4_paper(image)
        if paper_width_pixels:
            return paper_width_pixels / 21.0, 'paper'

        if user_height_cm is None:
            return None, None

        landmarks = self._get_pose_landmarks(image)
        if not landmarks:
            return None, None

        nose = landmarks[0]
        left_heel = landmarks[29]
        right_heel = landmarks[30]
        heel_y = min(left_heel.y, right_heel.y)
        frame_height = image.shape[0]
        pixel_distance = abs(nose.y - heel_y) * frame_height

        if pixel_distance <= 0:
            return None, None

        return pixel_distance / user_height_cm, 'height'

    def calculate_shoulder_width(self, image, pixels_per_cm):
        landmarks = self._get_pose_landmarks(image)
        if not landmarks:
            return None, None, None, None

        is_valid, distance_pct, warning = self.validate_user_distance(image, landmarks)
        if not is_valid:
            return None, None, is_valid, warning

        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]

        h, w, _ = image.shape
        left_x = int(left_shoulder.x * w)
        right_x = int(right_shoulder.x * w)

        pixel_distance = abs(left_x - right_x)
        cm_distance = pixel_distance / pixels_per_cm
        cm_distance_adjusted = cm_distance + self.SHOULDER_OFFSET

        return cm_distance_adjusted, landmarks, is_valid, warning

    def apply_temporal_smoothing(self, shoulder_width_cm):
        self.shoulder_buffer.append(shoulder_width_cm)
        return float(np.mean(list(self.shoulder_buffer)))

    def calculate_bmi(self, weight_kg, height_cm):
        if height_cm > 0:
            height_m = height_cm / 100.0
            return weight_kg / (height_m ** 2)
        return None

    def recommend_size(self, shoulder_width_cm, bmi=None):
        base_size = 'XXL'
        for size, (low, high) in self.SIZE_THRESHOLDS.items():
            if low <= shoulder_width_cm < high:
                base_size = size
                break

        fit_note = ''
        if bmi is not None:
            if bmi > 26:
                map_oversize = {'S': 'M', 'M': 'L', 'L': 'XL', 'XL': 'XXL'}
                fit_note = f" (Consider {map_oversize.get(base_size, 'XXL')} for relaxed fit)"
            elif bmi < 18.5:
                fit_note = ' (Slim fit recommended)'

        return base_size, fit_note

    def process_image(self, image_path, height_cm=None, weight_kg=None):
        if not os.path.exists(image_path):
            return {'error': 'Image file not found'}

        image = cv2.imread(image_path)
        if image is None:
            return {'error': 'Could not read image; ensure it is valid'}

        if height_cm is None:
            return {'error': 'Height is required for calibration'}

        pixels_per_cm, method = self.get_pixels_per_cm(image, height_cm)
        if not pixels_per_cm:
            return {'error': 'Calibration failed. Ensure full body or A4 paper is visible.'}

        shoulder_cm, _, is_valid, warning = self.calculate_shoulder_width(image, pixels_per_cm)
        if shoulder_cm is None:
            return {'error': warning or 'Could not detect shoulders'}

        smoothed_width = self.apply_temporal_smoothing(shoulder_cm)
        bmi = self.calculate_bmi(weight_kg, height_cm) if weight_kg else None
        size, fit_note = self.recommend_size(smoothed_width, bmi)

        return {
            'shoulder_width': shoulder_cm,
            'shoulder_width_smoothed': smoothed_width,
            'bmi': bmi,
            'recommended_size': size,
            'fit_note': fit_note,
            'calibration_method': method,
            'distance_pct': self.validate_user_distance(image, self._get_pose_landmarks(image))[1]
        }
