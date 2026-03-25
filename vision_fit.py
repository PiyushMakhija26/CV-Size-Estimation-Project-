import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import math
import os
import urllib.request
from collections import deque

def get_model_path():
    model_path = 'pose_landmarker_lite.task'
    if not os.path.exists(model_path):
        print("Downloading pose model...")
        url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'
        urllib.request.urlretrieve(url, model_path)
        print("Model downloaded.")
    return model_path

def validate_user_distance(image, landmarks):
    """
    Check if user is too close to the camera using nose-to-heel distance.
    Returns (is_valid, distance_percentage, warning_message)
    """
    nose = landmarks[0]
    left_heel = landmarks[29]
    right_heel = landmarks[30]
    
    # Use the lower heel
    heel_y = min(left_heel.y, right_heel.y)
    
    # Calculate pixel distance from nose to heel
    frame_height = image.shape[0]
    pixel_distance = abs(nose.y - heel_y) * frame_height
    
    # Calculate percentage of frame occupied
    distance_percentage = (pixel_distance / frame_height) * 100
    
    # If person occupies > 80% of frame, they're too close
    if distance_percentage > 80:
        return False, distance_percentage, "⚠️ Too close to camera! Move back about 1-2 steps."
    
    return True, distance_percentage, None

# Global buffer for temporal smoothing
shoulder_buffer = deque(maxlen=20)

def apply_temporal_smoothing(shoulder_width_cm):
    """
    Apply temporal smoothing to shoulder width by averaging over 20 frames.
    """
    shoulder_buffer.append(shoulder_width_cm)
    smoothed_width = np.mean(list(shoulder_buffer))
    return smoothed_width

def detect_a4_paper(image):
    """
    Detect A4 paper in the image using OpenCV.
    Returns the width in pixels if detected, else None.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) == 4:
            # Check aspect ratio for A4 (21:29.7 ≈ 0.707)
            rect = cv2.minAreaRect(contour)
            width, height = rect[1]
            if width > height:
                aspect = height / width
            else:
                aspect = width / height

            if 0.6 < aspect < 0.8:  # Allow some tolerance
                # Assume it's A4, return the width
                return min(width, height)  # A4 width is 21cm

    return None

def get_pixels_per_cm(image, user_height_cm=None):
    """
    Get pixels per cm using dual calibration.
    First try paper detection, else use height.
    """
    paper_width_pixels = detect_a4_paper(image)
    if paper_width_pixels:
        pixels_per_cm = paper_width_pixels / 21.0  # A4 width is 21cm
        print(f"Using paper calibration: {pixels_per_cm} pixels/cm")
        return pixels_per_cm

    if user_height_cm:
        # Use MediaPipe to detect pose and get height from nose to heel
        model_path = get_model_path()
        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE
        )
        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            result = landmarker.detect(mp_image)
            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                nose = landmarks[0]
                left_heel = landmarks[29]
                right_heel = landmarks[30]

                # Use the lower heel
                heel_y = min(left_heel.y, right_heel.y)
                pixel_distance = abs(nose.y - heel_y) * image.shape[0]
                pixels_per_cm = pixel_distance / user_height_cm
                print(f"Using height calibration: {pixels_per_cm} pixels/cm")
                return pixels_per_cm

    return None

def calculate_shoulder_width(image, pixels_per_cm):
    """
    Calculate shoulder width in cm using MediaPipe pose.
    Includes shoulder offset (+2.0cm) to account for deltoid muscle width.
    """
    model_path = get_model_path()
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE
    )
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        result = landmarker.detect(mp_image)
        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            
            # Validate user distance
            is_valid_distance, distance_pct, distance_warning = validate_user_distance(image, landmarks)
            
            # Get shoulder landmarks
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]

            # Get pixel coordinates
            h, w, _ = image.shape
            left_x = int(left_shoulder.x * w)
            right_x = int(right_shoulder.x * w)

            pixel_distance = abs(left_x - right_x)
            cm_distance = pixel_distance / pixels_per_cm
            
            # Apply shoulder offset (+2.0cm for deltoid/muscle width)
            SHOULDER_OFFSET = 2.0
            cm_distance_adjusted = cm_distance + SHOULDER_OFFSET
            
            return cm_distance_adjusted, result, is_valid_distance, distance_warning
        return None, None, None, None

def calculate_bmi(weight_kg, height_cm):
    """
    Calculate BMI from weight in kg and height in cm.
    """
    if height_cm:
        height_m = height_cm / 100.0
        bmi = weight_kg / (height_m ** 2)
        return bmi
    return None

def recommend_size(shoulder_width_cm, bmi=None):
    """
    Recommend T-shirt size based on shoulder width, with BMI refinement.
    Realistic thresholds for shoulder width (including +2cm offset):
    - S: < 40cm
    - M: 40-44cm
    - L: 44-48cm
    - XL: 48-52cm
    - XXL: > 52cm
    """
    if shoulder_width_cm < 40:
        base_size = "S"
    elif 40 <= shoulder_width_cm < 44:
        base_size = "M"
    elif 44 <= shoulder_width_cm < 48:
        base_size = "L"
    elif 48 <= shoulder_width_cm < 52:
        base_size = "XL"
    else:
        base_size = "XXL"

    fit_note = ""
    if bmi is not None:
        if bmi > 26:
            # Overweight, suggest larger size for relaxed fit
            if base_size == "S":
                fit_note = " (Consider M for a more relaxed Oversized fit)"
            elif base_size == "M":
                fit_note = " (Consider L for a more relaxed Oversized fit)"
            elif base_size == "L":
                fit_note = " (Consider XL for a more relaxed Oversized fit)"
            elif base_size == "XL":
                fit_note = " (Consider XXL for a more relaxed Oversized fit)"
            else:
                fit_note = " (Oversized fit recommended)"
        elif bmi < 18.5:
            # Underweight, suggest smaller or note
            fit_note = " (Slim fit recommended; consult size chart for precise fit)"

    return base_size, fit_note

def main(image_path=None, user_height_cm=None, user_weight_kg=None):
    """
    Main function to run the Vision-Fit system.
    If image_path is None, use webcam to capture.
    """
    if image_path:
        image = cv2.imread(image_path)
        if image is None:
            print("Error: Could not load image.")
            return
    else:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return
        print("Capturing image in 3 seconds...")
        import time
        time.sleep(3)
        ret, image = cap.read()
        if not ret:
            print("Error: Could not capture frame.")
            return
        cap.release()
        print("Image captured.")

    # Prompt for height if not provided
    if user_height_cm is None:
        try:
            user_height_cm = float(input("Enter your height in cm (default 175): ") or 175)
        except ValueError:
            user_height_cm = 175
            print("Using default height: 175 cm")

    # Prompt for weight
    if user_weight_kg is None:
        try:
            user_weight_kg = float(input("Enter your weight in kg (default 70): ") or 70)
        except ValueError:
            user_weight_kg = 70
            print("Using default weight: 70 kg")

    # Calculate BMI
    bmi = calculate_bmi(user_weight_kg, user_height_cm)
    if bmi:
        print(f"Calculated BMI: {bmi:.2f}")

    # Get calibration
    pixels_per_cm = get_pixels_per_cm(image, user_height_cm)
    if not pixels_per_cm:
        print("Error: Could not calibrate. Please provide height or include A4 paper in image.")
        return

    # Calculate shoulder width
    result_tuple = calculate_shoulder_width(image, pixels_per_cm)
    if result_tuple[0] is None:
        print("Error: Could not detect person pose.")
        return
    
    shoulder_width_cm, pose_results, is_valid_distance, distance_warning = result_tuple
    
    # Check distance validation
    if not is_valid_distance:
        print(distance_warning)
        return
    
    # Apply temporal smoothing (average over buffer)
    smoothed_shoulder_width = apply_temporal_smoothing(shoulder_width_cm)

    size, fit_note = recommend_size(smoothed_shoulder_width, bmi)
    print(f"Shoulder width (raw): {shoulder_width_cm:.2f} cm")
    print(f"Shoulder width (smoothed): {smoothed_shoulder_width:.2f} cm")
    print(f"Recommended size: {size}{fit_note}")

    # Visualization
    annotated_image = image.copy()
    # Note: Pose drawing removed for compatibility with new MediaPipe API

    # Add text
    cv2.putText(annotated_image, f"Size: {size}{fit_note}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(annotated_image, f"Shoulder (Raw): {shoulder_width_cm:.2f} cm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(annotated_image, f"Shoulder (Smoothed): {smoothed_shoulder_width:.2f} cm", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if bmi:
        cv2.putText(annotated_image, f"BMI: {bmi:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imwrite("vision_fit_result.jpg", annotated_image)
    print("Result saved as vision_fit_result.jpg")

if __name__ == "__main__":
    # Example usage: python vision_fit.py image.jpg 175 70
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        user_height_cm = float(sys.argv[2]) if len(sys.argv) > 2 else None
        user_weight_kg = float(sys.argv[3]) if len(sys.argv) > 3 else None
        main(image_path, user_height_cm, user_weight_kg)
    else:
        main()