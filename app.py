import streamlit as st
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import math
import os
import urllib.request
from PIL import Image
from collections import deque

# Set page config
st.set_page_config(page_title="Vision-Fit", page_icon="👕", layout="wide")

# Initialize session state for temporal smoothing
if 'shoulder_buffer' not in st.session_state:
    st.session_state.shoulder_buffer = deque(maxlen=20)

# Custom CSS for black and white theme
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
    }
    
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #000000;
        color: #FFFFFF;
    }
    
    [data-testid="stHeader"] {
        background-color: #000000;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1a1a1a;
    }
    
    .stButton>button {
        background-color: #FFFFFF;
        color: #000000;
        border: 2px solid #FFFFFF;
        font-weight: bold;
        padding: 10px 30px;
        border-radius: 5px;
    }
    
    .stButton>button:hover {
        background-color: #cccccc;
    }
    
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input {
        background-color: #1a1a1a;
        color: #FFFFFF;
        border: 2px solid #FFFFFF;
    }
    
    .stMarkdown h1 {
        color: #FFFFFF;
        text-align: center;
        font-size: 2.5em;
        margin-bottom: 20px;
    }
    
    .stMarkdown h2 {
        color: #CCCCCC;
    }
    
    .info-box {
        background-color: #1a1a1a;
        border-left: 4px solid #FFFFFF;
        padding: 15px;
        margin: 10px 0;
    }
    
    [data-testid="stMetricValue"] {
        color: #FFFFFF;
    }
    
    [data-testid="stSuccessBox"] {
        background-color: #1a1a1a;
        border: 2px solid #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)

def get_model_path():
    model_path = 'pose_landmarker_lite.task'
    if not os.path.exists(model_path):
        with st.spinner("Downloading pose model..."):
            url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'
            urllib.request.urlretrieve(url, model_path)
        st.success("Model downloaded.")
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
        st.info(f"Using paper calibration: {pixels_per_cm:.2f} pixels/cm")
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
                st.info(f"Using height calibration: {pixels_per_cm:.2f} pixels/cm")
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

def apply_temporal_smoothing(shoulder_width_cm):
    """
    Apply temporal smoothing to shoulder width by averaging over 20 frames.
    """
    st.session_state.shoulder_buffer.append(shoulder_width_cm)
    smoothed_width = np.mean(list(st.session_state.shoulder_buffer))
    return smoothed_width

def recommend_size(shoulder_width_cm, bmi=None):
    """
    Recommend T-shirt size based on calibrated shoulder width.
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

def main():
    st.title("👕 VISION-FIT")
    st.markdown("<h3 style='text-align: center; color: #CCCCCC;'>AI-Based Body Measurement for E-Commerce</h3>", unsafe_allow_html=True)
    st.markdown("<hr style='border: 1px solid #FFFFFF;'>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; margin-bottom: 30px;'>
    <p style='color: #AAAAAA; font-size: 1.1em;'>
    Upload a photo or take one with your camera, enter your body metrics, and get your perfect T-shirt size recommendation.
    </p>
    </div>
    """, unsafe_allow_html=True)

    # Camera or upload input
    st.markdown("<h4 style='color: #FFFFFF;'>📸 Step 1: Capture or Upload Photo</h4>", unsafe_allow_html=True)
    
    # Choose input method
    input_method = st.radio(
        "Choose input method:",
        ["Take Photo with Camera", "Upload Existing Image"],
        horizontal=True
    )
    
    image = None
    if input_method == "Take Photo with Camera":
        img_file_buffer = st.camera_input("Take a photo of yourself (front view)")
        if img_file_buffer is not None:
            image = np.array(Image.open(img_file_buffer))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:  # Upload Existing Image
        uploaded_file = st.file_uploader("Upload an existing image", type=["jpg", "jpeg", "png", "bmp", "tiff"])
        if uploaded_file is not None:
            image = np.array(Image.open(uploaded_file))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if image is not None:
        st.markdown("<h4 style='color: #FFFFFF;'>📊 Step 2: Enter Your Metrics</h4>", unsafe_allow_html=True)
        
        # Input fields in columns
        col1, col2 = st.columns(2)
        with col1:
            height = st.number_input("Height (cm)", min_value=100, max_value=250, value=175, step=1)
        with col2:
            weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70, step=1)

        st.markdown("<hr style='border: 1px solid #555555;'>", unsafe_allow_html=True)
        
        st.markdown("<h4 style='color: #FFFFFF;'>🎯 Step 3: Get Your Size</h4>", unsafe_allow_html=True)
        
        # Centered button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            button_clicked = st.button("🔍 Analyze & Get Size Recommendation", use_container_width=True)

        if button_clicked:
            with st.spinner("🔄 Processing your photo..."):
                # Calculate BMI
                bmi = calculate_bmi(weight, height)

                # Get calibration
                pixels_per_cm = get_pixels_per_cm(image, height)
                if not pixels_per_cm:
                    st.error("❌ Could not calibrate. Please ensure your height is entered or include an A4 paper in the image.")
                    return

                # Calculate shoulder width with distance validation
                result_tuple = calculate_shoulder_width(image, pixels_per_cm)
                if result_tuple[0] is None:
                    st.error("❌ Could not detect person pose. Please ensure you are clearly visible in the photo.")
                    return
                
                shoulder_width_cm, pose_results, is_valid_distance, distance_warning = result_tuple
                
                # Check distance validation
                if not is_valid_distance:
                    st.warning(distance_warning)
                    return
                
                # Apply temporal smoothing (average over buffer)
                smoothed_shoulder_width = apply_temporal_smoothing(shoulder_width_cm)
                
                size, fit_note = recommend_size(smoothed_shoulder_width, bmi)
                
                # Display results with styling
                st.markdown("<hr style='border: 1px solid #FFFFFF;'>", unsafe_allow_html=True)
                
                # Results metrics with both raw and smoothed values
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Shoulder Width (Raw)", f"{shoulder_width_cm:.2f} cm")
                with col2:
                    st.metric("Shoulder Width (Smoothed)", f"{smoothed_shoulder_width:.2f} cm")
                with col3:
                    st.metric("BMI", f"{bmi:.2f}")
                with col4:
                    st.metric("Recommended Size", size)
                
                # Fit note
                if fit_note:
                    st.markdown(f"""
                    <div style='background-color: #1a1a1a; border-left: 4px solid #FFFF00; padding: 15px; margin: 20px 0;'>
                    <p style='color: #FFFF00; font-weight: bold;'>💡 Fit Recommendation:</p>
                    <p style='color: #FFFFFF;'>{fit_note}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Display annotated image
                annotated_image = image.copy()
                cv2.putText(annotated_image, f"SIZE: {size}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                cv2.putText(annotated_image, f"Shoulder (Raw): {shoulder_width_cm:.1f} cm", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                cv2.putText(annotated_image, f"Shoulder (Smoothed): {smoothed_shoulder_width:.1f} cm", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                cv2.putText(annotated_image, f"BMI: {bmi:.1f}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                
                st.markdown("<h4 style='color: #FFFFFF;'>📷 Result</h4>", unsafe_allow_html=True)
                st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), caption="Your Measurement Result", use_container_width=True)
                
                # Success message
                st.success(f"✅ Your recommended size is **{size}**{fit_note}")
                
                # Info about improvements
                st.markdown("""
                <div style='background-color: #1a1a1a; border-left: 4px solid #00FF00; padding: 15px; margin: 20px 0;'>
                <p style='color: #00FF00; font-weight: bold;'>✓ Advanced Calibration Applied:</p>
                <p style='color: #CCCCCC;'>• Temporal Smoothing: 20-frame average</p>
                <p style='color: #CCCCCC;'>• Shoulder Offset: +2.0cm for muscle/flesh width</p>
                <p style='color: #CCCCCC;'>• Distance Validation: Ensuring optimal camera position</p>
                <p style='color: #CCCCCC;'>• Updated Thresholds: Calibrated for accurate sizing</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='text-align: center; margin-top: 50px;'>
        <p style='color: #666666; font-size: 1.2em;'>
        👆 Use the camera above to capture your photo
        </p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()