import streamlit as st
import cv2
import numpy as np
import os
import time
from PIL import Image
from vision_fit_processor import VisionFitProcessor

st.set_page_config(page_title='Vision-Fit', page_icon='👕', layout='wide')

if 'processor' not in st.session_state:
    st.session_state.processor = VisionFitProcessor()

if 'shoulder_buffer' not in st.session_state:
    st.session_state.shoulder_buffer = []

# Styling (black/white theme)
st.markdown('''
<style>
html, body, [data-testid="stAppViewContainer"] { background-color: #000000; color: #FFFFFF; }
.stButton>button { background-color: #FFFFFF; color: #000000; border: 2px solid #FFFFFF; font-weight: bold; }
.stButton>button:hover { background-color: #cccccc; }
</style>
''', unsafe_allow_html=True)

st.title('👕 VISION-FIT')
st.markdown('### AI-Based Body Measurement for Size Recommendation')

input_method = st.radio('Choose input method:', ['Take Photo with Camera', 'Upload Existing Image'])

image = None

if input_method == 'Take Photo with Camera':
    uploaded = st.camera_input('Take a photo of yourself (front view)')
    if uploaded is not None:
        image = np.array(Image.open(uploaded))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
else:
    uploaded = st.file_uploader('Upload an existing image', type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'])
    if uploaded is not None:
        image = np.array(Image.open(uploaded))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

if image is not None:
    col1, col2 = st.columns(2)
    with col1:
        height = st.number_input('Height (cm)', min_value=100, max_value=250, value=175)
    with col2:
        weight = st.number_input('Weight (kg)', min_value=30, max_value=200, value=70)

    if st.button('🔍 Analyze & Get Size Recommendation'):
        os.makedirs('captured_images', exist_ok=True)
        ts = int(time.time())
        temp_path = os.path.join('captured_images', f'streamlit_{ts}.png')
        cv2.imwrite(temp_path, image)

        with st.spinner('Processing...'):
            result = st.session_state.processor.process_image(temp_path, height, weight)

        if 'error' in result:
            st.error(result['error'])
        else:
            st.success(f"Recommended size: {result['recommended_size']} {result.get('fit_note','')}")
            st.metric('Shoulder width', f"{result['shoulder_width']:.2f} cm")
            st.metric('Smoothed width', f"{result['shoulder_width_smoothed']:.2f} cm")
            st.metric('BMI', f"{result['bmi']:.2f}")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption='Input image', use_container_width=True)
            st.write(result)

else:
    st.info('Please capture or upload an image to continue.')

if __name__ == '__main__':
    pass
