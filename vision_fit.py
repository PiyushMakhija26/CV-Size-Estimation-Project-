import cv2
import os
from vision_fit_processor import VisionFitProcessor


def main():
    print("Starting Vision-Fit CLI")
    processor = VisionFitProcessor()
    image_path = None

    while True:
        if image_path is None:
            use_file = input("Upload image path or type 'cam' for webcam: ").strip()
            if use_file.lower() == 'cam':
                cam = cv2.VideoCapture(0)
                if not cam.isOpened():
                    print("Could not open webcam")
                    continue

                print("Capturing in 3 seconds...")
                cv2.waitKey(3000)
                ret, frame = cam.read()
                cam.release()
                if not ret:
                    print("Capture failed")
                    continue

                os.makedirs('captured_images', exist_ok=True)
                image_path = os.path.join('captured_images', 'cli_capture.png')
                cv2.imwrite(image_path, frame)
                print(f"Saved image to {image_path}")
            else:
                if not os.path.exists(use_file):
                    print("File does not exist")
                    continue
                image_path = use_file

        height = float(input("Enter height (cm): ").strip() or 175)
        weight = float(input("Enter weight (kg): ").strip() or 70)

        result = processor.process_image(image_path, height, weight)
        if 'error' in result:
            print("Error:", result['error'])
        else:
            print(f"Shoulder width: {result['shoulder_width']:.2f} cm")
            print(f"Smoothed width: {result['shoulder_width_smoothed']:.2f} cm")
            print(f"BMI: {result['bmi']:.2f}")
            print(f"Size: {result['recommended_size']} {result['fit_note']}")
            print("Calibration:", result['calibration_method'])

        again = input("Run again (y/n)? ").strip().lower()
        if again != 'y':
            break
        image_path = None


if __name__ == '__main__':
    main()
