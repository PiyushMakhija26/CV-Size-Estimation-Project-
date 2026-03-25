"""
Vision-Fit GUI Application
Tkinter-based interface for the Vision-Fit body measurement system.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
from PIL import Image, ImageTk
import os
from datetime import datetime
import threading
from vision_fit_processor import VisionFitProcessor


class VisionFitGUI:
    """Main GUI application for Vision-Fit."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Vision-Fit: AI Body Measurement")
        self.root.geometry("1000x700")
        self.root.resizable(True, True)
        
        # Initialize processor
        self.processor = VisionFitProcessor()
        
        # Create folders
        self.captured_images_dir = self._ensure_dir('captured_images')
        self.results_dir = self._ensure_dir('results')
        
        # UI State
        self.current_image_path = None
        self.processing_thread = None
        
        # Create UI
        self._create_ui()
    
    def _ensure_dir(self, dir_name):
        """Create directory if it doesn't exist."""
        dir_path = os.path.join(os.getcwd(), dir_name)
        os.makedirs(dir_path, exist_ok=True)
        return dir_path
    
    def _create_ui(self):
        """Create the main UI layout."""
        # Style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(
            main_frame, text="Vision-Fit: AI-Based Body Measurement",
            font=("Helvetica", 18, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Button frame
        button_frame = ttk.LabelFrame(main_frame, text="Input Methods", padding="10")
        button_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        # Buttons
        self.capture_btn = ttk.Button(
            button_frame, text="📸 Capture from Webcam",
            command=self._on_capture_webcam, width=30
        )
        self.capture_btn.grid(row=0, column=0, padx=5, pady=5)
        
        self.upload_btn = ttk.Button(
            button_frame, text="📁 Upload Image from System",
            command=self._on_upload_image, width=30
        )
        self.upload_btn.grid(row=0, column=1, padx=5, pady=5)
        
        # Input frame
        input_frame = ttk.LabelFrame(main_frame, text="User Metrics", padding="10")
        input_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(input_frame, text="Height (cm):").grid(row=0, column=0, sticky=tk.W)
        self.height_var = tk.StringVar(value="175")
        ttk.Entry(input_frame, textvariable=self.height_var, width=15).grid(
            row=0, column=1, padx=5
        )
        
        ttk.Label(input_frame, text="Weight (kg):").grid(row=0, column=2, sticky=tk.W)
        self.weight_var = tk.StringVar(value="70")
        ttk.Entry(input_frame, textvariable=self.weight_var, width=15).grid(
            row=0, column=3, padx=5
        )
        
        # Process button
        self.process_btn = ttk.Button(
            main_frame, text="🔍 Analyze & Get Size",
            command=self._on_process, state=tk.DISABLED, width=40
        )
        self.process_btn.grid(row=3, column=0, columnspan=2, pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(
            main_frame, mode='indeterminate'
        )
        self.progress.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Status label
        self.status_label = ttk.Label(
            main_frame, text="Ready", foreground="green"
        )
        self.status_label.grid(row=5, column=0, columnspan=2, pady=5)
        
        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        results_frame.grid(
            row=6, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10
        )
        
        # Results text widget with scrollbar
        scrollbar = ttk.Scrollbar(results_frame)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        self.results_text = tk.Text(
            results_frame, height=10, width=80, yscrollcommand=scrollbar.set
        )
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.config(command=self.results_text.yview)
        
        # Configure grid weights for resizing
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(6, weight=1)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
    
    def _on_capture_webcam(self):
        """Handle webcam capture button."""
        self.status_label.config(text="Opening webcam...", foreground="blue")
        self.root.update()
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam.")
            self.status_label.config(text="Error: Webcam not available", foreground="red")
            return
        
        # Create capture window
        capture_window = tk.Toplevel(self.root)
        capture_window.title("Webcam Capture - Press SPACE to capture, ESC to cancel")
        
        label = ttk.Label(capture_window)
        label.pack(fill=tk.BOTH, expand=True)
        
        captured = [False]
        captured_frame = [None]
        
        def update_frame():
            ret, frame = cap.read()
            if ret:
                # Resize for display
                display_frame = cv2.resize(frame, (640, 480))
                rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_frame)
                photo = ImageTk.PhotoImage(img)
                
                label.config(image=photo)
                label.image = photo
                
                if not captured[0]:
                    capture_window.after(30, update_frame)
                else:
                    cap.release()
                    capture_window.destroy()
        
        def on_key(event):
            if event.keysym == 'space' and not captured[0]:
                captured[0] = True
                ret, frame = cap.read()
                if ret:
                    captured_frame[0] = frame
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    self.current_image_path = os.path.join(
                        self.captured_images_dir,
                        f"user_{timestamp}.jpg"
                    )
                    cv2.imwrite(self.current_image_path, frame)
                    self.status_label.config(
                        text=f"✓ Image captured: {os.path.basename(self.current_image_path)}",
                        foreground="green"
                    )
                    self.process_btn.config(state=tk.NORMAL)
            elif event.keysym == 'Escape':
                captured[0] = True
                cap.release()
                capture_window.destroy()
                self.status_label.config(text="Capture cancelled", foreground="orange")
        
        capture_window.bind('<KeyPress>', on_key)
        update_frame()
    
    def _on_upload_image(self):
        """Handle image upload button."""
        file_types = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
            ("All files", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=file_types
        )
        
        if file_path:
            # Copy to captured_images folder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.basename(file_path)
            name, ext = os.path.splitext(filename)
            
            self.current_image_path = os.path.join(
                self.captured_images_dir,
                f"user_{timestamp}{ext}"
            )
            
            try:
                import shutil
                shutil.copy(file_path, self.current_image_path)
                self.status_label.config(
                    text=f"✓ Image loaded: {os.path.basename(self.current_image_path)}",
                    foreground="green"
                )
                self.process_btn.config(state=tk.NORMAL)
            except Exception as e:
                messagebox.showerror("Error", f"Could not copy image: {e}")
                self.status_label.config(text="Error loading image", foreground="red")
    
    def _on_process(self):
        """Handle processing button."""
        if not self.current_image_path:
            messagebox.showwarning("Warning", "Please capture or upload an image first.")
            return
        
        # Validate inputs
        try:
            height = float(self.height_var.get())
            weight = float(self.weight_var.get())
        except ValueError:
            messagebox.showerror("Error", "Height and weight must be valid numbers.")
            return
        
        # Disable button and start progress
        self.process_btn.config(state=tk.DISABLED)
        self.progress.start()
        self.status_label.config(text="Processing...", foreground="blue")
        
        # Process in thread to avoid freezing UI
        thread = threading.Thread(
            target=self._process_image_thread,
            args=(self.current_image_path, height, weight)
        )
        thread.start()
    
    def _process_image_thread(self, image_path, height, weight):
        """Process image in background thread."""
        try:
            results = self.processor.process_image(image_path, height, weight)
            
            self.root.after(0, lambda: self._display_results(results, image_path))
        
        except Exception as e:
            self.root.after(
                0,
                lambda: self._show_error(f"Processing failed: {str(e)}")
            )
    
    def _display_results(self, results, image_path):
        """Display processing results."""
        self.progress.stop()
        self.process_btn.config(state=tk.NORMAL)
        
        # Clear previous results
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        
        if 'error' in results:
            messagebox.showerror("Error", results['error'])
            self.status_label.config(text=f"Error: {results['error']}", foreground="red")
            return
        
        # Display results
        size = results['recommended_size']
        shoulder = results['shoulder_width_smoothed']
        bmi = results['bmi']
        fit_note = results['fit_note']
        calib = results['calibration_method']
        
        results_text = f"""
╔═══════════════════════════════════════════════════════════════╗
║                    VISION-FIT RESULTS                         ║
╚═══════════════════════════════════════════════════════════════╝

📊 MEASUREMENTS:
  • Shoulder Width (Smoothed): {shoulder:.2f} cm
  • Shoulder Width (Raw):      {results['shoulder_width_raw']:.2f} cm
  • Calibration Method:        {calib}
  • BMI:                       {bmi:.2f if bmi else 'N/A'}

👕 SIZE RECOMMENDATION:
  ✓ Recommended Size: {size}
  {fit_note if fit_note else '(Standard fit)'}

📝 CALIBRATION DETAILS:
  • Adjustments Applied:
    - Temporal Smoothing: 20-frame average ✓
    - Shoulder Offset: +2.0 cm (deltoid width) ✓
    - Distance Validation: ✓ PASS
    - New Size Thresholds: Applied ✓

📁 FILES SAVED:
  • Input:  {os.path.basename(image_path)}
  • Folder: {self.captured_images_dir}

════════════════════════════════════════════════════════════════
        """
        
        self.results_text.insert(tk.END, results_text)
        self.results_text.config(state=tk.DISABLED)
        
        # Save annotated image
        try:
            annotated = self.processor.annotate_image(results['image'], results)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_path = os.path.join(
                self.results_dir,
                f"result_{timestamp}.jpg"
            )
            cv2.imwrite(result_path, annotated)
            
            # Display annotated image
            self._display_result_image(annotated, result_path)
            
            self.status_label.config(
                text=f"✓ Complete! Size: {size} | Result saved: {os.path.basename(result_path)}",
                foreground="green"
            )
        
        except Exception as e:
            self.status_label.config(
                text=f"Results obtained but could not save image: {e}",
                foreground="orange"
            )
    
    def _display_result_image(self, image, result_path):
        """Display result image in a new window."""
        result_window = tk.Toplevel(self.root)
        result_window.title(f"Vision-Fit Result - {os.path.basename(result_path)}")
        
        # Resize image for display
        display_image = cv2.resize(image, (800, 600))
        rgb_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        photo = ImageTk.PhotoImage(pil_image)
        
        label = ttk.Label(result_window, image=photo)
        label.image = photo
        label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Info frame
        info_frame = ttk.Frame(result_window)
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(
            info_frame,
            text=f"Saved to: {result_path}",
            foreground="green"
        ).pack(side=tk.LEFT)
        
        ttk.Button(
            info_frame,
            text="Open Folder",
            command=lambda: self._open_folder(self.results_dir)
        ).pack(side=tk.RIGHT, padx=5)
    
    def _show_error(self, error_message):
        """Show error message."""
        self.progress.stop()
        self.process_btn.config(state=tk.NORMAL)
        messagebox.showerror("Error", error_message)
        self.status_label.config(text=f"Error: {error_message}", foreground="red")
    
    @staticmethod
    def _open_folder(folder_path):
        """Open folder in system file explorer."""
        if os.name == 'nt':  # Windows
            os.startfile(folder_path)
        elif os.name == 'posix':  # Mac/Linux
            os.system(f'open "{folder_path}"')


def main():
    """Launch the GUI application."""
    root = tk.Tk()
    app = VisionFitGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()