import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
from pathlib import Path
from sudoku_extractor_from_image import solve_sudoku_from_image
from extract_puzzle_boundaries_from_screen import extract_puzzle

class SudokuCameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sudoku Solver - Camera Capture")
        self.root.geometry("1100x700")
        
        # Camera variables
        self.camera = None
        self.camera_running = False
        self.captured_image = None
        
        # Create GUI components
        self.create_widgets()


    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Sudoku Solver with Camera", font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Camera display frame
        camera_frame = ttk.LabelFrame(main_frame, text="Camera View", padding="10")
        camera_frame.grid(row=1, column=0, padx=10, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.camera_label = ttk.Label(camera_frame)
        self.camera_label.grid(row=0, column=0)
        
        # Control buttons frame
        control_frame = ttk.Frame(main_frame, padding="10")
        control_frame.grid(row=2, column=0, pady=10)
        
        self.start_camera_btn = ttk.Button(control_frame, text="Start Camera", command=self.start_camera)
        self.start_camera_btn.grid(row=0, column=0, padx=5)
        
        self.capture_btn = ttk.Button(control_frame, text="Capture Image", command=self.capture_image, state='disabled')
        self.capture_btn.grid(row=0, column=1, padx=5)
        
        self.solve_btn = ttk.Button(control_frame, text="Solve Sudoku", command=self.solve_sudoku, state='disabled')
        self.solve_btn.grid(row=0, column=2, padx=5)
        
        self.stop_camera_btn = ttk.Button(control_frame, text="Stop Camera", command=self.stop_camera, state='disabled')
        self.stop_camera_btn.grid(row=0, column=3, padx=5)
        
        # Result frame
        result_frame = ttk.LabelFrame(main_frame, text="Captured/Result", padding="10")
        result_frame.grid(row=1, column=1, padx=10, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.result_label = ttk.Label(result_frame)
        self.result_label.grid(row=0, column=0)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready. Click 'Start Camera' to begin.")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)


    # Start the camera feed
    def start_camera(self):
        try:
            # Try to open the default camera (0)
            self.camera = cv2.VideoCapture(0)
            
            if not self.camera.isOpened():
                messagebox.showerror("Error", "Could not access camera. Please check if it's connected.")
                return
            
            self.camera_running = True
            self.start_camera_btn.config(state='disabled')
            self.stop_camera_btn.config(state='normal')
            self.capture_btn.config(state='normal')
            self.status_var.set("Camera is running. Position the Sudoku puzzle and click 'Capture Image'.")
            
            self.update_camera_feed()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start camera: {str(e)}")


    # Update the camera feed display
    def update_camera_feed(self):
        if self.camera_running and self.camera.isOpened():
            ret, frame = self.camera.read()
            
            if ret:
                # Convert from BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize frame for display
                display_width = 640
                display_height = 480
                frame_resized = cv2.resize(frame_rgb, (display_width, display_height))
                
                # Convert to PIL Image
                img = Image.fromarray(frame_resized)
                img_tk = ImageTk.PhotoImage(image=img)
                
                # Update the label
                self.camera_label.img_tk = img_tk
                self.camera_label.configure(image=img_tk)
            
            # Schedule the next update
            self.root.after(10, self.update_camera_feed)


    # Capture the current camera frame
    def capture_image(self):
        if self.camera and self.camera.isOpened():
            ret, frame = self.camera.read()
            
            if ret:
                # Store the captured frame
                self.captured_image = frame.copy()
                
                # Display the captured image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = frame_rgb.copy()
                
                img = Image.fromarray(frame_resized)
                img_tk = ImageTk.PhotoImage(image=img)
                
                self.result_label.img_tk = img_tk
                self.result_label.configure(image=img_tk)
                
                # Save the captured image temporarily
                cv2.imwrite('screenshoot.png', self.captured_image)
                #cv2.imwrite('sudoku.png', self.captured_image)
                
                self.solve_btn.config(state='normal')
                self.status_var.set("Image captured! Click 'Solve Sudoku' to process.")
                
                messagebox.showinfo("Success", "Sudoku image captured successfully!")
            else:
                messagebox.showerror("Error", "Failed to capture image from camera.")

    def display_result_image(self, image):
        """
        Display an image in the result panel (right side)

        Args:
            image: Can be either:
                   - A file path (string) to an image
                   - A numpy array (OpenCV image in BGR format)
                   - A PIL Image object
        """
        try:
            # Handle different input types
            if isinstance(image, str):
                # It's a file path
                img_bgr = cv2.imread(image)
                if img_bgr is None:
                    raise ValueError(f"Could not load image from {image}")
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            elif isinstance(image, np.ndarray):
                # It's a numpy array (OpenCV image)
                img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                # Assume it's a PIL Image
                img_rgb = np.array(image)

            # Resize for display
            display_width = 400
            display_height = 400
            img_resized = cv2.resize(img_rgb, (display_width, display_height))

            # Convert to PIL Image and then to PhotoImage
            img_pil = Image.fromarray(img_resized)
            #img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(image=img_pil)

            # Update the result label
            self.result_label.img_tk = img_tk
            self.result_label.configure(image=img_tk)

        except Exception as e:
            messagebox.showerror("Display Error", f"Failed to display result image: {str(e)}")


    # solve sudoku from image
    def solve_sudoku(self):
        if self.captured_image is None:
            messagebox.showerror("Error", "Please capture an image first.")
            return
        
        try:
            self.status_var.set("Processing Sudoku puzzle")
            # screen pre-process : reads the 374K image 640x480 resolution and extract the puzzle boundaries o
            result = extract_puzzle("screenshoot.png", debug=False)
            #cv2.imwrite("extracted_puzzle.png", result)
            board_solution = solve_sudoku_from_image("extracted_puzzle.png")
            self.display_result_image("output.png")
            self.captured_image = cv2.imread("output.png")


            # For now, just show a placeholder message
            messagebox.showinfo("Sudoku Solver", 
                              "Integration point ready!\n\n"
                              "The captured image is saved at:\n"
                              "output.png")
            
            self.status_var.set("Sudoku solving completed!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to solve Sudoku: {str(e)}")
            self.status_var.set("Error occurred during solving.")


    # Stop the camera feed
    def stop_camera(self):
        self.camera_running = False
        
        if self.camera:
            self.camera.release()
            self.camera = None
        
        self.start_camera_btn.config(state='normal')
        self.stop_camera_btn.config(state='disabled')
        self.capture_btn.config(state='disabled')
        
        # Clear camera display
        self.camera_label.configure(image='')
        
        self.status_var.set("Camera stopped.")


    # Handle window closing
    def on_closing(self):
        self.stop_camera()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = SudokuCameraApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
