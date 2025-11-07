import tkinter as tk
from tkinter import filedialog, Frame, Label, Button
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import os

# --- 1. GLOBAL CONSTANTS ---
CLASS_NAMES = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']


# --- 2. APPLICATION CLASS DEFINITION ---

class FlowerClassifierApp:
    def __init__(self, root_window):
        """Initialize the application."""
        self.root = root_window
        self.model = None

        # --- Constants for easy styling ---
        self.BG_COLOR = "#f7f9fc"
        self.BUTTON_COLOR = "#007aff"
        self.BUTTON_HOVER_COLOR = "#0056b3"
        self.TEXT_COLOR = "#333333"
        self.FONT_FAMILY = "Helvetica"

        # --- Configure the main window ---
        self.root.title("Flower Recognition AI")
        self.root.geometry("600x800")
        self.root.resizable(False, False)
        self.root.configure(bg=self.BG_COLOR)

        # --- Create the UI widgets and then load the model ---
        self.create_widgets()
        self.load_model()

    def load_model(self):
        """Load the trained Keras model from file."""
        model_path = 'flower_model_v2.keras'
        try:
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path, compile=False)
                print("✅ Model loaded successfully.")
            else:
                self.show_error(f"Error: Model file not found at '{model_path}'")
        except Exception as e:
            self.show_error(f"An error occurred while loading the model: {e}")

    def create_widgets(self):
        """Create and arrange all the UI elements in the window."""
        title = Label(
            self.root,
            text="Flower Classifier",
            font=(self.FONT_FAMILY, 28, "bold"),
            bg=self.BG_COLOR,
            fg=self.TEXT_COLOR,
            pady=20
        )
        title.pack()

        image_frame = Frame(self.root, bg="#ffffff", bd=2, relief="groove")
        # THIS IS THE CORRECTED LINE:
        image_frame.pack(padx=20, pady=10, fill="both", expand=True)

        # We no longer need width/height on the label, it will fill the frame
        self.image_panel = Label(
            image_frame,
            text="Your image will appear here",
            font=(self.FONT_FAMILY, 14),
            bg="#ffffff"
        )
        self.image_panel.pack(fill="both", expand=True)

        self.result_label = Label(
            self.root,
            text="Select an image to begin",
            font=(self.FONT_FAMILY, 20, "bold"),
            bg=self.BG_COLOR,
            fg=self.TEXT_COLOR
        )
        self.result_label.pack(pady=(20, 5))

        self.confidence_label = Label(
            self.root,
            text="",
            font=(self.FONT_FAMILY, 16),
            bg=self.BG_COLOR,
            fg=self.TEXT_COLOR
        )
        self.confidence_label.pack()

        select_button = Button(
            self.root,
            text="Select Image",
            font=(self.FONT_FAMILY, 16, "bold"),
            command=self.select_image_and_predict,
            bg=self.BUTTON_COLOR,
            fg="white",
            pady=10,
            padx=20,
            relief="flat",
            activebackground=self.BUTTON_HOVER_COLOR,
            activeforeground="white",
            borderwidth=0
        )
        select_button.pack(pady=20)

        select_button.bind("<Enter>", lambda e: e.widget.config(bg=self.BUTTON_HOVER_COLOR))
        select_button.bind("<Leave>", lambda e: e.widget.config(bg=self.BUTTON_COLOR))

    def select_image_and_predict(self):
        """Handle the image selection and prediction process."""
        file_path = filedialog.askopenfilename(
            title="Select a Flower Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
        )
        if not file_path:
            return

        self.display_image(file_path)
        self.result_label.config(text="Analyzing...")
        self.confidence_label.config(text="")
        self.root.update_idletasks()
        self.run_prediction(file_path)

    def display_image(self, file_path):
        """Open and show the selected image in the UI panel."""
        # Get the actual size of the frame to fit the image perfectly
        frame_w = self.image_panel.winfo_width()
        frame_h = self.image_panel.winfo_height()

        img = Image.open(file_path)
        img.thumbnail((frame_w - 10, frame_h - 10)) # Use frame size, with a small margin
        photo = ImageTk.PhotoImage(img)
        self.image_panel.config(image=photo, text="")
        self.image_panel.image = photo

    def run_prediction(self, image_path):
        """Preprocess the image and get a prediction from the model."""
        if self.model is None:
            self.show_error("Model is not loaded. Cannot predict.")
            return

        try:
            img = tf.keras.utils.load_img(image_path, target_size=(180, 180))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)

            predictions = self.model.predict(img_array, verbose=0)
            score = predictions[0]

            predicted_class = CLASS_NAMES[np.argmax(score)]
            confidence = 100 * np.max(score)

            self.result_label.config(text=f"Prediction: {predicted_class.title()}")
            self.confidence_label.config(text=f"Confidence: {confidence:.2f}%")

        except Exception as e:
            self.show_error(f"Prediction Error: {e}")

    def show_error(self, message):
        """Display an error message in the UI."""
        self.result_label.config(text=message, fg="red")
        self.confidence_label.config(text="")


# --- 3. SCRIPT EXECUTION ---

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    main_window = tk.Tk()
    app = FlowerClassifierApp(main_window)
    main_window.mainloop()