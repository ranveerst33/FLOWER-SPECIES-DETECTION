# 🌸 Flower Recognition AI with Python & TensorFlow

This project is a desktop application that uses a custom-built Convolutional Neural Network (CNN) to classify images of flowers into five different categories. The model is built from scratch using TensorFlow/Keras, and the user-friendly interface is created with Python's built-in Tkinter library.

---

## ✨ Features

-   **Custom CNN Model:** A Convolutional Neural Network built from scratch to learn and identify flower features.
-   **Intuitive Graphical Interface:** A simple and clean GUI built with Tkinter allows users to easily upload an image and get a prediction.
-   **Real-Time Prediction:** Classifies images of daisies, dandelions, roses, sunflowers, and tulips.
-   **Confidence Score:** Displays the model's confidence in its prediction.
-   **Complete Training Pipeline:** Includes the full training script (`flower_recognition.py`), allowing anyone to train, experiment with, and improve the model.

---

## 🛠️ Tech Stack

-   **Backend & Model:** Python, TensorFlow, Keras
-   **User Interface:** Tkinter
-   **Image Processing:** Pillow (PIL), NumPy

---

## 👥 Authors

-   **Ranveer Singh Thakur** (Author) - `(ranveersinghthakur33@gmail.com)`
-   **Aryaman Giri** (Co-Author) 

---

## 🚀 Getting Started

Because the trained model file (`.keras`) is too large for this repository, you will generate it yourself by running the training script. This ensures you have a working model trained on your own machine.

### 1. Prerequisites

-   Python 3.9+
-   An Apple Silicon Mac (M1, M2, etc.) is recommended to follow these steps exactly.

### 2. Installation

First, create the necessary files to manage your project professionally.

**A. Create a `.gitignore` file**

This is the most important step to prevent you from accidentally trying to upload the large model file. Create a file named `.gitignore` and paste the following into it:

```
# Keras model files
*.keras

# Python virtual environment
venv/
__pycache__/

# macOS system files
.DS_Store
```

**B. Create a `requirements.txt` file**

This file lists all the necessary Python libraries. Create a file named `requirements.txt` and paste the following into it:

```
tensorflow-macos
tensorflow-metal
numpy
matplotlib
Pillow
```

**C. Install the Project**

Now, open your terminal and follow these steps:

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME

# 2. Create and activate a Python virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install all the required libraries
pip install -r requirements.txt
```

---

## 📋 Usage: The Two-Step Workflow

This project has two main scripts that must be run in order.

### Step 1: Train the Model (The "Factory")

First, you must run the training script. This will process the entire flower dataset and create the `flower_model_v2.keras` file which is required by the application.

In your terminal, run:
```bash
python flower_recognition.py
```
-   This process will take a significant amount of time (15-30 minutes depending on your machine) as it trains the model for 50 epochs.
-   You will see the progress for each epoch printed in the terminal.
-   When it's finished, you will have a new file named **`flower_model_v2.keras`** in your project folder.

### Step 2: Run the Application (The "Product")

Once the model file has been created, you can run the user-friendly application anytime you want.

In your terminal, run:
```bash
python predict.py
```
-   This will launch the graphical user interface.
-   Click the "Select Image" button, choose a picture of a flower, and the AI will display its prediction and confidence score!

---

## 📂 Project Structure

```
.
├── assets
│   └── screenshot.png      # Screenshot of the UI
├── flower_recognition.py   # The main script for training the AI model
├── predict.py              # The script for the UI and making predictions
├── requirements.txt        # List of Python dependencies
└── README.md               # This file
```

---

## 💡 Future Improvements

-   **Expand the Dataset:** Add more flower categories to make the model more versatile.
-   **Implement Transfer Learning:** Use a pre-trained model like MobileNetV2 as a base to achieve higher accuracy (>90%) with less training time.
-   **Web Deployment:** Rebuild the application using a web framework like Flask or Django to make it accessible from any browser.
-   **UI Enhancements:** Add a feature to display the full probability breakdown for all flower classes in the UI.
