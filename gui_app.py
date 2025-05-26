import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import joblib
import cv2
import numpy as np
import os

from descriptors.hog import extract_hog_features
from descriptors.lab import extract_lab_features

IMAGE_SIZE = (128, 128)

def load_models():
    """
    Loads pre-trained SVM models along with their corresponding label encoders
    and feature scalers.

    This function attempts to load models for both HOG and LAB feature descriptors.
    Each model is expected to have three associated files:
        - The SVM model file (`svm_<feature>_model.pkl`)
        - The label encoder (`<feature>_encoder.pkl`)
        - The feature scaler (`<feature>_scaler.pkl`)

    Parameters:
    -----------
    None

    Returns:
    --------
    models_data : A dictionary where each key is the name of a feature descriptor
        and the value is another dictionary with keys:
            - 'model': the trained SVM classifier
            - 'encoder': the label encoder used for class labels
            - 'scaler': the scaler used to normalize feature vectors
    """
    models_data = {}
    
    available_models = ['hog', 'lab']
    model_names = ['HOG', 'LAB']
    
    for model_file, model_name in zip(available_models, model_names):
        try:
            model_path = f'models/svm_{model_file}_model.pkl'
            encoder_path = f'models/{model_file}_encoder.pkl'
            scaler_path = f'models/{model_file}_scaler.pkl'
            
            if all(os.path.exists(path) for path in [model_path, encoder_path, scaler_path]):
                models_data[model_name] = {
                    'model': joblib.load(model_path),
                    'encoder': joblib.load(encoder_path),
                    'scaler': joblib.load(scaler_path)
                }
                print(f"[INFO] {model_name} model loaded successfully.")
            else:
                print(f"[WARNING] Files for {model_name} not found.")
        except Exception as e:
            print(f"[ERROR] Failed to load {model_name}: {e}")
    
    return models_data

def extract_features(image, method):
    """
    Extracts feature descriptors from a single image using the specified method.

    The function resizes the input image to a predefined size if needed, then
    extracts features using either HOG or LAB.

    Parameters:
    -----------
    image : The input image from which features will be extracted.

    method : The descriptor method to use. Supported values are:
        - 'HOG': for Histogram of Oriented Gradients
        - 'LAB': for LAB color space-based features

    Returns:
    --------
    features : A 1D array containing the extracted feature vector.

    Raises :
    -------
    ValueError:
        If the specified method is not supported.
    """

    if image.shape[:2] != IMAGE_SIZE:
        image = cv2.resize(image, IMAGE_SIZE)
    
    print(f"[DEBUG] Image resized to: {image.shape}")
    
    if method == 'HOG':
        features = extract_hog_features([image])[0]
    elif method == 'LAB':
        features = extract_lab_features([image])[0]
    else:
        raise ValueError("Invalid descriptor method.")
    
    print(f"[DEBUG] Features extracted: {features.shape}")
    return features

class ImageClassifierApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Classifier")
        self.master.geometry("600x700")
        
        print("[INFO] Loading models...")
        self.models_data = load_models()
        
        if not self.models_data:
            messagebox.showerror("Error", "Models could not be loaded")
            return
        
        self.setup_ui()
        
    def setup_ui(self):
        """
        Configures and lays out the user interface elements such as labels, 
        buttons, and radio buttons.
        """

        main_frame = ttk.Frame(self.master, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        title_label = ttk.Label(main_frame, text="Image Classifier", 
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 20))
        
        model_frame = ttk.LabelFrame(main_frame, text="Select Descriptor", padding="10")
        model_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.selected_method = tk.StringVar(value=list(self.models_data.keys())[0])
        
        for method in self.models_data.keys():
            ttk.Radiobutton(model_frame, text=method, variable=self.selected_method, 
                           value=method).pack(anchor=tk.W)
        
        ttk.Button(main_frame, text="Load Image", 
                  command=self.load_image).pack(pady=10)
        
        self.canvas_frame = ttk.Frame(main_frame)
        self.canvas_frame.pack(pady=10)
        
        self.canvas = ttk.Label(self.canvas_frame, text="No image loaded", 
                               background="lightgray", width=30)
        self.canvas.pack()
        
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        results_frame.pack(fill=tk.X, pady=20)
        
        self.result_label = ttk.Label(results_frame, text="Load an image to classify", 
                                     font=("Arial", 12))
        self.result_label.pack()
        
        self.debug_label = ttk.Label(results_frame, text="", font=("Arial", 9), 
                                    foreground="gray")
        self.debug_label.pack(pady=(10, 0))

    def load_image(self):
        """
        Opens a file dialog for the user to select an image, processes it, 
        and triggers classification.
        """

        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("JPEG", "*.jpg *.jpeg"),
                ("PNG", "*.png"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
            
        try:
            img = cv2.imread(file_path)
            if img is None:
                messagebox.showerror("Error", "Image could not be loaded")
                return
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            img_resized = cv2.resize(img_rgb, IMAGE_SIZE)
            
            self.display_image(img_resized)
            
            self.classify_image(img_resized)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error processing the image: {str(e)}")

    def display_image(self, img_array):
        """
        Displays the selected and resized image on the UI canvas.

        Parameters:
            img_array : RGB image array to be displayed.
        """

        try:
            img_display = cv2.resize(img_array, (200, 200))
            img_pil = Image.fromarray(img_display)
            img_tk = ImageTk.PhotoImage(img_pil)
            
            self.canvas.configure(image=img_tk, text="")
            self.canvas.image = img_tk
            
        except Exception as e:
            print(f"[ERROR] Error displaying image: {e}")

    def classify_image(self, image):
        """
        Classifies the given image using the selected descriptor method and 
        displays the results.

        Parameters:
            image : The image to classify, already resized to the 
            expected input size.
        """

        method = self.selected_method.get()
        
        if method not in self.models_data:
            messagebox.showerror("Error", f"Model {method} not available")
            return
        
        try:
            features = extract_features(image, method)
            
            model_info = self.models_data[method]
            model = model_info['model']
            encoder = model_info['encoder']
            scaler = model_info['scaler']
            
            features_scaled = scaler.transform([features])
            
            prediction_encoded = model.predict(features_scaled)[0]
            probabilities = model.predict_proba(features_scaled)[0]
            
            prediction = encoder.inverse_transform([prediction_encoded])[0]
            confidence = np.max(probabilities)
            
            result_text = f"Prediction: {prediction}\nConfidence: {confidence:.3f} ({confidence*100:.1f}%)"
            self.result_label.config(text=result_text)
            
            debug_text = f"Method: {method} | Classes: {len(encoder.classes_)}"
            self.debug_label.config(text=debug_text)
            
            print(f"[INFO] Classification successful: {prediction} (confidence: {confidence:.3f})")
            
        except Exception as e:
            error_msg = f"Error during classification: {str(e)}"
            print(f"[ERROR] {error_msg}")
            messagebox.showerror("Error", error_msg)
            self.result_label.config(text="Classification error")

def main():
    if not os.path.exists('models'):
        messagebox.showerror("Error", 
                           "'models' folder not found.\n"
                           "Make sure you have trained the models first.")
        return
    
    root = tk.Tk()
    app = ImageClassifierApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()