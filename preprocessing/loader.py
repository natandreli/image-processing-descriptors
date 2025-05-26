import os
import cv2

def load_dataset(path, size=(128, 128)):
    """
    Loads a set of images from a directory organized by class.
    Each image is read using OpenCV, resized to a specific size,
    and stored along with its label.

    Parameters:
    ----------
    path : Path to the main directory containing the subfolders per class.
    size : Size (width, height) to which all images will be resized.
    Default is (128, 128).

    Returns:
    -------
    images : List of loaded and resized images.
    labels : List of labels corresponding to each image,
    extracted from the subfolder name.
    """
    images = []
    labels = []
    for label in os.listdir(path):
        class_path = os.path.join(path, label)
        if not os.path.isdir(class_path):
            continue
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, size)
                images.append(img)
                labels.append(label)
    return images, labels
