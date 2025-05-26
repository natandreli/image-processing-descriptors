import numpy as np
import cv2

def extract_lab_features(images):
    """
    Extracts statistical and histogram features from the LAB color space for each image.
    This function converts each image from RGB to LAB space and calculates:
        - The mean and standard deviation for each channel (L, A, B).
        - A 32-bin histogram for each channel with values ​​in the range [0, 255].

    These features are combined into one vector per image, allowing them to be used
    as input for classification or analysis models.

    Parameters:
    -----------
    images: List of images in RGB format.

    Returns:
    --------
    features: Array with one feature vector per image. Each vector contains:
        - 6 statistics: mean and standard deviation for the L, A, and B channels (3 x 2)
        - 96 histogram values: 32 bins for each channel (3 x 32)
    """

    features = []
    
    for img in images:
        lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        
        l_mean, l_std = np.mean(lab_img[:,:,0]), np.std(lab_img[:,:,0])
        a_mean, a_std = np.mean(lab_img[:,:,1]), np.std(lab_img[:,:,1])
        b_mean, b_std = np.mean(lab_img[:,:,2]), np.std(lab_img[:,:,2])

        l_hist, _ = np.histogram(lab_img[:,:,0], bins=32, range=(0, 255))
        a_hist, _ = np.histogram(lab_img[:,:,1], bins=32, range=(0, 255))
        b_hist, _ = np.histogram(lab_img[:,:,2], bins=32, range=(0, 255))
        
        feature_vector = np.concatenate([
            [l_mean, l_std, a_mean, a_std, b_mean, b_std],
            l_hist, a_hist, b_hist
        ])
        features.append(feature_vector)
    
    return np.array(features)