from skimage.feature import hog
from skimage.color import rgb2gray
import numpy as np

def extract_hog_features(images):
    features = []
    for img in images:
        gray = rgb2gray(img)
        feat = hog(
            gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys'
        )
        features.append(feat)
    return np.array(features)
