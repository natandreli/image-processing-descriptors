import numpy as np
import cv2

def extract_hog_features(images):
    """
    Extracts HOG features from a list of images.
    This function uses OpenCV's HOGDescriptor to extract shape and texture-related
    features that capture the distribution of edge directions (gradients) within
    an image.

    Parameters:
    -----------
    images : A list of images in RGB or grayscale format. Each image will be converted to
        grayscale if it has 3 channels.

    Returns:
    --------
    features : A 2D NumPy array where each row is the HOG feature vector of an image.
    """

    hog = cv2.HOGDescriptor(
        _winSize=(128, 128),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
        _derivAperture=1,
        _winSigma=-1,
        _histogramNormType=cv2.HOGDESCRIPTOR_L2HYS,
        _L2HysThreshold=0.2,
        _gammaCorrection=True,
        _nlevels=cv2.HOGDESCRIPTOR_DEFAULT_NLEVELS
    )
    
    features = []
    
    for img in images:
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
            
        resized = cv2.resize(gray, (128, 128))
        
        feat = hog.compute(resized)
        features.append(feat.flatten())
    
    return np.array(features)