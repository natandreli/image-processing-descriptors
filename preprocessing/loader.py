import os
import cv2

def load_dataset(path, size=(128, 128)):
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
