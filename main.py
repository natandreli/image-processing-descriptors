from preprocessing.loader import load_dataset
from descriptors.hog import extract_hog_features
from descriptors.lab import extract_lab_features
from model.train_svm import train_svm

def main():
    print("[INFO] Loading dataset...")
    images, labels = load_dataset("data/dataset", size=(128, 128))
    print(f"[INFO] Loaded {len(images)} images from {len(set(labels))} classes.")

    print("[INFO] Extracting HOG features...")
    hog_features = extract_hog_features(images)
    train_svm(hog_features, labels, name_prefix='hog')

    print("[INFO] Extracting LAB features...")
    lab_features = extract_lab_features(images)
    train_svm(lab_features, labels, name_prefix='lab')

if __name__ == "__main__":
    main()
