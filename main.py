from preprocessing.loader import load_dataset
from descriptors.hog import extract_hog_features
from models.train_svm import train_svm

def main():
    print("[INFO] Loading dataset...")
    images, labels = load_dataset("data/dataset", size=(128, 128))
    print(f"[INFO] Loaded {len(images)} images from {len(set(labels))} classes.")

    print("[INFO] Extracting HOG features...")
    hog_features = extract_hog_features(images)
    print(f"[INFO] Feature matrix shape: {hog_features.shape}")

    print("[INFO] Training classifier...")
    model, encoder = train_svm(hog_features, labels)

if __name__ == "__main__":
    main()
