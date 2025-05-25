from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def train_svm(features, labels):
    print("[INFO] Encoding labels...")
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)

    print("[INFO] Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        features, encoded_labels, test_size=0.2, random_state=42
    )

    print("[INFO] Training SVM classifier...")
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)

    print("[INFO] Evaluating model...")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model, encoder
