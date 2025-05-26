import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

def train_svm(features, labels, name_prefix):
    """
    Trains an SVM classifier with the provided data, evaluates its performance, 
    and saves the trained model along with the label encoder and feature scaler.

    Parameters:
    ----------
    features: Set of features extracted from the input images or data.

    labels: Class labels corresponding to each data sample.

    name_prefix: Prefix to use to name the saved model, encoder, and scaler files.

    Returns:
    -------
    model: Trained SVM model.

    encoder: Label encoder used to transform text labels into numeric values.

    scaler: Feature scaler used to normalize the data before training.
    """

    print(f"[INFO] Encoding labels for {name_prefix}...")
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)

    print(f"[INFO] Scaling features for {name_prefix}...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    print(f"[INFO] Splitting data for {name_prefix}... (80% train - 20% test)")
    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, encoded_labels, test_size=0.2, random_state=42
    )

    print(f"[INFO] Training SVM classifier for {name_prefix}...")
    model = SVC(probability=True)
    model.fit(X_train, y_train)

    print(f"[INFO] Evaluating model for {name_prefix}...")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

    os.makedirs('models', exist_ok=True)

    joblib.dump(model, f"models/svm_{name_prefix}_model.pkl")
    joblib.dump(encoder, f"models/{name_prefix}_encoder.pkl")
    joblib.dump(scaler, f"models/{name_prefix}_scaler.pkl")

    print(f"[INFO] Model for {name_prefix} saved.\n")

    return model, encoder, scaler
