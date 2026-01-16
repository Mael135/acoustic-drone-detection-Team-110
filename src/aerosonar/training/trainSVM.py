import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# change 1
def train_drone_svm(spectrograms, labels):
    """
    spectrograms: A list or 3D numpy array of shape (N, height, width)
    labels: A 1D array of integers (1 for drone, 0 for non-drone)
    """

    # 1. Reshaping (Flattening)
    # SVMs cannot take 2D images directly. We flatten (height * width) into a single vector.
    num_samples = len(spectrograms)
    data_reshaped = np.array(spectrograms).reshape(num_samples, -1)

    print(f"Original shape: {np.shape(spectrograms)}")
    print(f"Flattened shape for SVM: {data_reshaped.shape}")

    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        data_reshaped, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # 3. Feature Scaling
    # Essential for SVM: it ensures features with larger magnitudes don't dominate
    # the distance calculations.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Model Definition
    # Using 'rbf' (Radial Basis Function) to handle non-linear relationships in the frequency data.
    model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)

    # 5. Training
    print("Training SVM model...")
    model.fit(X_train_scaled, y_train)

    # 6. Evaluation
    y_pred = model.predict(X_test_scaled)

    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=['Non-Drone', 'Drone']))

    return model, scaler

# --- Example Usage with Dummy Data ---
# Assume 100 samples of 128x128 spectrograms
# mock_specs = np.random.rand(100, 128, 128)
# mock_labels = np.random.randint(0, 2, 100)
# trained_model, fitted_scaler = train_drone_svm(mock_specs, mock_labels)