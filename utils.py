import os
import numpy as np
import pandas as pd
import cv2
from skimage.feature import graycomatrix, graycoprops
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Constants
CLASSES = ["Angora", "British Shorthair", "Persian"]
IMG_SIZE = (128, 128)  # Resize images to this size for consistency


def load_dataset(dataset_path, split="train_cleaned"):
    """
    Load dataset from the specified path and split

    Args:
        dataset_path (str): Path to the dataset directory
        split (str): Split to load (train_cleaned, test_cleaned, valid_cleaned)

    Returns:
        tuple: (image_paths, labels)
    """
    split_dir = os.path.join(dataset_path, split)
    csv_path = os.path.join(split_dir, "_classes.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")

    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Get image paths and labels
    image_paths = [os.path.join(split_dir, filename) for filename in df["filename"]]

    # Convert one-hot encoded labels to class indices
    labels = []
    for _, row in df.iterrows():
        for i, cls in enumerate(CLASSES):
            if row[cls] == 1:
                labels.append(i)
                break

    return image_paths, np.array(labels)


def load_all_splits(dataset_path):
    """
    Load all dataset splits (train, validation, test)

    Args:
        dataset_path (str): Path to the dataset directory

    Returns:
        dict: Dictionary containing image paths and labels for each split
    """
    splits = ["train_cleaned", "valid_cleaned", "test_cleaned"]
    dataset = {}

    for split in splits:
        try:
            image_paths, labels = load_dataset(dataset_path, split)
            dataset[split] = {"image_paths": image_paths, "labels": labels}
            print(f"Loaded {split}: {len(image_paths)} images")
        except FileNotFoundError as e:
            print(f"Warning: {e}")

    return dataset


def preprocess_image(image_path, target_size=IMG_SIZE):
    """
    Load and preprocess an image

    Args:
        image_path (str): Path to the image
        target_size (tuple): Target size for resizing

    Returns:
        tuple: (original_image, grayscale_image)
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Convert from BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize the image
    img_resized = cv2.resize(img_rgb, target_size)

    # Convert to grayscale
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)

    return img_resized, img_gray


def extract_glcm_features(
    gray_image,
    distances=[1, 2, 4],  # Added more distances for better texture analysis
    angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
    levels=16,  # Reduced default levels for faster computation
    quantization_method="uniform",  # Added quantization method parameter
):
    """
    Extract GLCM (Gray-Level Co-occurrence Matrix) features from a grayscale image

    Args:
        gray_image (numpy.ndarray): Grayscale image
        distances (list): List of distances for GLCM
        angles (list): List of angles for GLCM
        levels (int): Number of gray levels
        quantization_method (str): Method for quantizing the image ('uniform' or 'equal')

    Returns:
        numpy.ndarray: GLCM features
    """
    # Start timing
    start_time = time.time()

    # Normalize the image to reduce the number of intensity values
    if quantization_method == "uniform":
        # Simple uniform quantization
        gray_image = (gray_image / (256 / levels)).astype(np.uint8)
    elif quantization_method == "equal":
        # Histogram equalization followed by quantization
        hist, bins = np.histogram(gray_image.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * levels / cdf[-1]
        gray_image = np.interp(
            gray_image.flatten(), range(256), cdf_normalized
        ).reshape(gray_image.shape)
        gray_image = gray_image.astype(np.uint8)

    # Calculate GLCM
    glcm = graycomatrix(
        gray_image,
        distances=distances,
        angles=angles,
        levels=levels,
        symmetric=True,
        normed=True,
    )

    # Calculate GLCM properties
    properties = [
        "contrast",
        "dissimilarity",
        "homogeneity",
        "energy",
        "correlation",
        "ASM",
    ]
    features = []

    for prop in properties:
        feature = graycoprops(glcm, prop).flatten()
        features.extend(feature)

    # Print timing information for debugging
    # print(f"GLCM extraction took {time.time() - start_time:.4f} seconds")

    return np.array(features)


def extract_features_from_dataset(image_paths, progress_callback=None):
    """
    Extract features from all images in the dataset

    Args:
        image_paths (list): List of image paths
        progress_callback (callable): Callback function for progress updates

    Returns:
        numpy.ndarray: Features for all images
    """
    features = []

    total_images = len(image_paths)
    for i, image_path in enumerate(image_paths):
        try:
            # Update progress
            if progress_callback:
                progress_callback((i + 1) / total_images)

            # Preprocess image
            _, gray_image = preprocess_image(image_path)

            # Extract GLCM features
            glcm_features = extract_glcm_features(gray_image)

            features.append(glcm_features)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    return np.array(features)


def visualize_tsne(features, labels, perplexity=30, n_iter=1000):
    """
    Visualize features using t-SNE

    Args:
        features (numpy.ndarray): Features to visualize
        labels (numpy.ndarray): Labels for coloring
        perplexity (int): Perplexity parameter for t-SNE
        n_iter (int): Number of iterations for t-SNE

    Returns:
        matplotlib.figure.Figure: Figure with t-SNE visualization
    """
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    features_tsne = tsne.fit_transform(features_scaled)

    # Create a DataFrame for easier plotting
    df_tsne = pd.DataFrame(
        {
            "x": features_tsne[:, 0],
            "y": features_tsne[:, 1],
            "label": [CLASSES[label] for label in labels],
        }
    )

    # Create the plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df_tsne, x="x", y="y", hue="label", palette="viridis")
    plt.title("t-SNE Visualization of GLCM Features")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")

    return plt.gcf()


def train_knn_model(features, labels, n_neighbors=5, use_grid_search=False):
    """
    Train a KNN model with optional hyperparameter tuning

    Args:
        features (numpy.ndarray): Features for training
        labels (numpy.ndarray): Labels for training
        n_neighbors (int): Number of neighbors for KNN (used if use_grid_search=False)
        use_grid_search (bool): Whether to use GridSearchCV for hyperparameter tuning

    Returns:
        tuple: (model, scaler, accuracy, report)
    """
    # Split the data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        features, labels, test_size=0.3, random_state=42, stratify=labels
    )

    # Further split the temp set into validation and test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # Print split sizes for debugging
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Train KNN model
    if use_grid_search:
        # Define parameter grid
        param_grid = {
            "n_neighbors": [3, 5, 7, 9, 11],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan", "minkowski"],
        }

        # Create and fit GridSearchCV
        start_time = time.time()
        print("Starting GridSearchCV...")

        grid_search = GridSearchCV(
            KNeighborsClassifier(),
            param_grid,
            cv=5,
            scoring="accuracy",
            n_jobs=-1,  # Use all available cores
        )

        grid_search.fit(X_train_scaled, y_train)

        # Get best parameters and model
        best_params = grid_search.best_params_
        print(f"Best parameters: {best_params}")
        print(f"GridSearchCV took {time.time() - start_time:.2f} seconds")

        # Train final model with best parameters
        knn = KNeighborsClassifier(**best_params)
    else:
        # Use specified n_neighbors
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Fit the model on training data
    knn.fit(X_train_scaled, y_train)

    # Evaluate on validation set
    val_accuracy = knn.score(X_val_scaled, y_val)
    print(f"Validation accuracy: {val_accuracy:.4f}")

    # Evaluate on test set
    test_accuracy = knn.score(X_test_scaled, y_test)
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Generate detailed report
    y_pred = knn.predict(X_test_scaled)
    report = classification_report(y_test, y_pred, target_names=CLASSES)

    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Return model, scaler, accuracy, and evaluation metrics
    return (
        knn,
        scaler,
        test_accuracy,
        {
            "report": report,
            "confusion_matrix": cm,
            "validation_accuracy": val_accuracy,
            "test_accuracy": test_accuracy,
        },
    )


def predict_image(model, scaler, image_path, glcm_params=None):
    """
    Predict the class of an image with confidence scores

    Args:
        model: Trained model
        scaler: Fitted scaler
        image_path (str): Path to the image
        glcm_params (dict, optional): Parameters for GLCM feature extraction

    Returns:
        tuple: (predicted_class, probabilities, confidence_score)
    """
    # Preprocess image
    _, gray_image = preprocess_image(image_path)

    # Extract GLCM features with custom parameters if provided
    if glcm_params is None:
        glcm_features = extract_glcm_features(gray_image)
    else:
        glcm_features = extract_glcm_features(
            gray_image,
            distances=glcm_params.get("distances", [1, 2, 4]),
            angles=glcm_params.get("angles", [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]),
            levels=glcm_params.get("levels", 16),
            quantization_method=glcm_params.get("quantization_method", "uniform"),
        )

    # Scale features
    features_scaled = scaler.transform(glcm_features.reshape(1, -1))

    # Predict
    predicted_class = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]

    # Calculate confidence score (difference between top two probabilities)
    sorted_probs = np.sort(probabilities)[::-1]
    if len(sorted_probs) > 1:
        confidence_score = sorted_probs[0] - sorted_probs[1]
    else:
        confidence_score = sorted_probs[0]

    # Get nearest neighbors and their distances
    distances, indices = model.kneighbors(features_scaled)

    # Return prediction results
    return (
        predicted_class,
        probabilities,
        {
            "confidence_score": confidence_score,
            "nearest_neighbors": {"indices": indices[0], "distances": distances[0]},
        },
    )
