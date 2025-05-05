import os
import numpy as np
import pandas as pd
import cv2
from skimage.feature import graycomatrix, graycoprops
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

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
    distances=[1],
    angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
    levels=256,
):
    """
    Extract GLCM (Gray-Level Co-occurrence Matrix) features from a grayscale image

    Args:
        gray_image (numpy.ndarray): Grayscale image
        distances (list): List of distances for GLCM
        angles (list): List of angles for GLCM
        levels (int): Number of gray levels

    Returns:
        numpy.ndarray: GLCM features
    """
    # Normalize the image to reduce the number of intensity values
    gray_image = (gray_image / 16).astype(np.uint8)
    levels = 16  # Reduce levels for faster computation

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


def train_knn_model(features, labels, n_neighbors=5):
    """
    Train a KNN model

    Args:
        features (numpy.ndarray): Features for training
        labels (numpy.ndarray): Labels for training
        n_neighbors (int): Number of neighbors for KNN

    Returns:
        tuple: (model, scaler, accuracy)
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train KNN model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train_scaled, y_train)

    # Evaluate the model
    accuracy = knn.score(X_test_scaled, y_test)

    return knn, scaler, accuracy


def predict_image(model, scaler, image_path):
    """
    Predict the class of an image

    Args:
        model: Trained model
        scaler: Fitted scaler
        image_path (str): Path to the image

    Returns:
        tuple: (predicted_class, probabilities)
    """
    # Preprocess image
    _, gray_image = preprocess_image(image_path)

    # Extract GLCM features
    glcm_features = extract_glcm_features(gray_image)

    # Scale features
    features_scaled = scaler.transform(glcm_features.reshape(1, -1))

    # Predict
    predicted_class = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]

    return predicted_class, probabilities
