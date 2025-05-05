import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from utils import (
    load_dataset,
    load_all_splits,
    preprocess_image,
    extract_glcm_features,
    extract_features_from_dataset,
    visualize_tsne,
    train_knn_model,
    predict_image,
    CLASSES,
)

# Set page configuration
st.set_page_config(page_title="Cat Breed Classification", page_icon="🐱", layout="wide")

# Title and description
st.title("Cat Breed Classification using GLCM and KNN")
st.markdown(
    """
This application demonstrates cat breed classification using:
- Gray-Level Co-occurrence Matrix (GLCM) for feature extraction
- K-Nearest Neighbors (KNN) for classification
- t-SNE for feature visualization
"""
)

# Initialize session state variables
if "dataset_loaded" not in st.session_state:
    st.session_state.dataset_loaded = False
if "features_extracted" not in st.session_state:
    st.session_state.features_extracted = False
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False
if "image_paths" not in st.session_state:
    st.session_state.image_paths = None
if "labels" not in st.session_state:
    st.session_state.labels = None
if "features" not in st.session_state:
    st.session_state.features = None
if "model" not in st.session_state:
    st.session_state.model = None
if "scaler" not in st.session_state:
    st.session_state.scaler = None
if "accuracy" not in st.session_state:
    st.session_state.accuracy = None
if "dataset_splits" not in st.session_state:
    st.session_state.dataset_splits = None
if "evaluation_metrics" not in st.session_state:
    st.session_state.evaluation_metrics = None
if "glcm_params" not in st.session_state:
    st.session_state.glcm_params = {
        "distances": [1, 2, 4],
        "angles": [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        "levels": 16,
        "quantization_method": "uniform",
    }

# Sidebar with steps
st.sidebar.title("Steps")
steps = [
    "1. Load Dataset",
    "2. Preprocessing Gambar ke Grayscale",
    "3. Ekstraksi Fitur GLCM",
    "4. Visualisasi t-SNE",
    "5. Pelatihan Model KNN",
    "6. Prediksi Gambar",
]

selected_step = st.sidebar.radio("Select a step:", steps)

# Step 1: Load Dataset
if selected_step == steps[0]:
    st.header("1. Load Dataset")

    dataset_path = st.text_input("Dataset Path", value="dataset")

    load_method = st.radio(
        "Loading Method",
        ["Load Single Split", "Load All Splits (Recommended)"],
        index=1,
    )

    if load_method == "Load Single Split":
        split_options = ["train_cleaned", "test_cleaned", "valid_cleaned"]
        selected_split = st.selectbox("Select Split", split_options)

        if st.button("Load Single Split"):
            with st.spinner("Loading dataset..."):
                try:
                    image_paths, labels = load_dataset(dataset_path, selected_split)

                    st.session_state.image_paths = image_paths
                    st.session_state.labels = labels
                    st.session_state.dataset_loaded = True

                    # Display dataset information
                    st.success(
                        f"Dataset loaded successfully! Found {len(image_paths)} images."
                    )

                    # Display class distribution
                    class_counts = np.bincount(labels)
                    class_df = pd.DataFrame({"Class": CLASSES, "Count": class_counts})

                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Class Distribution")
                        st.dataframe(class_df)

                    with col2:
                        st.subheader("Class Distribution Visualization")
                        fig, ax = plt.subplots()
                        ax.bar(CLASSES, class_counts)
                        ax.set_ylabel("Count")
                        ax.set_title("Class Distribution")
                        st.pyplot(fig)

                    # Display sample images
                    st.subheader("Sample Images")
                    sample_indices = np.random.choice(
                        len(image_paths), min(5, len(image_paths)), replace=False
                    )

                    cols = st.columns(len(sample_indices))
                    for i, idx in enumerate(sample_indices):
                        with cols[i]:
                            img = Image.open(image_paths[idx])
                            st.image(
                                img,
                                caption=f"Class: {CLASSES[labels[idx]]}",
                                use_column_width=True,
                            )

                except Exception as e:
                    st.error(f"Error loading dataset: {e}")
    else:
        if st.button("Load All Splits"):
            with st.spinner("Loading all dataset splits..."):
                try:
                    # Load all splits
                    dataset_splits = load_all_splits(dataset_path)

                    if not dataset_splits:
                        st.error("No dataset splits found!")
                    else:
                        # Store in session state
                        st.session_state.dataset_splits = dataset_splits

                        # For backward compatibility, use train split as the default
                        if "train_cleaned" in dataset_splits:
                            st.session_state.image_paths = dataset_splits[
                                "train_cleaned"
                            ]["image_paths"]
                            st.session_state.labels = dataset_splits["train_cleaned"][
                                "labels"
                            ]
                        else:
                            # Use the first available split
                            first_split = list(dataset_splits.keys())[0]
                            st.session_state.image_paths = dataset_splits[first_split][
                                "image_paths"
                            ]
                            st.session_state.labels = dataset_splits[first_split][
                                "labels"
                            ]

                        st.session_state.dataset_loaded = True

                        # Display dataset information
                        st.success("All dataset splits loaded successfully!")

                        # Create tabs for each split
                        tabs = st.tabs(list(dataset_splits.keys()))

                        for i, (split_name, split_data) in enumerate(
                            dataset_splits.items()
                        ):
                            with tabs[i]:
                                st.subheader(f"{split_name} Split")
                                st.write(
                                    f"Number of images: {len(split_data['image_paths'])}"
                                )

                                # Display class distribution
                                class_counts = np.bincount(split_data["labels"])
                                class_df = pd.DataFrame(
                                    {"Class": CLASSES, "Count": class_counts}
                                )

                                col1, col2 = st.columns(2)

                                with col1:
                                    st.subheader("Class Distribution")
                                    st.dataframe(class_df)

                                with col2:
                                    st.subheader("Class Distribution Visualization")
                                    fig, ax = plt.subplots()
                                    ax.bar(CLASSES, class_counts)
                                    ax.set_ylabel("Count")
                                    ax.set_title(f"Class Distribution - {split_name}")
                                    st.pyplot(fig)

                                # Display sample images
                                st.subheader("Sample Images")
                                sample_indices = np.random.choice(
                                    len(split_data["image_paths"]),
                                    min(5, len(split_data["image_paths"])),
                                    replace=False,
                                )

                                cols = st.columns(len(sample_indices))
                                for j, idx in enumerate(sample_indices):
                                    with cols[j]:
                                        img = Image.open(split_data["image_paths"][idx])
                                        st.image(
                                            img,
                                            caption=f"Class: {CLASSES[split_data['labels'][idx]]}",
                                            use_column_width=True,
                                        )

                except Exception as e:
                    st.error(f"Error loading dataset splits: {e}")
                    st.exception(e)

    if st.session_state.dataset_loaded:
        st.success("✅ Dataset already loaded!")

        # Add option to clear loaded dataset
        if st.button("Clear Loaded Dataset"):
            st.session_state.dataset_loaded = False
            st.session_state.features_extracted = False
            st.session_state.model_trained = False
            st.session_state.image_paths = None
            st.session_state.labels = None
            st.session_state.features = None
            st.session_state.model = None
            st.session_state.scaler = None
            st.session_state.accuracy = None
            st.session_state.dataset_splits = None
            st.session_state.evaluation_metrics = None
            st.experimental_rerun()

# Step 2: Preprocessing Images to Grayscale
elif selected_step == steps[1]:
    st.header("2. Preprocessing Gambar ke Grayscale")

    if not st.session_state.dataset_loaded:
        st.warning("Please load the dataset first (Step 1).")
    else:
        st.info(
            "This step demonstrates how images are converted to grayscale for feature extraction."
        )

        # Select a random image to demonstrate preprocessing
        sample_idx = np.random.choice(len(st.session_state.image_paths))
        sample_path = st.session_state.image_paths[sample_idx]
        sample_label = CLASSES[st.session_state.labels[sample_idx]]

        # Display original and preprocessed images
        original_img, gray_img = preprocess_image(sample_path)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(
                original_img, caption=f"Class: {sample_label}", use_column_width=True
            )

        with col2:
            st.subheader("Grayscale Image")
            st.image(gray_img, caption="Grayscale", use_column_width=True)

        st.markdown(
            """
        ### Why convert to grayscale?

        - GLCM features are extracted from grayscale images
        - Reduces computational complexity
        - Focuses on texture patterns rather than color

        ### Preprocessing steps:
        1. Load the image
        2. Resize to a consistent size (128x128 pixels)
        3. Convert to grayscale
        """
        )

# Step 3: GLCM Feature Extraction
elif selected_step == steps[2]:
    st.header("3. Ekstraksi Fitur GLCM")

    if not st.session_state.dataset_loaded:
        st.warning("Please load the dataset first (Step 1).")
    else:
        st.info(
            "This step extracts GLCM (Gray-Level Co-occurrence Matrix) features from the images."
        )

        # Explanation of GLCM
        st.markdown(
            """
        ### What is GLCM?

        GLCM (Gray-Level Co-occurrence Matrix) is a statistical method for examining texture by considering the spatial relationship of pixels. It calculates how often pairs of pixels with specific values and in a specified spatial relationship occur in an image.

        ### GLCM Features:
        - **Contrast**: Measures local variations in the GLCM
        - **Dissimilarity**: Similar to contrast but increases linearly
        - **Homogeneity**: Measures the closeness of the distribution of elements in the GLCM to the GLCM diagonal
        - **Energy**: Sum of squared elements in the GLCM
        - **Correlation**: Measures how correlated a pixel is to its neighbor
        - **ASM (Angular Second Moment)**: Measures the uniformity of an image
        """
        )

        # GLCM Parameters
        st.subheader("GLCM Parameters")

        # Create tabs for parameters and preview
        param_tab, preview_tab = st.tabs(["Parameters", "Feature Preview"])

        with param_tab:
            col1, col2 = st.columns(2)

            with col1:
                # Distance options
                distance_options = [[1], [1, 2], [1, 2, 4], [1, 3, 5, 7]]
                selected_distance_idx = st.selectbox(
                    "Select Distances",
                    range(len(distance_options)),
                    format_func=lambda x: str(distance_options[x]),
                )
                selected_distances = distance_options[selected_distance_idx]

                # Quantization levels
                levels = st.slider(
                    "Quantization Levels", min_value=8, max_value=64, value=16, step=8
                )

            with col2:
                # Quantization method
                quantization_method = st.selectbox(
                    "Quantization Method",
                    ["uniform", "equal"],
                    help="Uniform: Simple division, Equal: Histogram equalization",
                )

                # Angle options (in degrees for display)
                angle_options = [
                    [0],
                    [0, 90],
                    [0, 45, 90, 135],
                    [0, 30, 60, 90, 120, 150],
                ]
                selected_angle_idx = st.selectbox(
                    "Select Angles (degrees)",
                    range(len(angle_options)),
                    format_func=lambda x: str([f"{a}°" for a in angle_options[x]]),
                )
                # Convert to radians for processing
                selected_angles = [
                    a * np.pi / 180 for a in angle_options[selected_angle_idx]
                ]

            # Save parameters to session state
            st.session_state.glcm_params = {
                "distances": selected_distances,
                "angles": selected_angles,
                "levels": levels,
                "quantization_method": quantization_method,
            }

            st.info(
                "These parameters will be used for feature extraction and model training."
            )

        with preview_tab:
            # Select a random image to demonstrate GLCM
            sample_idx = np.random.choice(len(st.session_state.image_paths))
            sample_path = st.session_state.image_paths[sample_idx]

            # Display original and grayscale images
            original_img, gray_img = preprocess_image(sample_path)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Grayscale Image")
                st.image(gray_img, caption="Grayscale", use_column_width=True)

            with col2:
                st.subheader("GLCM Features")
                # Extract GLCM features with current parameters
                glcm_features = extract_glcm_features(
                    gray_img,
                    distances=st.session_state.glcm_params["distances"],
                    angles=st.session_state.glcm_params["angles"],
                    levels=st.session_state.glcm_params["levels"],
                    quantization_method=st.session_state.glcm_params[
                        "quantization_method"
                    ],
                )

                # Create a DataFrame for better display
                properties = [
                    "contrast",
                    "dissimilarity",
                    "homogeneity",
                    "energy",
                    "correlation",
                    "ASM",
                ]

                # Convert angles back to degrees for display
                angles_deg = [
                    int(a * 180 / np.pi) for a in st.session_state.glcm_params["angles"]
                ]

                # Calculate the shape based on the number of properties and angles
                num_angles = len(st.session_state.glcm_params["angles"])
                num_distances = len(st.session_state.glcm_params["distances"])

                # Reshape features for display
                feature_matrix = glcm_features.reshape(
                    len(properties), num_angles * num_distances
                )

                # Create column labels
                col_labels = []
                for d in st.session_state.glcm_params["distances"]:
                    for a in angles_deg:
                        col_labels.append(f"d={d},θ={a}°")

                feature_df = pd.DataFrame(
                    feature_matrix, index=properties, columns=col_labels
                )

                st.dataframe(feature_df)

        # Extract features from all images
        if st.button("Extract GLCM Features from All Images"):
            with st.spinner("Extracting GLCM features from all images..."):
                # Create a progress bar
                progress_bar = st.progress(0)

                # Define a callback for updating the progress bar
                def update_progress(progress):
                    progress_bar.progress(progress)

                # Start timing
                start_time = time.time()

                # Define a custom feature extraction function that uses the current parameters
                def extract_features_with_params(image_path):
                    _, gray_image = preprocess_image(image_path)
                    return extract_glcm_features(
                        gray_image,
                        distances=st.session_state.glcm_params["distances"],
                        angles=st.session_state.glcm_params["angles"],
                        levels=st.session_state.glcm_params["levels"],
                        quantization_method=st.session_state.glcm_params[
                            "quantization_method"
                        ],
                    )

                # Extract features from all images
                features = []
                total_images = len(st.session_state.image_paths)

                for i, image_path in enumerate(st.session_state.image_paths):
                    try:
                        # Update progress
                        update_progress((i + 1) / total_images)

                        # Extract features
                        glcm_features = extract_features_with_params(image_path)
                        features.append(glcm_features)
                    except Exception as e:
                        st.error(f"Error processing {image_path}: {e}")

                features = np.array(features)

                # Store features in session state
                st.session_state.features = features
                st.session_state.features_extracted = True

                # Display timing information
                elapsed_time = time.time() - start_time
                st.success(f"Features extracted successfully! Shape: {features.shape}")
                st.info(
                    f"Extraction took {elapsed_time:.2f} seconds ({elapsed_time/total_images:.4f} seconds per image)"
                )

                # Display feature statistics
                st.subheader("Feature Statistics")
                feature_stats = pd.DataFrame(
                    {
                        "Min": np.min(features, axis=0),
                        "Max": np.max(features, axis=0),
                        "Mean": np.mean(features, axis=0),
                        "Std": np.std(features, axis=0),
                    }
                )
                st.dataframe(feature_stats)

# Step 4: t-SNE Visualization
elif selected_step == steps[3]:
    st.header("4. Visualisasi t-SNE")

    if not st.session_state.dataset_loaded:
        st.warning("Please load the dataset first (Step 1).")
    elif not st.session_state.features_extracted:
        st.warning("Please extract GLCM features first (Step 3).")
    else:
        st.info(
            "This step visualizes the extracted features using t-SNE (t-Distributed Stochastic Neighbor Embedding)."
        )

        # Explanation of t-SNE
        st.markdown(
            """
        ### What is t-SNE?

        t-SNE (t-Distributed Stochastic Neighbor Embedding) is a machine learning algorithm for dimensionality reduction. It is particularly well-suited for visualizing high-dimensional data in 2D or 3D space.

        ### How t-SNE works:
        - Converts similarities between data points to joint probabilities
        - Creates a similar probability distribution in the low-dimensional space
        - Minimizes the KL divergence between the two distributions
        """
        )

        # t-SNE parameters
        col1, col2 = st.columns(2)

        with col1:
            perplexity = st.slider(
                "Perplexity", min_value=5, max_value=50, value=30, step=5
            )

        with col2:
            n_iter = st.slider(
                "Number of Iterations",
                min_value=250,
                max_value=2000,
                value=1000,
                step=250,
            )

        if st.button("Generate t-SNE Visualization"):
            with st.spinner("Generating t-SNE visualization..."):
                # Visualize using t-SNE
                fig = visualize_tsne(
                    st.session_state.features,
                    st.session_state.labels,
                    perplexity=perplexity,
                    n_iter=n_iter,
                )

                st.pyplot(fig)

                st.success("t-SNE visualization generated successfully!")

                st.markdown(
                    """
                ### Interpreting the t-SNE Visualization:

                - Points that are close together in the visualization are similar in the high-dimensional space
                - Clusters indicate groups of similar images
                - The distance between clusters indicates how different they are

                Note: t-SNE focuses on preserving local structure, so the global structure may not be accurately represented.
                """
                )

# Step 5: KNN Model Training
elif selected_step == steps[4]:
    st.header("5. Pelatihan Model KNN")

    if not st.session_state.dataset_loaded:
        st.warning("Please load the dataset first (Step 1).")
    elif not st.session_state.features_extracted:
        st.warning("Please extract GLCM features first (Step 3).")
    else:
        st.info(
            "This step trains a K-Nearest Neighbors (KNN) model on the extracted features."
        )

        # Explanation of KNN
        st.markdown(
            """
        ### What is KNN?

        K-Nearest Neighbors (KNN) is a simple, instance-based learning algorithm used for classification and regression. It works by finding the k nearest training examples in the feature space and using their classes to predict the class of a new instance.

        ### How KNN works:
        - Calculate the distance between the new instance and all training instances
        - Select the k nearest instances
        - Assign the most common class among these k instances to the new instance
        """
        )

        # Create tabs for basic and advanced training
        basic_tab, advanced_tab = st.tabs(
            ["Basic Training", "Advanced Training (GridSearchCV)"]
        )

        with basic_tab:
            # KNN parameters
            n_neighbors = st.slider(
                "Number of Neighbors (K)", min_value=1, max_value=20, value=5, step=2
            )

            if st.button("Train KNN Model (Basic)"):
                with st.spinner("Training KNN model..."):
                    # Train KNN model
                    model, scaler, accuracy, metrics = train_knn_model(
                        st.session_state.features,
                        st.session_state.labels,
                        n_neighbors=n_neighbors,
                        use_grid_search=False,
                    )

                    st.session_state.model = model
                    st.session_state.scaler = scaler
                    st.session_state.accuracy = accuracy
                    st.session_state.evaluation_metrics = metrics
                    st.session_state.model_trained = True

                    st.success(
                        f"Model trained successfully! Test Accuracy: {accuracy:.2%}"
                    )

                    # Display validation and test accuracy
                    st.info(
                        f"Validation Accuracy: {metrics['validation_accuracy']:.2%}"
                    )
                    st.info(f"Test Accuracy: {metrics['test_accuracy']:.2%}")

                    # Display confusion matrix
                    cm = metrics["confusion_matrix"]

                    # Plot confusion matrix
                    fig, ax = plt.subplots(figsize=(8, 6))
                    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
                    ax.figure.colorbar(im, ax=ax)
                    ax.set(
                        xticks=np.arange(cm.shape[1]),
                        yticks=np.arange(cm.shape[0]),
                        xticklabels=CLASSES,
                        yticklabels=CLASSES,
                        title="Confusion Matrix (Test Set)",
                        ylabel="True label",
                        xlabel="Predicted label",
                    )

                    # Rotate the tick labels and set their alignment
                    plt.setp(
                        ax.get_xticklabels(),
                        rotation=45,
                        ha="right",
                        rotation_mode="anchor",
                    )

                    # Loop over data dimensions and create text annotations
                    thresh = cm.max() / 2.0
                    for i in range(cm.shape[0]):
                        for j in range(cm.shape[1]):
                            ax.text(
                                j,
                                i,
                                format(cm[i, j], "d"),
                                ha="center",
                                va="center",
                                color="white" if cm[i, j] > thresh else "black",
                            )

                    fig.tight_layout()
                    st.pyplot(fig)

                    # Display classification report
                    st.text("Classification Report (Test Set):")
                    st.text(metrics["report"])

        with advanced_tab:
            st.warning("Grid search can take a long time to complete. Be patient!")

            # Create columns for grid search parameters
            col1, col2 = st.columns(2)

            with col1:
                # K values to try
                k_min = st.number_input("Minimum K", min_value=1, max_value=10, value=3)
                k_max = st.number_input(
                    "Maximum K", min_value=k_min, max_value=20, value=11
                )
                k_step = st.number_input("K Step", min_value=1, max_value=5, value=2)

                # Calculate k values
                k_values = list(range(k_min, k_max + 1, k_step))
                st.write(f"K values to try: {k_values}")

            with col2:
                # Weight options
                weight_options = st.multiselect(
                    "Weight Functions",
                    ["uniform", "distance"],
                    default=["uniform", "distance"],
                )

                # Metric options
                metric_options = st.multiselect(
                    "Distance Metrics",
                    ["euclidean", "manhattan", "minkowski", "chebyshev"],
                    default=["euclidean", "manhattan"],
                )

            if st.button("Train KNN Model with Grid Search"):
                with st.spinner("Running Grid Search for optimal parameters..."):
                    # Override the default grid search parameters
                    param_grid = {
                        "n_neighbors": k_values,
                        "weights": weight_options,
                        "metric": metric_options,
                    }

                    # Create a custom train function that uses our param_grid
                    def train_with_custom_grid(features, labels):
                        # Split the data into train, validation, and test sets
                        X_train, X_temp, y_train, y_temp = train_test_split(
                            features,
                            labels,
                            test_size=0.3,
                            random_state=42,
                            stratify=labels,
                        )

                        # Further split the temp set into validation and test
                        X_val, X_test, y_val, y_test = train_test_split(
                            X_temp,
                            y_temp,
                            test_size=0.5,
                            random_state=42,
                            stratify=y_temp,
                        )

                        # Print split sizes for debugging
                        st.write(f"Training set: {X_train.shape[0]} samples")
                        st.write(f"Validation set: {X_val.shape[0]} samples")
                        st.write(f"Test set: {X_test.shape[0]} samples")

                        # Standardize features
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_val_scaled = scaler.transform(X_val)
                        X_test_scaled = scaler.transform(X_test)

                        # Create and fit GridSearchCV
                        start_time = time.time()
                        st.write("Starting GridSearchCV...")

                        from sklearn.model_selection import GridSearchCV
                        from sklearn.neighbors import KNeighborsClassifier

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
                        st.write(f"Best parameters: {best_params}")
                        st.write(
                            f"GridSearchCV took {time.time() - start_time:.2f} seconds"
                        )

                        # Train final model with best parameters
                        knn = KNeighborsClassifier(**best_params)
                        knn.fit(X_train_scaled, y_train)

                        # Evaluate on validation set
                        val_accuracy = knn.score(X_val_scaled, y_val)
                        st.write(f"Validation accuracy: {val_accuracy:.4f}")

                        # Evaluate on test set
                        test_accuracy = knn.score(X_test_scaled, y_test)
                        st.write(f"Test accuracy: {test_accuracy:.4f}")

                        # Generate detailed report
                        y_pred = knn.predict(X_test_scaled)
                        from sklearn.metrics import (
                            classification_report,
                            confusion_matrix,
                        )

                        report = classification_report(
                            y_test, y_pred, target_names=CLASSES
                        )

                        # Create confusion matrix
                        cm = confusion_matrix(y_test, y_pred)

                        return (
                            knn,
                            scaler,
                            test_accuracy,
                            {
                                "report": report,
                                "confusion_matrix": cm,
                                "validation_accuracy": val_accuracy,
                                "test_accuracy": test_accuracy,
                                "best_params": best_params,
                                "cv_results": grid_search.cv_results_,
                            },
                        )

                    # Train the model with grid search
                    model, scaler, accuracy, metrics = train_with_custom_grid(
                        st.session_state.features, st.session_state.labels
                    )

                    # Store in session state
                    st.session_state.model = model
                    st.session_state.scaler = scaler
                    st.session_state.accuracy = accuracy
                    st.session_state.evaluation_metrics = metrics
                    st.session_state.model_trained = True

                    # Display results
                    st.success(
                        f"Model trained successfully with Grid Search! Test Accuracy: {accuracy:.2%}"
                    )

                    # Display best parameters
                    st.subheader("Best Parameters")
                    st.json(metrics["best_params"])

                    # Display confusion matrix
                    st.subheader("Confusion Matrix (Test Set)")
                    cm = metrics["confusion_matrix"]

                    # Plot confusion matrix
                    fig, ax = plt.subplots(figsize=(8, 6))
                    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
                    ax.figure.colorbar(im, ax=ax)
                    ax.set(
                        xticks=np.arange(cm.shape[1]),
                        yticks=np.arange(cm.shape[0]),
                        xticklabels=CLASSES,
                        yticklabels=CLASSES,
                        title="Confusion Matrix (Test Set)",
                        ylabel="True label",
                        xlabel="Predicted label",
                    )

                    # Rotate the tick labels and set their alignment
                    plt.setp(
                        ax.get_xticklabels(),
                        rotation=45,
                        ha="right",
                        rotation_mode="anchor",
                    )

                    # Loop over data dimensions and create text annotations
                    thresh = cm.max() / 2.0
                    for i in range(cm.shape[0]):
                        for j in range(cm.shape[1]):
                            ax.text(
                                j,
                                i,
                                format(cm[i, j], "d"),
                                ha="center",
                                va="center",
                                color="white" if cm[i, j] > thresh else "black",
                            )

                    fig.tight_layout()
                    st.pyplot(fig)

                    # Display classification report
                    st.text("Classification Report (Test Set):")
                    st.text(metrics["report"])

                    # Display grid search results
                    st.subheader("Grid Search Results")

                    # Convert cv_results to DataFrame for better display
                    cv_results = pd.DataFrame(metrics["cv_results"])

                    # Select only the most important columns
                    important_cols = [
                        col
                        for col in cv_results.columns
                        if col.startswith("param_")
                        or col.startswith("mean_test_")
                        or col.startswith("std_test_")
                        or col.startswith("rank_")
                    ]

                    cv_results_display = cv_results[important_cols].sort_values(
                        "rank_test_score"
                    )

                    # Display top 10 results
                    st.dataframe(cv_results_display.head(10))

# Step 6: Image Prediction
elif selected_step == steps[5]:
    st.header("6. Prediksi Gambar")

    if not st.session_state.model_trained:
        st.warning("Please train the model first (Step 5).")
    else:
        st.info("This step allows you to upload an image and predict its cat breed.")

        # Upload image
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        # Option to use GLCM parameters from feature extraction
        use_saved_params = st.checkbox(
            "Use GLCM parameters from feature extraction", value=True
        )

        if not use_saved_params:
            st.subheader("Custom GLCM Parameters")

            col1, col2 = st.columns(2)

            with col1:
                # Distance options
                distance_options = [[1], [1, 2], [1, 2, 4], [1, 3, 5, 7]]
                selected_distance_idx = st.selectbox(
                    "Select Distances for Prediction",
                    range(len(distance_options)),
                    format_func=lambda x: str(distance_options[x]),
                )
                selected_distances = distance_options[selected_distance_idx]

                # Quantization levels
                levels = st.slider(
                    "Quantization Levels", min_value=8, max_value=64, value=16, step=8
                )

            with col2:
                # Quantization method
                quantization_method = st.selectbox(
                    "Quantization Method",
                    ["uniform", "equal"],
                    help="Uniform: Simple division, Equal: Histogram equalization",
                )

                # Angle options (in degrees for display)
                angle_options = [
                    [0],
                    [0, 90],
                    [0, 45, 90, 135],
                    [0, 30, 60, 90, 120, 150],
                ]
                selected_angle_idx = st.selectbox(
                    "Select Angles (degrees)",
                    range(len(angle_options)),
                    format_func=lambda x: str([f"{a}°" for a in angle_options[x]]),
                )
                # Convert to radians for processing
                selected_angles = [
                    a * np.pi / 180 for a in angle_options[selected_angle_idx]
                ]

            # Create custom GLCM parameters
            custom_glcm_params = {
                "distances": selected_distances,
                "angles": selected_angles,
                "levels": levels,
                "quantization_method": quantization_method,
            }

        if uploaded_file is not None:
            # Save the uploaded file temporarily
            temp_path = os.path.join("temp_upload.jpg")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Display the uploaded image
            st.image(temp_path, caption="Uploaded Image", use_column_width=True)

            if st.button("Predict"):
                with st.spinner("Predicting..."):
                    # Determine which GLCM parameters to use
                    glcm_params = (
                        st.session_state.glcm_params
                        if use_saved_params
                        else custom_glcm_params
                    )

                    # Predict the class with confidence scores
                    predicted_class, probabilities, confidence_info = predict_image(
                        st.session_state.model,
                        st.session_state.scaler,
                        temp_path,
                        glcm_params=glcm_params,
                    )

                    # Display the prediction with confidence
                    confidence_score = confidence_info["confidence_score"]
                    confidence_level = (
                        "High"
                        if confidence_score > 0.5
                        else "Medium" if confidence_score > 0.2 else "Low"
                    )

                    st.success(f"Predicted Breed: {CLASSES[predicted_class]}")
                    st.info(f"Confidence: {confidence_score:.2f} ({confidence_level})")

                    # Display probabilities
                    prob_df = pd.DataFrame(
                        {"Breed": CLASSES, "Probability": probabilities}
                    )

                    # Sort by probability
                    prob_df = prob_df.sort_values(
                        "Probability", ascending=False
                    ).reset_index(drop=True)

                    # Display as bar chart
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.barh(prob_df["Breed"], prob_df["Probability"])

                    # Color the bars based on prediction
                    for i, bar in enumerate(bars):
                        if prob_df.iloc[i]["Breed"] == CLASSES[predicted_class]:
                            bar.set_color("green")
                        else:
                            bar.set_color("gray")

                    ax.set_xlabel("Probability")
                    ax.set_title("Prediction Probabilities")
                    ax.invert_yaxis()  # Display highest probability at the top

                    st.pyplot(fig)

                    # Display nearest neighbors information
                    st.subheader("Nearest Neighbors")

                    # Get nearest neighbors info
                    nn_indices = confidence_info["nearest_neighbors"]["indices"]
                    nn_distances = confidence_info["nearest_neighbors"]["distances"]

                    # Create a DataFrame for nearest neighbors
                    nn_data = []

                    # Only show if we have training data in session state
                    if st.session_state.image_paths is not None and len(nn_indices) > 0:
                        for i, (idx, dist) in enumerate(zip(nn_indices, nn_distances)):
                            if idx < len(st.session_state.image_paths):
                                img_path = st.session_state.image_paths[idx]
                                label = CLASSES[st.session_state.labels[idx]]
                                nn_data.append(
                                    {
                                        "Index": idx,
                                        "Distance": dist,
                                        "Class": label,
                                        "Image Path": img_path,
                                    }
                                )

                        # Display nearest neighbors as a table
                        nn_df = pd.DataFrame(nn_data)
                        st.dataframe(nn_df)

                        # Display nearest neighbor images
                        st.subheader("Nearest Neighbor Images")

                        # Show up to 5 nearest neighbors
                        num_neighbors = min(5, len(nn_data))
                        cols = st.columns(num_neighbors)

                        for i in range(num_neighbors):
                            with cols[i]:
                                img_path = nn_data[i]["Image Path"]
                                img = Image.open(img_path)
                                st.image(
                                    img,
                                    caption=f"Class: {nn_data[i]['Class']}\nDistance: {nn_data[i]['Distance']:.4f}",
                                    use_column_width=True,
                                )

                    # Display preprocessing steps
                    st.subheader("Preprocessing Steps")

                    # Preprocess the image
                    original_img, gray_img = preprocess_image(temp_path)

                    col1, col2 = st.columns(2)

                    with col1:
                        st.image(
                            original_img,
                            caption="Original Image (Resized)",
                            use_column_width=True,
                        )

                    with col2:
                        st.image(
                            gray_img, caption="Grayscale Image", use_column_width=True
                        )

                    # Extract and display GLCM features with the selected parameters
                    glcm_features = extract_glcm_features(
                        gray_img,
                        distances=glcm_params["distances"],
                        angles=glcm_params["angles"],
                        levels=glcm_params["levels"],
                        quantization_method=glcm_params["quantization_method"],
                    )

                    # Create a DataFrame for better display
                    properties = [
                        "contrast",
                        "dissimilarity",
                        "homogeneity",
                        "energy",
                        "correlation",
                        "ASM",
                    ]

                    # Convert angles back to degrees for display
                    angles_deg = [int(a * 180 / np.pi) for a in glcm_params["angles"]]

                    # Calculate the shape based on the number of properties and angles
                    num_angles = len(glcm_params["angles"])
                    num_distances = len(glcm_params["distances"])

                    # Reshape features for display
                    feature_matrix = glcm_features.reshape(
                        len(properties), num_angles * num_distances
                    )

                    # Create column labels
                    col_labels = []
                    for d in glcm_params["distances"]:
                        for a in angles_deg:
                            col_labels.append(f"d={d},θ={a}°")

                    feature_df = pd.DataFrame(
                        feature_matrix, index=properties, columns=col_labels
                    )

                    st.subheader("GLCM Features")
                    st.dataframe(feature_df)

            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)

# Footer
st.sidebar.markdown("---")
st.sidebar.info(
    "This application demonstrates cat breed classification using GLCM features and KNN. "
    "Follow the steps in order to load the dataset, extract features, visualize the data, "
    "train a model, and make predictions."
)

# Advanced options in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Advanced Options")

# Add caching option
enable_caching = st.sidebar.checkbox("Enable Caching", value=True)
if enable_caching:
    st.sidebar.success("Caching is enabled for better performance")
else:
    st.sidebar.warning("Caching is disabled")

# Add about section
st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.markdown(
    """
    **Features:**
    - Load and explore dataset with proper train/validation/test splits
    - Preprocess images to grayscale with customizable parameters
    - Extract GLCM texture features with multiple parameters
    - Visualize high-dimensional features with t-SNE
    - Train KNN models with hyperparameter tuning
    - Make predictions with confidence scores

    **Version:** 2.0
    """
)
