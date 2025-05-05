import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import time
from utils import (
    load_dataset,
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
    split_options = ["train_cleaned", "test_cleaned", "valid_cleaned"]
    selected_split = st.selectbox("Select Split", split_options)

    if st.button("Load Dataset"):
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

    if st.session_state.dataset_loaded:
        st.success("✅ Dataset already loaded!")

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
            # Extract GLCM features
            glcm_features = extract_glcm_features(gray_img)

            # Create a DataFrame for better display
            properties = [
                "contrast",
                "dissimilarity",
                "homogeneity",
                "energy",
                "correlation",
                "ASM",
            ]
            angles = [0, 45, 90, 135]  # in degrees for display

            feature_df = pd.DataFrame(
                glcm_features.reshape(len(properties), len(angles)),
                index=properties,
                columns=[f"{angle}°" for angle in angles],
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

                # Extract features
                features = extract_features_from_dataset(
                    st.session_state.image_paths, progress_callback=update_progress
                )

                st.session_state.features = features
                st.session_state.features_extracted = True

                st.success(f"Features extracted successfully! Shape: {features.shape}")

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

        # KNN parameters
        n_neighbors = st.slider(
            "Number of Neighbors (K)", min_value=1, max_value=20, value=5, step=2
        )

        if st.button("Train KNN Model"):
            with st.spinner("Training KNN model..."):
                # Train KNN model
                model, scaler, accuracy = train_knn_model(
                    st.session_state.features,
                    st.session_state.labels,
                    n_neighbors=n_neighbors,
                )

                st.session_state.model = model
                st.session_state.scaler = scaler
                st.session_state.accuracy = accuracy
                st.session_state.model_trained = True

                st.success(f"Model trained successfully! Accuracy: {accuracy:.2%}")

                # Display confusion matrix
                y_pred = model.predict(scaler.transform(st.session_state.features))

                # Create confusion matrix
                from sklearn.metrics import confusion_matrix, classification_report

                cm = confusion_matrix(st.session_state.labels, y_pred)

                # Plot confusion matrix
                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
                ax.figure.colorbar(im, ax=ax)
                ax.set(
                    xticks=np.arange(cm.shape[1]),
                    yticks=np.arange(cm.shape[0]),
                    xticklabels=CLASSES,
                    yticklabels=CLASSES,
                    title="Confusion Matrix",
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
                report = classification_report(
                    st.session_state.labels, y_pred, target_names=CLASSES
                )
                st.text("Classification Report:")
                st.text(report)

# Step 6: Image Prediction
elif selected_step == steps[5]:
    st.header("6. Prediksi Gambar")

    if not st.session_state.model_trained:
        st.warning("Please train the model first (Step 5).")
    else:
        st.info("This step allows you to upload an image and predict its cat breed.")

        # Upload image
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Save the uploaded file temporarily
            temp_path = os.path.join("temp_upload.jpg")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Display the uploaded image
            st.image(temp_path, caption="Uploaded Image", use_column_width=True)

            if st.button("Predict"):
                with st.spinner("Predicting..."):
                    # Predict the class
                    predicted_class, probabilities = predict_image(
                        st.session_state.model, st.session_state.scaler, temp_path
                    )

                    # Display the prediction
                    st.success(f"Predicted Breed: {CLASSES[predicted_class]}")

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
                    ax.barh(prob_df["Breed"], prob_df["Probability"])
                    ax.set_xlabel("Probability")
                    ax.set_title("Prediction Probabilities")
                    ax.invert_yaxis()  # Display highest probability at the top

                    st.pyplot(fig)

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

                    # Extract and display GLCM features
                    glcm_features = extract_glcm_features(gray_img)

                    # Create a DataFrame for better display
                    properties = [
                        "contrast",
                        "dissimilarity",
                        "homogeneity",
                        "energy",
                        "correlation",
                        "ASM",
                    ]
                    angles = [0, 45, 90, 135]  # in degrees for display

                    feature_df = pd.DataFrame(
                        glcm_features.reshape(len(properties), len(angles)),
                        index=properties,
                        columns=[f"{angle}°" for angle in angles],
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
