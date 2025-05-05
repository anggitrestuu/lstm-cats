import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import cv2
import os

from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from skimage.feature.texture import graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import warnings

warnings.filterwarnings("ignore")


# Fungsi untuk memuat dataset
@st.cache_data
def load_data(dataset_path="dataset_restructured", split="train"):
    """
    Load dataset from directory structure
    """
    data = []
    split_path = os.path.join(dataset_path, split)

    # Get all class folders
    class_folders = [
        f for f in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, f))
    ]

    # Loop through each class folder
    for class_name in class_folders:
        class_path = os.path.join(split_path, class_name)
        # Get all images in this class folder
        images = [
            f for f in os.listdir(class_path) if f.endswith((".jpg", ".jpeg", ".png"))
        ]

        # Add each image to our data list
        for img in images:
            data.append(
                {
                    "image": img,
                    "label": class_name,
                    "filepath": os.path.join(class_path, img),
                }
            )

    # Convert to DataFrame
    df = pd.DataFrame(data)
    return df


# Fungsi untuk menambahkan fitur GLCM
def add_glcm(df):
    color = ["red", "green", "blue"]
    features = [
        "dissimilarity",
        "homogeneity",
        "contrast",
        "energy",
        "ASM",
        "correlation",
    ]

    for col in color:
        for feat in features:
            df[f"{feat[0:4]}_{col}"] = 0

    for i in df.index:
        path = df["filepath"].iloc[i]
        img = cv2.imread(path)
        if img is None:
            st.error(f"Failed to read image: {path}")
            continue
        for j, col in enumerate(color):
            glcm = graycomatrix(
                img[:, :, j],
                distances=[5],
                angles=[0],
                levels=256,
                symmetric=True,
                normed=True,
            )
            for feat in features:
                val = graycoprops(glcm, feat)[0][0]
                df.loc[i, f"{feat[0:4]}_{col}"] = val
    return df


# Variabel fitur yang akan dihapus, didefinisikan secara global
to_remove = [
    "ener_blue",
    "diss_blue",
    "cont_green",
    "diss_green",
    "ASM_red",
    "homo_red",
    "diss_red",
    "ener_green",
]


# Fungsi untuk melakukan prediksi gambar
def predict_image(knn, scaler, image_path):
    img = cv2.imread(image_path)
    if img is None:
        st.error(f"Failed to read image: {image_path}")
        return None

    color = ["red", "green", "blue"]
    features = [
        "dissimilarity",
        "homogeneity",
        "contrast",
        "energy",
        "ASM",
        "correlation",
    ]

    # Ekstrak fitur GLCM dari gambar yang diunggah
    data = {}
    for col in color:
        for feat in features:
            data[f"{feat[0:4]}_{col}"] = 0

    for j, col in enumerate(color):
        glcm = graycomatrix(
            img[:, :, j],
            distances=[5],
            angles=[0],
            levels=256,
            symmetric=True,
            normed=True,
        )
        for feat in features:
            val = graycoprops(glcm, feat)[0][0]
            data[f"{feat[0:4]}_{col}"] = val

    # Buat DataFrame untuk fitur
    feature_df = pd.DataFrame([data])
    feature_df = feature_df.drop(to_remove, axis=1, errors="ignore")
    scaled_features = scaler.transform(feature_df)

    # Prediksi
    prediction = knn.predict(scaled_features)
    return prediction[0]  # Return single prediction instead of argmax


# Fungsi untuk preprocessing semua gambar ke grayscale
def preprocess_all_to_grayscale(df):
    # Create gray folder path
    gray_base_dir = "gray_dataset"
    os.makedirs(gray_base_dir, exist_ok=True)

    # Create class folders
    classes = df["label"].unique()
    for cls in classes:
        os.makedirs(os.path.join(gray_base_dir, cls), exist_ok=True)

    df["gray_filepath"] = df.apply(
        lambda row: os.path.join(
            gray_base_dir, row["label"], os.path.basename(row["filepath"])
        ),
        axis=1,
    )

    for i in df.index:
        original_path = df["filepath"].iloc[i]
        gray_path = df["gray_filepath"].iloc[i]

        # Membaca gambar dan mengubahnya menjadi grayscale
        img = cv2.imread(original_path)
        if img is None:
            st.error(f"Failed to read image: {original_path}")
            continue

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Menyimpan gambar grayscale
        cv2.imwrite(gray_path, gray_img)

    return df


# Sidebar untuk navigasi menu
st.sidebar.title("MENU")
menu = st.sidebar.selectbox(
    "Pilih Menu:",
    [
        "1. Load Dataset",
        "2. Preprocessing Gambar ke Grayscale",
        "3. Ekstraksi Fitur GLCM",
        "4. Visualisasi t-SNE",
        "5. Pelatihan Model KNN",
        "6. Prediksi Gambar",
    ],
)

# Menu 1: Load Dataset
if menu == "1. Load Dataset":
    st.title("Load Dataset")
    st.write("Di menu ini, dataset akan di-load dari direktori `dataset_restructured`.")

    # Add option to select split
    split = st.selectbox("Pilih Split Dataset:", ["train", "val", "test"])

    df = load_data(split=split)
    st.write(f"Dataset Preview ({split}):")
    st.dataframe(df.head())

    # Show class distribution
    st.write("Distribusi Kelas:")
    class_counts = df["label"].value_counts()
    st.bar_chart(class_counts)

    # Show sample images
    st.write("Contoh Gambar:")
    classes = df["label"].unique()
    cols = st.columns(len(classes))

    for i, cls in enumerate(classes):
        with cols[i]:
            st.write(f"Kelas: {cls}")
            sample = df[df["label"] == cls].sample(1).iloc[0]
            img = cv2.imread(sample["filepath"])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(img, use_column_width=True)

# Menu 2: Preprocessing Gambar ke Grayscale
elif menu == "2. Preprocessing Gambar ke Grayscale":
    st.title("Preprocessing Gambar ke Grayscale")
    st.write("Semua gambar pada dataset akan diubah menjadi grayscale.")

    # Add option to select split
    split = st.selectbox("Pilih Split Dataset:", ["train", "val", "test"])

    # Load dataset
    df = load_data(split=split)

    if st.button("Mulai Preprocessing"):
        # Proses preprocessing gambar
        with st.spinner("Melakukan preprocessing..."):
            df = preprocess_all_to_grayscale(df)
        st.success("Proses preprocessing selesai.")
        st.dataframe(df.head())

        # Tampilkan contoh gambar asli dan grayscale
        st.write("Contoh gambar sebelum dan sesudah preprocessing:")
        sample = df.sample(1).iloc[0]
        original_img_path = sample["filepath"]
        gray_img_path = sample["gray_filepath"]

        original_img = cv2.imread(original_img_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        gray_img = cv2.imread(gray_img_path, cv2.IMREAD_GRAYSCALE)

        col1, col2 = st.columns(2)
        with col1:
            st.image(original_img, caption="Gambar Asli", use_column_width=True)
        with col2:
            st.image(gray_img, caption="Gambar Grayscale", use_column_width=True)

# Menu 3: Ekstraksi Fitur GLCM
elif menu == "3. Ekstraksi Fitur GLCM":
    st.title("Ekstraksi Fitur GLCM")
    st.write("Ekstraksi fitur GLCM dari gambar pada dataset.")

    # Add option to select split
    split = st.selectbox("Pilih Split Dataset:", ["train", "val", "test"])

    # Load data
    df = load_data(split=split)

    # Option to limit sample size for faster processing
    sample_size = st.slider("Jumlah sampel per kelas:", 10, 100, 30)

    if st.button("Ekstrak Fitur GLCM"):
        with st.spinner("Mengekstrak fitur GLCM..."):
            # Sample data evenly from each class
            classes = df["label"].unique()
            sampled_data = []

            for cls in classes:
                class_data = df[df["label"] == cls]
                if len(class_data) > sample_size:
                    sampled_data.append(class_data.sample(sample_size))
                else:
                    sampled_data.append(class_data)

            sampled_df = pd.concat(sampled_data)
            sampled_df.reset_index(drop=True, inplace=True)

            # Ekstraksi fitur GLCM
            df_with_features = add_glcm(sampled_df)

        st.success("Ekstraksi GLCM selesai.")
        st.write("Data dengan fitur GLCM:")
        st.dataframe(df_with_features.head())

        # Save to session state for other steps
        st.session_state["glcm_features"] = df_with_features

# Menu 4: Visualisasi t-SNE
elif menu == "4. Visualisasi t-SNE":
    st.title("Visualisasi t-SNE dari Fitur GLCM")
    st.write(
        "Visualisasi data menggunakan t-SNE berdasarkan fitur GLCM yang telah diekstrak."
    )

    if "glcm_features" not in st.session_state:
        st.warning("Silakan ekstrak fitur GLCM terlebih dahulu (Menu 3).")
    else:
        df = st.session_state["glcm_features"]

        # Get feature columns (all columns that start with feature prefixes)
        feature_cols = [
            col
            for col in df.columns
            if any(
                col.startswith(f"{feat[0:4]}_")
                for feat in [
                    "dissimilarity",
                    "homogeneity",
                    "contrast",
                    "energy",
                    "ASM",
                    "correlation",
                ]
            )
        ]

        # Remove features that should be excluded
        feature_cols = [col for col in feature_cols if col not in to_remove]

        # t-SNE parameters
        perplexity = st.slider("Perplexity:", 5, 50, 30)
        n_iter = st.slider("Iterations:", 250, 2000, 1000)

        if st.button("Visualisasi t-SNE"):
            with st.spinner("Menjalankan t-SNE..."):
                # t-SNE
                model = TSNE(
                    n_components=2, random_state=0, perplexity=perplexity, n_iter=n_iter
                )
                tsne_data = model.fit_transform(df[feature_cols])

                # Create plot
                plt.figure(figsize=(10, 8))
                scatter = sb.scatterplot(
                    x=tsne_data[:, 0],
                    y=tsne_data[:, 1],
                    hue=df["label"],
                    palette="bright",
                )
                plt.title("t-SNE Visualization")
                plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

                st.pyplot(plt)

# Menu 5: Pelatihan Model KNN
elif menu == "5. Pelatihan Model KNN":
    st.title("Pelatihan Model KNN")
    st.write("Latih model KNN untuk klasifikasi menggunakan fitur GLCM.")

    if "glcm_features" not in st.session_state:
        st.warning("Silakan ekstrak fitur GLCM terlebih dahulu (Menu 3).")
    else:
        df = st.session_state["glcm_features"]

        # Get feature columns
        feature_cols = [
            col
            for col in df.columns
            if any(
                col.startswith(f"{feat[0:4]}_")
                for feat in [
                    "dissimilarity",
                    "homogeneity",
                    "contrast",
                    "energy",
                    "ASM",
                    "correlation",
                ]
            )
        ]

        # Remove features that should be excluded
        feature_cols = [col for col in feature_cols if col not in to_remove]

        # KNN parameters
        n_neighbors = st.slider("Jumlah Tetangga (k):", 1, 20, 5)
        test_size = st.slider("Ukuran Data Test:", 0.1, 0.5, 0.2)

        if st.button("Latih Model KNN"):
            with st.spinner("Melatih model KNN..."):
                # Prepare features and target
                X = df[feature_cols]

                # Convert labels to numeric
                classes = df["label"].unique()
                class_to_num = {cls: i for i, cls in enumerate(classes)}
                y = df["label"].map(class_to_num)

                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )

                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Train KNN model
                knn = KNeighborsClassifier(n_neighbors=n_neighbors)
                knn.fit(X_train_scaled, y_train)

                # Evaluate
                y_pred = knn.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)

                # Save model to session state
                st.session_state["knn_model"] = knn
                st.session_state["scaler"] = scaler
                st.session_state["classes"] = classes
                st.session_state["class_to_num"] = class_to_num

                # Display results
                st.success(f"Model berhasil dilatih dengan akurasi: {accuracy:.2f}")

                # Confusion Matrix
                cm = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(10, 8))
                sb.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=classes,
                    yticklabels=classes,
                )
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                plt.title("Confusion Matrix")
                st.pyplot(plt)

# Menu 6: Prediksi Gambar
elif menu == "6. Prediksi Gambar":
    st.title("Prediksi Gambar")
    st.write("Unggah gambar dan model akan memprediksi kelasnya.")

    if "knn_model" not in st.session_state:
        st.warning("Silakan latih model KNN terlebih dahulu (Menu 5).")
    else:
        # Upload file gambar
        uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Simpan gambar
            with open("temp_image.jpg", "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Baca gambar asli
            original_img = cv2.imread("temp_image.jpg")
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

            # Konversi ke grayscale
            gray_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)

            # Simpan gambar grayscale sementara
            cv2.imwrite("temp_image_gray.jpg", gray_img)

            # Tampilkan gambar asli dan hasil preprocessing grayscale
            st.write("Gambar asli dan hasil preprocessing grayscale:")
            col1, col2 = st.columns(2)
            with col1:
                st.image(original_img, caption="Gambar Asli", use_column_width=True)
            with col2:
                st.image(gray_img, caption="Gambar Grayscale", use_column_width=True)

            if st.button("Prediksi"):
                with st.spinner("Melakukan prediksi..."):
                    # Get model and scaler from session state
                    knn = st.session_state["knn_model"]
                    scaler = st.session_state["scaler"]
                    classes = st.session_state["classes"]

                    # Prediksi gambar yang diunggah
                    predicted_class_num = predict_image(knn, scaler, "temp_image.jpg")

                    if predicted_class_num is not None:
                        # Map numeric prediction back to class name
                        predicted_class = classes[predicted_class_num]
                        st.success(f"Hasil prediksi: {predicted_class}")
