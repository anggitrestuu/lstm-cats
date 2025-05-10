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

# Define cat breed classes
CLASSES = ["domistik", "persian", "turkish"]


# Fungsi untuk memuat dataset
@st.cache_data
def load_data():
    # Create a dataframe from the directory structure
    data = []
    dataset_dir = "dataset_restructured"

    # Process training data
    train_dir = os.path.join(dataset_dir, "train")
    for label in CLASSES:
        class_dir = os.path.join(train_dir, label)
        if os.path.exists(class_dir):
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                    data.append(
                        {
                            "image": img_file,
                            "label": label,
                            "split": "training",
                            "filepath": os.path.join(class_dir, img_file),
                        }
                    )

    # Process validation data
    val_dir = os.path.join(dataset_dir, "val")
    for label in CLASSES:
        class_dir = os.path.join(val_dir, label)
        if os.path.exists(class_dir):
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                    data.append(
                        {
                            "image": img_file,
                            "label": label,
                            "split": "validation",
                            "filepath": os.path.join(class_dir, img_file),
                        }
                    )

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
            st.error(f"Could not read image at {path}")
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
        st.error(f"Could not read image at {image_path}")
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
    return prediction[0]


# Fungsi untuk preprocessing semua gambar ke grayscale
def preprocess_all_to_grayscale(df):
    # Create gray folder path
    gray_base_dir = "gray_dataset_restructured"
    os.makedirs(gray_base_dir, exist_ok=True)

    # Create subdirectories
    for split in ["training", "validation"]:
        for label in CLASSES:
            os.makedirs(os.path.join(gray_base_dir, split, label), exist_ok=True)

    # Add gray filepath column
    df["gray_filepath"] = df.apply(
        lambda row: os.path.join(
            gray_base_dir, row["split"], row["label"], os.path.basename(row["filepath"])
        ),
        axis=1,
    )

    for i in df.index:
        original_path = df["filepath"].iloc[i]
        gray_path = df["gray_filepath"].iloc[i]

        # Membaca gambar dan mengubahnya menjadi grayscale
        img = cv2.imread(original_path)
        if img is None:
            st.warning(f"Could not read image at {original_path}")
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

    df = load_data()
    st.write("Dataset Preview:")
    st.dataframe(df.head())

    # Display class distribution
    st.write("Distribusi Kelas:")
    class_counts = df["label"].value_counts()
    st.bar_chart(class_counts)

# Menu 2: Preprocessing Gambar ke Grayscale
elif menu == "2. Preprocessing Gambar ke Grayscale":
    st.title("Preprocessing Gambar ke Grayscale")
    st.write("Semua gambar pada dataset akan diubah menjadi grayscale.")

    # Load dataset
    df = load_data()

    # Proses preprocessing gambar
    if st.button("Proses Grayscale"):
        with st.spinner("Memproses gambar ke grayscale..."):
            df = preprocess_all_to_grayscale(df)
            st.success("Proses preprocessing selesai.")
            st.dataframe(df.head())

            # Tampilkan contoh gambar asli dan grayscale
            st.write("Contoh gambar sebelum dan sesudah preprocessing:")
            original_img_path = df["filepath"].iloc[0]
            gray_img_path = df["gray_filepath"].iloc[0]

            original_img = cv2.imread(original_img_path)
            gray_img = cv2.imread(gray_img_path, cv2.IMREAD_GRAYSCALE)

            col1, col2 = st.columns(2)
            with col1:
                st.image(
                    cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB),
                    caption="Gambar Asli",
                    use_column_width=True,
                )
            with col2:
                st.image(gray_img, caption="Gambar Grayscale", use_column_width=True)

# Menu 3: Ekstraksi Fitur GLCM
elif menu == "3. Ekstraksi Fitur GLCM":
    st.title("Ekstraksi Fitur GLCM")
    st.write("Ekstraksi fitur GLCM dari gambar kucing pada dataset.")

    # Load data
    df = load_data()

    # Sample data for faster processing if needed
    sample_size = st.slider("Jumlah sampel per kelas", 10, 300, 180)

    if st.button("Ekstrak Fitur GLCM"):
        with st.spinner("Mengekstrak fitur GLCM..."):
            # Sample data
            sampled_df = pd.DataFrame()
            for label in CLASSES:
                class_df = df[df["label"] == label].sample(
                    n=min(sample_size, len(df[df["label"] == label]))
                )
                sampled_df = pd.concat([sampled_df, class_df])

            sampled_df.reset_index(drop=True, inplace=True)

            # Ekstraksi fitur GLCM
            sampled_df = add_glcm(sampled_df)
            st.success("Ekstraksi GLCM selesai.")
            st.dataframe(sampled_df.head())

            # Save to session state for later use
            st.session_state.glcm_df = sampled_df

# Menu 4: Visualisasi t-SNE
elif menu == "4. Visualisasi t-SNE":
    st.title("Visualisasi t-SNE dari Fitur GLCM")
    st.write(
        "Visualisasi data menggunakan t-SNE berdasarkan fitur GLCM yang telah diekstrak."
    )

    if "glcm_df" not in st.session_state:
        st.warning("Silakan ekstrak fitur GLCM terlebih dahulu (Menu 3).")
    else:
        df = st.session_state.glcm_df

        # t-SNE
        if st.button("Visualisasi t-SNE"):
            with st.spinner("Membuat visualisasi t-SNE..."):
                # Get GLCM features
                features_cols = [
                    col
                    for col in df.columns
                    if col.startswith(("cont", "diss", "homo", "ener", "ASM", "corr"))
                ]
                features = df[features_cols].drop(to_remove, axis=1, errors="ignore")

                # Create label mapping
                label_mapping = {label: i for i, label in enumerate(CLASSES)}
                labels = df["label"].map(label_mapping)

                # Apply t-SNE
                model = TSNE(n_components=2, random_state=0)
                tsne_data = model.fit_transform(features)

                # Create plot
                plt.figure(figsize=(10, 8))
                for i, label in enumerate(CLASSES):
                    indices = labels == i
                    plt.scatter(
                        tsne_data[indices, 0], tsne_data[indices, 1], label=label
                    )

                plt.title("t-SNE Visualization of Cat Breeds")
                plt.legend()
                st.pyplot(plt)

# Menu 5: Pelatihan Model KNN
elif menu == "5. Pelatihan Model KNN":
    st.title("Pelatihan Model KNN")
    st.write("Latih model KNN untuk klasifikasi jenis kucing menggunakan fitur GLCM.")

    if "glcm_df" not in st.session_state:
        st.warning("Silakan ekstrak fitur GLCM terlebih dahulu (Menu 3).")
    else:
        df = st.session_state.glcm_df

        # Get features and labels
        features_cols = [
            col
            for col in df.columns
            if col.startswith(("cont", "diss", "homo", "ener", "ASM", "corr"))
        ]
        features = df[features_cols].drop(to_remove, axis=1, errors="ignore")

        # Create label mapping
        label_mapping = {label: i for i, label in enumerate(CLASSES)}
        labels = df["label"].map(label_mapping)

        # KNN parameters
        n_neighbors = st.slider("Jumlah Tetangga (K)", 1, 20, 5)
        test_size = st.slider("Ukuran Data Test", 0.1, 0.5, 0.2)

        if st.button("Latih Model KNN"):
            with st.spinner("Melatih model KNN..."):
                # Split data
                X_train, X_val, y_train, y_val = train_test_split(
                    features,
                    labels,
                    test_size=test_size,
                    random_state=42,
                    stratify=labels,
                )

                # Scale features
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_val = scaler.transform(X_val)

                # Train KNN
                knn = KNeighborsClassifier(n_neighbors=n_neighbors)
                knn.fit(X_train, y_train)

                # Evaluate
                y_pred = knn.predict(X_val)
                accuracy = accuracy_score(y_val, y_pred)

                # Save model to session state
                st.session_state.knn_model = knn
                st.session_state.scaler = scaler

                # Display results
                st.success(f"Akurasi model KNN: {accuracy:.2f}")

                # Confusion Matrix
                st.write("Confusion Matrix:")
                conf_matrix = confusion_matrix(y_val, y_pred)
                plt.figure(figsize=(8, 6))
                sb.heatmap(
                    conf_matrix,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=CLASSES,
                    yticklabels=CLASSES,
                )
                plt.ylabel("Actual")
                plt.xlabel("Predicted")
                plt.title("Confusion Matrix")
                st.pyplot(plt)

# Menu 6: Prediksi Gambar
elif menu == "6. Prediksi Gambar":
    st.title("Prediksi Jenis Kucing")
    st.write("Unggah gambar kucing dan model akan memprediksi jenisnya.")

    if "knn_model" not in st.session_state or "scaler" not in st.session_state:
        st.warning("Silakan latih model KNN terlebih dahulu (Menu 5).")
    else:
        # Upload file gambar
        uploaded_file = st.file_uploader(
            "Unggah gambar kucing", type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            # Simpan gambar
            with open("temp_image.jpg", "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Baca gambar asli
            original_img = cv2.imread("temp_image.jpg")

            if original_img is None:
                st.error("Tidak dapat membaca gambar yang diunggah.")
            else:
                # Konversi ke RGB untuk tampilan
                display_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

                # Tampilkan gambar
                st.image(
                    display_img, caption="Gambar yang Diunggah", use_column_width=True
                )

                # Lakukan prediksi
                knn = st.session_state.knn_model
                scaler = st.session_state.scaler

                # Prediksi gambar yang diunggah
                predicted_label = predict_image(knn, scaler, "temp_image.jpg")

                if predicted_label is not None:
                    st.success(f"Hasil prediksi: {CLASSES[predicted_label]}")
