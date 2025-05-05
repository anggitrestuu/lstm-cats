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
CLASSES = ["Angora", "British Shorthair", "Persian"]


# Fungsi untuk memuat dataset
@st.cache_data
def load_dataset(dataset_path="dataset", split="train_cleaned"):
    """
    Load dataset from the specified path and split
    """
    split_dir = os.path.join(dataset_path, split)
    csv_path = os.path.join(split_dir, "_classes.csv")

    if not os.path.exists(csv_path):
        st.error(f"CSV file not found at {csv_path}")
        return None, None

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

    return pd.DataFrame({"filepath": image_paths, "label": labels})


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
    return prediction.argmax(axis=1)


# Fungsi untuk mengonversi gambar menjadi grayscale
def convert_to_grayscale(image_path):
    img = cv2.imread(image_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img


# Fungsi untuk preprocessing semua gambar ke grayscale
def preprocess_all_to_grayscale(df):
    # Create dynamic gray folder path
    df["gray_filepath"] = df["filepath"].apply(
        lambda x: os.path.join("gray_" + os.path.dirname(x), os.path.basename(x))
    )

    for i in df.index:
        original_path = df["filepath"].iloc[i]
        gray_path = df["gray_filepath"].iloc[i]

        # Membaca gambar dan mengubahnya menjadi grayscale
        img = cv2.imread(original_path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Membuat direktori jika belum ada
        gray_folder = os.path.dirname(gray_path)
        if not os.path.exists(gray_folder):
            os.makedirs(gray_folder)

        # Menyimpan gambar grayscale
        cv2.imwrite(gray_path, gray_img)

    return df


# Sidebar untuk navigasi menu
st.sidebar.title("Cat Breed Classification")
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
    st.write("Di menu ini, dataset kucing akan di-load dari direktori dataset.")

    # Add split selection
    split_options = ["train_cleaned", "test_cleaned", "valid_cleaned"]
    selected_split = st.selectbox("Select Split", split_options, index=0)

    df = load_dataset(split=selected_split)

    if df is not None:
        st.write(f"Dataset Preview ({len(df)} images):")

        # Convert numeric labels to breed names for display
        df_display = df.copy()
        df_display["breed"] = df_display["label"].apply(lambda x: CLASSES[x])

        st.dataframe(df_display.head())

        # Display class distribution
        st.subheader("Class Distribution")
        class_counts = df["label"].value_counts().sort_index()
        class_df = pd.DataFrame({"Breed": CLASSES, "Count": class_counts.values})

        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(class_df)

        with col2:
            fig, ax = plt.subplots()
            ax.bar(CLASSES, class_counts.values)
            ax.set_ylabel("Count")
            ax.set_title("Class Distribution")
            st.pyplot(fig)

        # Display sample images
        st.subheader("Sample Images")
        sample_indices = np.random.choice(len(df), min(5, len(df)), replace=False)

        cols = st.columns(len(sample_indices))
        for i, idx in enumerate(sample_indices):
            with cols[i]:
                img_path = df["filepath"].iloc[idx]
                label = CLASSES[df["label"].iloc[idx]]
                img = cv2.imread(img_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                st.image(img_rgb, caption=f"Class: {label}", use_column_width=True)

# Menu 2: Preprocessing Gambar ke Grayscale
elif menu == "2. Preprocessing Gambar ke Grayscale":
    st.title("Preprocessing Gambar ke Grayscale")
    st.write("Semua gambar pada dataset akan diubah menjadi grayscale.")

    # Load dataset
    split_options = ["train_cleaned", "test_cleaned", "valid_cleaned"]
    selected_split = st.selectbox("Select Split", split_options, index=0)

    df = load_dataset(split=selected_split)

    if df is not None:
        # Proses preprocessing gambar
        if st.button("Process to Grayscale"):
            df = preprocess_all_to_grayscale(df)
            st.write("Proses preprocessing selesai.")
            st.dataframe(df.head())

            # Tampilkan contoh gambar asli dan grayscale
            st.write("Contoh gambar sebelum dan sesudah preprocessing:")
            original_img_path = df["filepath"].iloc[0]
            gray_img_path = df["gray_filepath"].iloc[0]

            original_img = cv2.imread(original_img_path)
            original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            gray_img = cv2.imread(gray_img_path, cv2.IMREAD_GRAYSCALE)

            col1, col2 = st.columns(2)
            with col1:
                st.image(original_img_rgb, caption="Gambar Asli", use_column_width=True)
            with col2:
                st.image(gray_img, caption="Gambar Grayscale", use_column_width=True)

# Menu 3: Ekstraksi Fitur GLCM
elif menu == "3. Ekstraksi Fitur GLCM":
    st.title("Ekstraksi Fitur GLCM")
    st.write("Ekstraksi fitur GLCM dari gambar kucing pada dataset.")

    # Load data
    split_options = ["train_cleaned", "test_cleaned", "valid_cleaned"]
    selected_split = st.selectbox("Select Split", split_options, index=0)

    df = load_dataset(split=selected_split)

    if df is not None:
        # Sample option
        sample_size = st.slider("Sample Size per Class", 10, 100, 50)

        if st.button("Extract GLCM Features"):
            # Sampling dan ekstraksi fitur GLCM
            sampled_df = pd.DataFrame()
            for class_idx in range(len(CLASSES)):
                class_df = df[df["label"] == class_idx]
                if len(class_df) > sample_size:
                    class_sample = class_df.sample(n=sample_size)
                else:
                    class_sample = class_df
                sampled_df = pd.concat([sampled_df, class_sample], axis=0)

            sampled_df.reset_index(drop=True, inplace=True)

            with st.spinner("Extracting GLCM features..."):
                # Ekstraksi fitur GLCM
                df_with_glcm = add_glcm(sampled_df)
                st.write("Ekstraksi GLCM selesai.")
                st.dataframe(df_with_glcm.head())

                # Save to session state for later use
                st.session_state.df_with_glcm = df_with_glcm

# Menu 4: Visualisasi t-SNE
elif menu == "4. Visualisasi t-SNE":
    st.title("Visualisasi t-SNE dari Fitur GLCM")
    st.write(
        "Visualisasi data menggunakan t-SNE berdasarkan fitur GLCM yang telah diekstrak."
    )

    if "df_with_glcm" not in st.session_state:
        st.warning("Please extract GLCM features first (Step 3).")
    else:
        df = st.session_state.df_with_glcm

        # t-SNE parameters
        perplexity = st.slider("Perplexity", 5, 50, 30)
        n_iter = st.slider("Iterations", 250, 1000, 300)

        if st.button("Generate t-SNE Visualization"):
            with st.spinner("Computing t-SNE..."):
                # t-SNE
                model = TSNE(
                    n_components=2, random_state=0, perplexity=perplexity, n_iter=n_iter
                )
                tsne_data = model.fit_transform(df.loc[:, "cont_red":"corr_blue"])

                # Create plot
                plt.figure(figsize=(10, 8))

                # Convert numeric labels to breed names for display
                labels = [CLASSES[label] for label in df["label"]]

                sb.scatterplot(x=tsne_data[:, 0], y=tsne_data[:, 1], hue=labels)
                plt.title("t-SNE Visualization of Cat Breeds")
                st.pyplot(plt)

# Menu 5: Pelatihan Model KNN
elif menu == "5. Pelatihan Model KNN":
    st.title("Pelatihan Model KNN")
    st.write("Latih model KNN untuk klasifikasi breed kucing menggunakan fitur GLCM.")

    if "df_with_glcm" not in st.session_state:
        st.warning("Please extract GLCM features first (Step 3).")
    else:
        df = st.session_state.df_with_glcm

        # KNN parameters
        n_neighbors = st.slider("Number of Neighbors (K)", 1, 20, 5, step=2)
        test_size = st.slider("Test Size", 0.1, 0.5, 0.2, step=0.05)

        if st.button("Train KNN Model"):
            with st.spinner("Training model..."):
                # Fitur yang dihapus
                df = df.drop(to_remove, axis=1, errors="ignore")

                # Split data dan pelatihan model
                features = df.loc[:, "cont_red":"corr_blue"]
                target = pd.get_dummies(df["label"])
                X_train, X_val, Y_train, Y_val = train_test_split(
                    features, target, test_size=test_size, random_state=42
                )

                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_val = scaler.transform(X_val)

                knn = KNeighborsClassifier(n_neighbors=n_neighbors)
                knn.fit(X_train, Y_train)

                # Validasi dan akurasi
                Y_pred = knn.predict(X_val)
                accuracy = accuracy_score(Y_val, Y_pred)
                st.write(f"Akurasi model KNN: {accuracy:.2f}")

                # Save model and scaler to session state
                st.session_state.knn = knn
                st.session_state.scaler = scaler

                # Confusion Matrix
                st.write("Confusion Matrix:")
                conf_matrix = confusion_matrix(
                    Y_val.values.argmax(axis=1), Y_pred.argmax(axis=1)
                )
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
    st.title("Prediksi Breed Kucing")
    st.write("Unggah gambar kucing dan model akan memprediksi breednya.")

    if "knn" not in st.session_state or "scaler" not in st.session_state:
        st.warning("Please train the model first (Step 5).")
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
            original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

            # Konversi ke grayscale
            gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

            # Simpan gambar grayscale sementara
            cv2.imwrite("temp_image_gray.jpg", gray_img)

            # Tampilkan gambar asli dan hasil preprocessing grayscale
            st.write("Gambar asli dan hasil preprocessing grayscale:")
            col1, col2 = st.columns(2)
            with col1:
                st.image(original_img_rgb, caption="Gambar Asli", use_column_width=True)
            with col2:
                st.image(
                    gray_img,
                    caption="Gambar Grayscale",
                    use_column_width=True,
                )

            # Lakukan prediksi
            if st.button("Predict"):
                with st.spinner("Predicting..."):
                    # Prediksi gambar yang diunggah
                    predicted_label = predict_image(
                        st.session_state.knn, st.session_state.scaler, "temp_image.jpg"
                    )
                    breed = CLASSES[predicted_label[0]]
                    st.success(f"Hasil prediksi: {breed}")

                    st.image(
                        original_img_rgb,
                        caption=f"Predicted: {breed}",
                        use_column_width=True,
                    )
