# Cat Breed Classification App

This Streamlit application demonstrates cat breed classification using GLCM (Gray-Level Co-occurrence Matrix) features and KNN (K-Nearest Neighbors) algorithm.

## Features

1. **Load Dataset**: Load and explore the cat breed dataset
2. **Preprocessing Images to Grayscale**: Convert images to grayscale for feature extraction
3. **GLCM Feature Extraction**: Extract texture features using GLCM
4. **t-SNE Visualization**: Visualize high-dimensional features in 2D space
5. **KNN Model Training**: Train a KNN classifier on the extracted features
6. **Image Prediction**: Upload and classify new cat images

## Dataset

The application uses a dataset of cat images with three breeds:
- Angora
- British Shorthair
- Persian

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

## How It Works

### GLCM Feature Extraction

Gray-Level Co-occurrence Matrix (GLCM) is a statistical method for examining texture by considering the spatial relationship of pixels. The application extracts the following GLCM features:

- Contrast
- Dissimilarity
- Homogeneity
- Energy
- Correlation
- Angular Second Moment (ASM)

### t-SNE Visualization

t-Distributed Stochastic Neighbor Embedding (t-SNE) is used to visualize the high-dimensional GLCM features in 2D space, allowing us to see how well the features separate the different cat breeds.

### KNN Classification

K-Nearest Neighbors (KNN) is a simple, instance-based learning algorithm used for classification. The application uses KNN to classify cat images based on their GLCM features.

## Project Structure

- `app.py`: Main Streamlit application
- `utils.py`: Utility functions for image processing, feature extraction, and model training
- `dataset/`: Directory containing the cat images dataset

## Requirements

- Python 3.7+
- OpenCV
- scikit-image
- scikit-learn
- Streamlit
- NumPy
- Pandas
- Matplotlib
- Seaborn



## Dataset 
curl -L "https://app.roboflow.com/ds/JmGcEN6KdW?key=AT9EOBTAdi" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip