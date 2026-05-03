# Hand Gesture Letter Recognition

A computer vision and machine learning project for detecting and classifying hand gestures captured from a laptop camera. The project combines face detection, skin-color-based hand localization, dataset generation, and neural network classification to recognize hand-made letter signs.

This repository contains a university computer vision lab project originally developed in Google Colab and later exported to GitHub for documentation and portfolio purposes.

## Overview

The project explores a full pipeline for real-time hand gesture recognition:

1. detect the face in the camera frame,
2. use the detected face region to estimate a skin-color distribution,
3. suppress the face region and search for the hand,
4. capture and preprocess hand images,
5. build datasets for selected letters,
6. train MLP models to classify the gestures,
7. run inference on live camera input.

The selected gesture classes in this project are the letters **M**, **N**, and **W**.

## Main components

- **Face detection** using a Haar cascade on grayscale images
- **Region of interest tracking** for identifying relevant areas in the frame
- **CamShift-based color tracking** to model skin-color distribution
- **Hand extraction and cropping** from the video feed
- **Dataset generation** with different class-balance and variability settings
- **MLP classification** for recognizing hand gesture letters
- **Live prediction** on camera input

## Repository contents

- `CompVision_Ilaria.ipynb` — main notebook containing the full project workflow
- `dataset1.txt`, `dataset2.txt`, `dataset3.txt` — dataset and experiment output logs
- `model1.json`, `model2.json`, `model3.json` — saved model architectures
- `model1_weights.h5`, `model2_weights.h5`, `model3_weights.h5` — trained model weights
- `images/` — image assets or project data used in the notebook

## Method

### 1. Face detection
The first stage detects the face using a Haar cascade on a grayscale image. Grayscale reduces the amount of information to process and makes detection more efficient than working directly on full-color frames.

### 2. Face-based color modeling
After detecting the face, the project uses the face region as a reference area to estimate a skin-color distribution. This information is then used to search for other regions in the frame with similar characteristics.

### 3. Hand localization
The face region is excluded from the probability map so that the algorithm focuses on locating the hands instead of repeatedly identifying the face.

### 4. Data collection
The system captures hand images at user-defined intervals and stores them in multiple sizes, including **16×16** and **224×224**, for later processing and training.

### 5. Dataset creation
Three datasets were created to compare how class balance and variability affect model performance:

- **Dataset 1**: balanced classes with high variability
- **Dataset 2**: unbalanced classes (50 / 100 / 150 samples) with high variability
- **Dataset 3**: balanced classes where one class has low variability

### 6. Model training
Three MLP models were trained and evaluated on the datasets to compare their behavior under different data conditions.

## Results

### Model 1

| Dataset | Train/Test Split | Validation Loss | Validation Accuracy |
|---|---:|---:|---:|
| Dataset 1 | 210 / 90 | 1.4553 | 0.6556 |
| Dataset 2 | 244 / 106 | 0.9063 | 0.8302 |
| Dataset 3 | 210 / 90 | 0.6691 | 0.8444 |

**Observation:** Model 1 performs best on Datasets 2 and 3.

### Model 2

| Dataset | Train/Test Split | Validation Loss | Validation Accuracy |
|---|---:|---:|---:|
| Dataset 1 | 210 / 90 | 1.7095 | 0.7667 |
| Dataset 2 | 244 / 106 | 0.9224 | 0.8396 |
| Dataset 3 | 210 / 90 | 1.2521 | 0.7556 |

**Observation:** Model 2 performs best on Dataset 2, likely benefiting from the dominant class distribution.

### Model 3

| Dataset | Train/Test Split | Validation Loss | Validation Accuracy |
|---|---:|---:|---:|
| Dataset 1 | 210 / 90 | 1.2044 | 0.7889 |
| Dataset 2 | 244 / 106 | 1.2693 | 0.7642 |
| Dataset 3 | 210 / 90 | 1.8945 | 0.7000 |

**Observation:** Model 3 performs best on Datasets 1 and 2.

## Test phase

For the live test phase, the project uses **Model 1** for prediction. The system:

1. detects the hand in the camera frame,
2. generates a grayscale probability image,
3. reshapes the processed image for model input,
4. loads the trained model,
5. predicts the performed letter,
6. overlays the prediction on the video stream.

## Technologies used

- Python
- OpenCV
- NumPy
- Matplotlib
- TensorFlow / Keras
- Google Colab

## Notes on reproducibility

This project was originally developed in **Google Colab** and includes Colab-specific components such as:

- camera capture through browser-side JavaScript,
- Google Drive mounting,
- Colab utility imports.

Because of this, the notebook is best understood as a documented academic project and prototype rather than a packaged, fully reproducible local application.

## Limitations

- The implementation is tightly coupled to the Google Colab environment.
- Only three gesture classes are considered: **M**, **N**, and **W**.
- The dataset is relatively small and tailored to the project experiment.
- The repository is focused on demonstrating the pipeline and results rather than production deployment.

## Author

**Ilaria Enache**  
Computer Vision and Machine Learning Lab project
