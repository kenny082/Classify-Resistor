# Resistor Classification and Segmentation Model

## Overview
This repository contains a set of models and utilities for **resistor classification** and **image segmentation**. The models are based on Convolutional Neural Networks (CNN) for classification and a U-Net architecture for segmentation, originally sourced from [BradyWynn's GitHub](https://github.com/BradyWynn). This project builds upon BradyWynn’s work with several key improvements aimed at enhancing model performance, flexibility, and usability.

This is an update to the submission made on [AI Community Internal Kaggle Competition](https://www.kaggle.com/competitions/ai-community-internal-comp) which improve classification accuracy.

## Dataset and Reproduction Instructions

### 1. **Dataset**:
   The image dataset used in this project is too large to include directly in the repository. To reproduce the results, you must download the dataset from the [Kaggle AI Community Internal Competition](https://www.kaggle.com/competitions/ai-community-internal-comp).

   Ensure you have the **test** and **train** folders from the dataset downloaded into your working directory. Additionally, **delete the `train.csv` file** inside the `train` folder before running the code.

### 2. **Model Training**:
   There are **two separate models** that need to be trained:

   - **U-Net for Image Segmentation**
   - **CNN for Resistor Classification**

   The **U-Net class** used in this project was sourced from [jaxony/unet-pytorch](https://github.com/jaxony/unet-pytorch/blob/master/model.py). The CNN model was built specifically for resistor classification tasks within Stony Brook University Engineering Department. The model sole purpose is for the identification of resistors within the university's engineering department. To reiterate, this is an improvement on BraddWynn's code.

### 3. **Training Time**:
   - On an **RX 6600** GPU:
     - **U-Net** took **67 minutes** to train.
     - **CNN** took **20 seconds** to train.
   - The training time may vary based on your hardware. If you are using a weaker GPU or training on CPU, expect longer training times.

### 4. **Submission File Generation**:
   The **`test.ipynb`** notebook is used to generate the submission file. Both the **U-Net** and **CNN** models need to be trained and saved inside the `/models` folder before running the notebook.

---

## How to Use

1. **Clone the repository** using the `git clone` command.
2. **Install dependencies** using `pip install name_of_package`.
3. **Download the dataset** from Kaggle and place it in the correct directory structure.
4. **Train the U-Net and CNN models** by running the appropriate scripts (`unet_train.py`, `cnn_train.py`).
5. **Run `test.ipynb`** to generate the competition submission file.

Packages:
pip install torch==1.13.0

pip install numpy==1.23.1

pip install matplotlib==3.5.1

pip install opencv-python==4.5.3
