# Resistor Classification and Segmentation Model

## Overview
This repository contains a set of models and utilities for **resistor classification** and **image segmentation**. The models are based on Convolutional Neural Networks (CNN) for classification and a U-Net architecture for segmentation, originally sourced from [BradyWynn's GitHub](https://github.com/BradyWynn). This project builds upon BradyWynn’s work with several key improvements aimed at enhancing model performance, flexibility, and usability.

This is an update to the submisson made on [AI Community Internal Kaggle Competition](https://www.kaggle.com/competitions/ai-community-internal-comp) which theoretically should break the 90% accuracy mark.

## Improvements and Modifications

### 1. **Model Architecture Enhancements**:
   - **Original**: BradyWynn's initial implementation of the models used a basic CNN for classification and U-Net for segmentation.
   - **Improvement**: 
     - **Additional Layers and Filters**: In the CNN model, we introduced additional layers and filters to increase the depth and capacity of the network. This allows the model to better extract complex features from images, improving classification accuracy.
     - **Batch Normalization**: We added **batch normalization** after each convolutional layer. This helps stabilize the training process by normalizing activations, leading to faster convergence and better generalization.
     - **Dropout**: To prevent overfitting, **dropout** layers were added in the CNN model. This helps improve the model's ability to generalize by randomly disabling neurons during training.

   #### **Why?**
   - By adding more layers and filters, we aim to increase the capacity of the model to capture intricate patterns from the images.
   - Batch normalization improves the stability of the network during training, while dropout helps avoid overfitting, making the model more robust when deployed.

### 2. **Improved U-Net Model**:
   - **Original**: The original U-Net model had a basic encoder-decoder structure for segmentation tasks.
   - **Improvement**: 
     - We made the U-Net more flexible by allowing the number of filters, kernel sizes, and depth of the network to be passed as arguments. This allows for greater customization based on the complexity of the segmentation task.
     - **Residual Connections** were added to improve the flow of gradients through the network. This prevents vanishing gradients and improves the performance of deeper networks.
     - **Leaky ReLU** activations were introduced in place of traditional ReLU to help with the "dying ReLU" problem, where neurons stop learning entirely.

   #### **Why?**
   - The flexibility of hyperparameters helps users experiment with different configurations based on their specific needs. It makes the model more adaptable to different image segmentation tasks.
   - Residual connections help to address the problem of gradient vanishing and exploding, making it easier to train deeper networks.
   - Leaky ReLU prevents the issue of inactive neurons, ensuring that all neurons continue to learn throughout training.

### 3. **Improved Data Handling and Mask Visualization**:
   - **Original**: The original code used basic methods for loading and visualizing mask files.
   - **Improvement**: 
     - We switched from `os.listdir()` to `glob.glob()` for more flexible file loading. This ensures that only valid files (e.g., `.npy` files) are loaded and simplifies the code.
     - **Visualization Improvements**: Enhanced the visualization of average masks by using `matplotlib` to display the final mask with colorbars and better labels for clearer results.

   #### **Why?**
   - `glob` is a more efficient and flexible way to handle file loading with patterns, as it ensures that only the correct files are retrieved (e.g., mask files with specific extensions).
   - Improved visualization helps users better interpret results, especially in tasks like segmentation, where visual feedback is crucial for debugging and fine-tuning the model.

### 4. **Model Initialization and Weights**:
   - **Original**: The original code used Xavier initialization for convolutional layers, but linear layers were not consistently initialized.
   - **Improvement**:
     - We applied **Xavier initialization** across both convolutional and fully connected layers. This ensures that the weights are better scaled to avoid issues with vanishing/exploding gradients, especially for deeper networks.

   #### **Why?**
   - Proper weight initialization is key to efficient training. Xavier initialization is well-suited for models with ReLU activations, as it ensures that the variance of the outputs across layers stays relatively stable, leading to faster convergence.

### 5. **Code Modularity and Readability**:
   - **Original**: The code used a more monolithic approach with all layers defined within a single class.
   - **Improvement**:
     - We refactored the code into more modular components. Layers, activation functions, and initialization were split into separate methods or classes where appropriate. This improves code readability and maintainability.

   #### **Why?**
   - Refactoring the code into smaller, reusable components makes it easier to modify, debug, and extend. It enhances collaboration by allowing others to contribute without being overwhelmed by a monolithic code structure.

---

## Dataset and Reproduction Instructions

### 1. **Dataset**:
   The image dataset used in this project is too large to include directly in the repository. To reproduce the results, you must download the dataset from the [Kaggle AI Community Internal Competition](https://www.kaggle.com/competitions/ai-community-internal-comp).

   Ensure you have the **test** and **train** folders from the dataset downloaded into your working directory. Additionally, **delete the `train.csv` file** inside the `train` folder before running the code.

### 2. **Model Training**:
   There are **two separate models** that need to be trained:

   - **U-Net for Image Segmentation**
   - **CNN for Resistor Classification**

   The **U-Net class** used in this project was sourced from [jaxony/unet-pytorch](https://github.com/jaxony/unet-pytorch/blob/master/model.py). The CNN model was built specifically for resistor classification tasks.

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

pip install scikit-learn==1.0.2
