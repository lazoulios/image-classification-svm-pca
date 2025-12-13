
# Support Vector Machines for CIFAR-10 Classification

The goal of this repository is to implement and compare **Support Vector Machines (SVM)** with different kernels (Linear vs. RBF) against a baseline **Multi-Layer Perceptron (MLP)** for multi-class image classification on the CIFAR-10 dataset.

The project utilizes **Principal Component Analysis (PCA)** for dimensionality reduction to make SVM training computationally feasible.

## 📊 Key Results

The RBF Kernel proved superior to the Linear Kernel, demonstrating that the image data is not linearly separable. The SVM (RBF) also slightly outperformed the MLP baseline on the reduced feature set.

| Architecture | Kernel / Config | Test Accuracy | Comments |
| :--- | :--- | :--- | :--- |
| **SVM (RBF)** | Non-Linear (RBF) | **43.91%** | Optimal performance; handles non-linearity well. |
| **MLP** | 1 Hidden Layer | 40.03% | Very fast training; competitive accuracy. |
| **SVM (Linear)** | Linear | 35.48% | Failed to converge efficiently; poor separability. |

-----

## 🚀 Project Setup and Execution

### 1. Requirements

The project uses Python 3.x and relies on the following major libraries:

* `Scikit-learn` (for SVM, MLP, PCA, and Scaling)
* `TensorFlow / Keras` (only for loading the CIFAR-10 dataset)
* `Joblib` (for saving/loading large models)
* `Matplotlib / Seaborn` (for plotting confusion matrices)

### 2. Installation

Clone the repository and set up a virtual environment:

```bash
# Clone the repository
git clone [https://github.com/lazoulios/neural-network-image-recognition]
cd neural-network-image-recognition

# Create and activate the virtual environment
python -m venv venv
source venv/bin/activate  # Use venv\Scripts\Activate.ps1 on Windows PowerShell

# Install dependencies from requirements.txt
pip install -r requirements.txt
```

### 3\. Running the Models

The repository contains two main scripts. Both scripts use a **"Load or Fit"** logic: they check if the trained model exists in the `models/` folder; if so, they load it instantly; otherwise, they train it from scratch and save the model.

#### A. Training and Evaluation

Runs the training process for SVM (RBF), SVM (Linear), and MLP. It applies PCA (90% variance) and prints the classification reports.

```bash
python main.py
```

#### B. Visualization

Generates visual evidence, including **Confusion Matrices** and **Correct/Incorrect prediction examples** for the trained models.

```bash
python visualize.py
```

-----

## 📁 Repository Structure

  * `main.py`: Main script for training the SVM and MLP models.
  * `visualize.py`: Script for generating plots and visual examples.
  * `utils.py`: Contains helper functions (loading data, saving models).
  * `models/`: Directory where trained `.pkl` models are saved (ignored by Git).
  * `tex/`: Source code for the LaTeX report.
  * `report.pdf`: The final compiled project report.
  * `requirements.txt`: List of all required Python packages.