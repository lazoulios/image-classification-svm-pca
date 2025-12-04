import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import utils
from keras.datasets import cifar10
from sklearn.metrics import confusion_matrix

print("Loading data for visualization...")
(_, _), (x_test_orig, y_test) = cifar10.load_data()
y_test = y_test.ravel()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

scaler = utils.load_model('cifar10_scaler.pkl')
pca = utils.load_model('cifar10_pca.pkl')

if scaler is None or pca is None:
    print("ERROR: Scaler/PCA files not found. Please run training first!")
    exit()

x_test_flat = x_test_orig.reshape(x_test_orig.shape[0], -1).astype('float32')
x_test_scaled = scaler.transform(x_test_flat)
x_test_pca = pca.transform(x_test_scaled)

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {title}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

def show_examples(model, x_pca, x_orig, y_true, title, num_samples=5):
    y_pred = model.predict(x_pca)
    
    correct_indices = np.where(y_pred == y_true)[0]
    incorrect_indices = np.where(y_pred != y_true)[0]
    
    plt.figure(figsize=(15, 6))
    plt.suptitle(f'{title}: Correct (Top) vs Incorrect (Bottom)', fontsize=16)

    for i in range(num_samples):
        if i < len(correct_indices):
            idx = correct_indices[i]
            plt.subplot(2, num_samples, i + 1)
            plt.imshow(x_orig[idx])
            plt.title(f"True: {class_names[y_true[idx]]}\nPred: {class_names[y_pred[idx]]}", color="green", fontsize=10)
            plt.axis('off')

    for i in range(num_samples):
        if i < len(incorrect_indices):
            idx = incorrect_indices[i]
            plt.subplot(2, num_samples, num_samples + i + 1)
            plt.imshow(x_orig[idx])
            plt.title(f"True: {class_names[y_true[idx]]}\nPred: {class_names[y_pred[idx]]}", color="red", fontsize=10)
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    return y_pred

models_to_visualize = [
    ("SVM (RBF Kernel)", "svm_rbf_cifar10.pkl"),
    ("SVM (Linear Kernel)", "svm_linear_cifar10.pkl")
]

for model_name, filename in models_to_visualize:
    print(f"\nLoading model: {filename} ({model_name})...")
    model = utils.load_model(filename)

    if model:
        print(f"Creating images for {model_name}...")
        y_pred = show_examples(model, x_test_pca, x_test_orig, y_test, model_name)
        plot_confusion_matrix(y_test, y_pred, model_name)
    else:
        print(f"Model {filename} not found.")