import time
import numpy as np
import utils
from keras.datasets import cifar10
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

print("\nLoading CIFAR-10 dataset...")
(x_train_full, y_train_full), (x_test, y_test) = cifar10.load_data()

print("Reducing training set to 5000 samples...")
x_train_full, _, y_train_full, _ = train_test_split(
    x_train_full, y_train_full, train_size=5000, stratify=y_train_full, random_state=42
)

y_train = y_train_full.ravel()
y_test = y_test.ravel()

x_train_flat = x_train_full.reshape(x_train_full.shape[0], -1).astype('float32')
x_test_flat = x_test.reshape(x_test.shape[0], -1).astype('float32')

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train_flat)
x_test_scaled = scaler.transform(x_test_flat)

print("Checking for existing PCA model...")
pca = utils.load_model('cifar10_pca.pkl')

if pca is None:
    print("Training PCA on full dataset (this may take a while)...")
    pca = PCA(n_components=0.90, random_state=42)
    x_train_pca = pca.fit_transform(x_train_scaled)
    utils.save_model(pca, 'cifar10_pca.pkl')
else:
    print("Existing PCA found and loaded. Transforming data...")
    x_train_pca = pca.transform(x_train_scaled)

x_test_pca = pca.transform(x_test_scaled)
print(f"Retained {x_train_pca.shape[1]} components.")
utils.save_model(scaler, 'cifar10_scaler.pkl')

def train_and_evaluate(model, model_name, filename):
    print(f"\n--- Checking for {model_name} ---")
    
    clf = utils.load_model(filename)
    
    if clf:
        print("Model loaded! Proceeding to evaluation...")
    else:
        print(f"Starting training for {model_name} (Please wait!)...")
        clf = model
        start_time = time.time()
        
        clf.fit(x_train_pca, y_train)
        
        end_time = time.time()
        duration = end_time - start_time
        print(f"Training completed in {duration:.2f} seconds.")
        
        utils.save_model(clf, filename)

    print("Calculating accuracy on Test set...")
    start_test = time.time()
    y_test_pred = clf.predict(x_test_pca)
    test_time = time.time() - start_test
    
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"Testing Accuracy: {test_acc:.4f}")
    print(f"Prediction Time (Test): {test_time:.2f} sec")
    
    print(classification_report(y_test, y_test_pred))
    
    return test_acc

# SVM RBF
svm_rbf = SVC(kernel='rbf', C=10, gamma='scale', cache_size=1000)
train_and_evaluate(svm_rbf, "SVM (RBF Kernel)", "svm_rbf_cifar10.pkl")

# SVM Linear
svm_linear = SVC(kernel='linear', C=1, cache_size=1000)
train_and_evaluate(svm_linear, "SVM (Linear Kernel)", "svm_linear_cifar10.pkl")

# MLP
mlp = MLPClassifier(hidden_layer_sizes=(256,), activation='relu', solver='adam', 
                    max_iter=300, random_state=42)
train_and_evaluate(mlp, "MLP (1 Hidden Layer)", "mlp_cifar10.pkl")

print("\nThe process is complete")