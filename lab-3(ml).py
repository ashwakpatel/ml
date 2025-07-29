import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
import os

# --- Function Definitions (Modularized Code) ---

# A1: Function to evaluate class separation
def evaluate_class_separation(class_a_features, class_b_features):
    """
    Calculates and returns the centroids, spreads, and interclass distance.
    [cite: 21]
    """
    # Calculate centroids (mean vectors) for each class [cite: 23]
    centroid_a = np.mean(class_a_features, axis=0)
    centroid_b = np.mean(class_b_features, axis=0)

    # Calculate spread (standard deviation) for each class [cite: 25]
    spread_a = np.std(class_a_features, axis=0)
    spread_b = np.std(class_b_features, axis=0)

    # Calculate the Euclidean distance between the centroids [cite: 27]
    interclass_dist = np.linalg.norm(centroid_a - centroid_b)

    return centroid_a, centroid_b, spread_a, spread_b, interclass_dist

# A2: Function to analyze the density pattern of a feature
def analyze_feature_density(feature_vector):
    """
    Calculates histogram data, mean, and variance for a given feature.
    [cite: 33, 34]
    """
    # Calculate histogram data using 5 bins
    hist_data, bin_edges = np.histogram(feature_vector, bins=5)

    # Calculate mean and variance
    mean_val = np.mean(feature_vector)
    variance_val = np.var(feature_vector)

    return hist_data, bin_edges, mean_val, variance_val

# A3: Function to calculate a series of Minkowski distances
def calculate_minkowski_series(vector1, vector2, max_r):
    """
    Calculates Minkowski distance for r from 1 to max_r.
    [cite: 36]
    """
    distances = []
    r_values = range(1, max_r + 1)
    for r in r_values:
        # Calculate distance using numpy's norm function with a specified order 'r'
        distance = np.linalg.norm(vector1 - vector2, ord=r)
        distances.append(distance)
    return list(r_values), distances

# A4: Function to split the dataset
def split_data(feature_matrix, target_vector, test_size=0.3, random_state=42):
    """
    Splits the dataset into training and testing sets using scikit-learn.
    [cite: 37, 41]
    """
    X_train, X_test, y_train, y_test = train_test_split(
        feature_matrix, target_vector, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

# A5: Function to train a k-NN classifier
def train_knn(features_train, labels_train, n_neighbors=3):
    """
    Trains a KNeighborsClassifier model.
    [cite: 45]
    """
    # Initialize the classifier with the specified number of neighbors
    classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    # Fit the model to the training data
    classifier.fit(features_train, labels_train)
    return classifier

# A6: Function to evaluate the accuracy of a classifier
def evaluate_accuracy(classifier, features_test, labels_test):
    """
    Calculates the accuracy of a trained classifier on the test set.
    [cite: 49, 50]
    """
    accuracy = classifier.score(features_test, labels_test)
    return accuracy

# A7: Functions for prediction
def get_predictions(classifier, features_test):
    """
    Gets predictions for a set of test vectors.
    [cite: 52]
    """
    return classifier.predict(features_test)

def predict_single_vector(classifier, vector):
    """
    Predicts the class for a single feature vector.
    [cite: 53]
    """
    # Reshape the vector to be a 2D array as required by predict
    return classifier.predict(vector.reshape(1, -1))

# A8: Function to find accuracies for a range of k values
def get_knn_accuracy_for_k_range(features_train, labels_train, features_test, labels_test, max_k):
    """
    Trains k-NN for k=1 to max_k and returns the accuracies.
    [cite: 54]
    """
    k_values = range(1, max_k + 1)
    accuracies = []

    for k in k_values:
        # Train a new classifier for each value of k
        classifier = KNeighborsClassifier(n_neighbors=k)
        classifier.fit(features_train, labels_train)
        # Evaluate its accuracy and store it
        accuracy = classifier.score(features_test, labels_test)
        accuracies.append(accuracy)

    return list(k_values), accuracies

# A9: Function to get comprehensive performance metrics
def evaluate_performance(classifier, features_train, labels_train, features_test, labels_test):
    """
    Generates confusion matrix and classification reports for train and test data.
    [cite: 55]
    """
    # Get predictions for both training and testing data
    y_train_pred = classifier.predict(features_train)
    y_test_pred = classifier.predict(features_test)

    # Calculate confusion matrices
    cm_train = confusion_matrix(labels_train, y_train_pred)
    cm_test = confusion_matrix(labels_test, y_test_pred)

    # Generate classification reports which include precision, recall, and F1-score
    report_train = classification_report(labels_train, y_train_pred, zero_division=0)
    report_test = classification_report(labels_test, y_test_pred, zero_division=0)

    return cm_train, report_train, cm_test, report_test


# --- Main Program (Execution Block) ---
if __name__ == '__main__':
    # Setup: Create and save the dataset to a CSV file
    data = {
        'feature1': [2.9, 3.5, 1.8, 4.1, 2.5, 4.5, 2.2, 3.8, 3.1, 2.7, 8.1, 9.2, 7.5, 6.9, 8.8, 6.5, 9.5, 7.2, 8.4, 7.8],
        'feature2': [6.7, 5.5, 8.2, 4.9, 7.1, 4.5, 7.8, 5.2, 6.3, 7.5, 2.1, 1.5, 3.2, 2.8, 1.9, 3.5, 1.1, 2.5, 2.3, 2.9],
        'class': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    }
    df = pd.DataFrame(data)
    csv_file_path = 'classification_data.csv'
    df.to_csv(csv_file_path, index=False)
    
    # Load data from the CSV file
    loaded_df = pd.read_csv(csv_file_path)
    X = loaded_df[['feature1', 'feature2']].values
    y = loaded_df['class'].values

    # --- A1: Intraclass Spread and Interclass Distance ---
    print("--- A1: Class Separation Analysis ---")
    class_0_features = loaded_df[loaded_df['class'] == 0][['feature1', 'feature2']].values
    class_1_features = loaded_df[loaded_df['class'] == 1][['feature1', 'feature2']].values
    c0, c1, s0, s1, dist = evaluate_class_separation(class_0_features, class_1_features)
    print(f"Class 0 Centroid: {c0}")
    print(f"Class 1 Centroid: {c1}\n")
    print(f"Class 0 Spread (Std Dev): {s0}")
    print(f"Class 1 Spread (Std Dev): {s1}\n")
    print(f"Distance between Class Centroids: {dist:.2f}")
    print("-" * 40 + "\n")

    # --- A2: Feature Histogram and Density Pattern ---
    print("--- A2: Feature Density Analysis for 'feature1' ---")
    feature1_vec = loaded_df['feature1'].values
    hist_counts, bin_edges, mean_f1, var_f1 = analyze_feature_density(feature1_vec)
    print(f"Histogram Data (Counts per bin): {hist_counts}")
    print(f"Bin Edges: {bin_edges}")
    print(f"Mean: {mean_f1:.2f}")
    print(f"Variance: {var_f1:.2f}\n")
    # Plot the histogram
    plt.figure(figsize=(8, 6))
    plt.hist(feature1_vec, bins=bin_edges, edgecolor='black', alpha=0.7)
    plt.title('A2: Histogram of Feature 1')
    plt.xlabel('Feature 1 Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
    print("-" * 40 + "\n")

    # --- A3: Minkowski Distance Plot ---
    print("--- A3: Minkowski Distance Analysis ---")
    vec_a = X[0]  # First data point
    vec_b = X[10] # Eleventh data point
    r_vals, minkowski_dists = calculate_minkowski_series(vec_a, vec_b, 10)
    print(f"Comparing Vector 1: {vec_a} and Vector 2: {vec_b}")
    for r, d in zip(r_vals, minkowski_dists):
        print(f"Minkowski distance for r={r}: {d:.2f}")
    # Plot the distances
    plt.figure(figsize=(8, 6))
    plt.plot(r_vals, minkowski_dists, marker='o', linestyle='--')
    plt.title('A3: Minkowski Distance vs. r-value')
    plt.xlabel('r value')
    plt.ylabel('Minkowski Distance')
    plt.grid(True)
    plt.xticks(r_vals)
    plt.show()
    print("-" * 40 + "\n")

    # --- A4: Split Dataset ---
    print("--- A4: Splitting Dataset ---")
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Total samples: {len(X)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print("-" * 40 + "\n")

    # --- A5: Train k-NN Classifier (k=3) ---
    print("--- A5: Training k-NN Classifier (k=3) ---")
    knn_model_k3 = train_knn(X_train, y_train, n_neighbors=3)
    print("Classifier trained successfully!")
    print(f"Model details: {knn_model_k3}")
    print("-" * 40 + "\n")

    # --- A6: Test Accuracy of the k-NN Classifier ---
    print("--- A6: Testing k-NN Accuracy (k=3) ---")
    accuracy_k3 = evaluate_accuracy(knn_model_k3, X_test, y_test)
    print(f"Accuracy on the test set for k=3: {accuracy_k3:.2f}")
    print("-" * 40 + "\n")

    # --- A7: Use the predict() Function ---
    print("--- A7: Prediction Behavior Analysis ---")
    all_preds = get_predictions(knn_model_k3, X_test)
    print(f"Predictions for the entire test set: {all_preds}")
    print(f"Actual labels for the test set:   {y_test}\n")
    test_vector = X_test[0]
    single_pred = predict_single_vector(knn_model_k3, test_vector)
    print(f"Test vector: {test_vector}")
    print(f"Predicted class for the single vector: {single_pred[0]}")
    print(f"Actual class for the single vector: {y_test[0]}")
    print("-" * 40 + "\n")
    
    # --- A8: Vary k and Plot Accuracy ---
    print("--- A8: Accuracy vs. k Analysis ---")
    k_values, accuracy_scores = get_knn_accuracy_for_k_range(X_train, y_train, X_test, y_test, 11)
    for k, acc in zip(k_values, accuracy_scores):
        print(f"k = {k}, Accuracy = {acc:.2f}")
    # Plot the accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracy_scores, marker='o', linestyle='-', color='b')
    plt.title('A8: k-NN Test Accuracy vs. Number of Neighbors (k)')
    plt.xlabel('k (Number of Neighbors)')
    plt.ylabel('Accuracy')
    plt.xticks(k_values)
    plt.grid(True)
    plt.show()
    print("-" * 40 + "\n")

    # --- A9: Confusion Matrix and Performance Metrics ---
    print("--- A9: Performance Evaluation (k=3) ---")
    cm_train, report_train, cm_test, report_test = evaluate_performance(knn_model_k3, X_train, y_train, X_test, y_test)
    
    print("\n--- Training Data Performance ---")
    print("Confusion Matrix:")
    print(cm_train)
    print("\nClassification Report:")
    print(report_train)
    
    print("\n--- Test Data Performance ---")
    print("Confusion Matrix:")
    print(cm_test)
    print("\nClassification Report:")
    print(report_test)
    
    # Infer the model's learning outcome [cite: 56]
    train_accuracy = evaluate_accuracy(knn_model_k3, X_train, y_train)
    test_accuracy = accuracy_k3 # Already calculated in A6
    print("\n--- Model Fit Inference ---")
    print(f"Training Accuracy: {train_accuracy:.2f}")
    print(f"Test Accuracy: {test_accuracy:.2f}")
    
    # A difference of >0.1 might suggest overfitting, but this is a heuristic
    if train_accuracy > test_accuracy and (train_accuracy - test_accuracy) > 0.1:
        print("Inference: The model might be OVERFITTING.")
    elif train_accuracy < 0.8 and test_accuracy < 0.8:
        print("Inference: The model might be UNDERFITTING.")
    else:
        print("Inference: The model appears to have a REGULAR FIT (Good Fit).")
    print("-" * 40 + "\n")
    
    # Clean up the created CSV file
    if os.path.exists(csv_file_path):
        os.remove(csv_file_path)
        print(f"Cleaned up the temporary file: {csv_file_path}")