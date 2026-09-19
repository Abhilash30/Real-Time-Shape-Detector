import numpy as np
from dataframe import df

feature_vector = ["area", "perimeter", "vertices", "aspect_ratio", "circularity"]



X = df[feature_vector]

y = df["label"]

means = X.mean()
stds = X.std()

X_standardized = (X - means) / stds



def predict_knn_numpy(new_sample, X_train, y_train, k=3):
    X_mat = np.array(X_train)
    y_arr = np.array(y_train)
    sample = np.array(new_sample)
    

    distances = np.sqrt(np.sum((X_mat - sample) ** 2, axis=1))
    
    # 2. Sort indices by distance (ascending)
    sorted_indices = np.argsort(distances)
    
    # 3. Extract the labels of the K nearest neighbors
    top_k_labels = y_arr[sorted_indices[:k]]
    

    labels, counts = np.unique(top_k_labels, return_counts=True) #Gets the labels based on the frequency
    

    return labels[np.argmax(counts)]


X_train = X_standardized.to_numpy()
y_train = y.to_numpy()

new_sample = X_train[2]

print("NumPy Prediction:", predict_knn_numpy(new_sample, X_train, y_train, k=3))

