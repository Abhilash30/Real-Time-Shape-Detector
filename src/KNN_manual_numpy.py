import numpy as np
from dataframe import df
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

feature_vector = ["area", "perimeter", "vertices", "aspect_ratio", "circularity"]



X = df[feature_vector]

y = df["label"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Manual split has an accuracy of 98.16%
# rng = np.random.default_rng(42) 
# indices = rng.permutation(len(X))
# rng = np.random.default_rng(42)
# indices = rng.permutation(len(X))

# split = int(0.8 * len(X))
# train_idx = indices[:split]
# test_idx = indices[split:]      

# X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
# y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]


# means = X.mean()
# stds = X.std()

# # X_standardized = (X - means) / stds replaceing with 80% training data
# #rng.permutation(len(X)) shuffles the row numbers 0..n-1. You split those numbers rather than the data itself, so the same indices can pick out matching rows of X and y.
# #.iloc[train_idx] selects rows by position. This keeps each sample paired with its label, since both use the same indices.

# X_train = ((X_train - means) / stds).to_numpy()
# X_test = ((X_test - means) / stds).to_numpy()
# y_train = y_train.to_numpy()
# y_test = y_test.to_numpy()


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


predictions = [predict_knn_numpy(sample, X_train, y_train, k=3) for sample in X_test]

print("Accuracy:", accuracy_score(y_test, predictions))