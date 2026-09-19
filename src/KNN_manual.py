import math
from collections import Counter

def euclidean_distance(x1, x2):
    total = 0

    for a, b in zip(x1, x2):
        total += (a - b) ** 2

    return math.sqrt(total)

def predict_knn(new_sample, X_train, y_train, k = 3):
    distance=[]
    for features, label in zip(X_train, y_train):
        d = euclidean_distance(new_sample, features)
        distance.append((d,label))
    distance.sort(key=lambda item: item[0])
    top_k_labels = [item[1] for item in distance[:k]]
    most_common = Counter(top_k_labels).most_common(1)
    return most_common[0][0]
    


X_train = [
    [1, 2],
    [2, 3],
    [8, 9],
    [9, 8]
]

y_train = [
    "triangle",
    "triangle",
    "circle",
    "circle"
]


new_sample = [2, 2]

print("Python Prediction:", predict_knn(new_sample, X_train, y_train, k=3))