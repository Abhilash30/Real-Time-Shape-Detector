import csv
import os


DATASET_PATH = "data/shape_features.csv"

HEADERS = [
    "area",
    "perimeter",
    "vertices",
    "aspect_ratio",
    "circularity",
    "label"
]


def create_dataset():
    os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)

    if not os.path.exists(DATASET_PATH):
        with open(DATASET_PATH, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(HEADERS)


def save_sample(feature_vector, label):
    create_dataset()

    with open(DATASET_PATH, "a", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            *feature_vector,
            label
        ])