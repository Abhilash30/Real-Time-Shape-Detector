import pandas as pd

df = pd.read_csv("data/labeled.csv", header = None, names = ["area", "perimeter", "vertices", "aspect_ratio", "circularity", "label"])
