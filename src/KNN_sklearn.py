import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import os
import joblib
from dataframe import df

os.makedirs("models", exist_ok=True)

feature_vector = [
    "area",
    "perimeter",
    "vertices",
    "aspect_ratio",
    "circularity"
]

X = df[feature_vector]
y = df["label"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



scaler = StandardScaler() 

X_train_scaled = scaler.fit_transform(X_train) #Attaches numerical feature distribution in dataset 
X_test_scaled = scaler.transform(X_test)


knn = KNeighborsClassifier(n_neighbors=3)


knn.fit(X_train_scaled, y_train)

# joblib.dump(knn, "models/knn.pkl") USED TO SAVE SCALER AND MODEL
# joblib.dump(scaler, "models/scaler.pkl")

predictions = knn.predict(X_test_scaled)


accuracy = accuracy_score(y_test, predictions)

print("Predictions:", predictions)
print("Actual:", y_test.to_numpy())


print("Accuracy:", accuracy)