import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns


#Loading feature data
train_data = np.load("traffic_light_features_train.npz")
val_data   = np.load("traffic_light_features_val.npz")

X_train, y_train = train_data["X"], train_data["y"]
X_val, y_val     = val_data["X"], val_data["y"]

print("Data loaded successfully!")
print(f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")
print(f"Feature vector length: {X_train.shape[1]}\n")


#Uncomment to train each model

#SVM
svm_model = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)

#Random Forrest
rf_model = RandomForestClassifier(n_estimators=150, random_state=42)

#KNN
knn_model = KNeighborsClassifier(n_neighbors=5)


#Training the models
models = {
    "SVM": svm_model,
    "Random Forest": rf_model,
    "KNN": knn_model
}

results = {}

for name, model in models.items():
    print(f"[INFO] Training {name}...")
    model.fit(X_train, y_train)
    
    #Make predictions
    y_pred = model.predict(X_val)
    
    #Compute accuracy and store
    acc = accuracy_score(y_val, y_pred)
    results[name] = acc
    
    print(f"\n--- {name} RESULTS ---")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))
    
    # Confusion matrix visualization
    cm = confusion_matrix(y_val, y_pred, labels=np.unique(y_train))
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d',
                xticklabels=np.unique(y_train),
                yticklabels=np.unique(y_train))
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()


#Prining summary of results
print("\n==================== Summary ====================")
for name, acc in results.items():
    print(f"{name}: {acc:.4f}")
print("=================================================\n")
