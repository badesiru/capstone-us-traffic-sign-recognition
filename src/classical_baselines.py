#this trains SVM, Random forest, and KNN, and we wanted to evaluate models
#displays accuracy, classification report, and confusion matrix
import os
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

BASE_DIR = os.path.dirname(os.path.dirname(__file__)) 
#loading the feature data
train_data = np.load(os.path.join(BASE_DIR, "traffic_light_features_train.npz"))
val_data   = np.load(os.path.join(BASE_DIR, "traffic_light_features_val.npz"))
X_train, y_train = train_data["X"], train_data["y"]
X_val, y_val     = val_data["X"], val_data["y"]


print(f"Training samples -  {X_train.shape[0]}, Validation samples - {X_val.shape[0]}")
print(f"Feature vector length - {X_train.shape[1]}\n")




#SVM
svm_model = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)

#Random Forrest
rf_model = RandomForestClassifier(n_estimators=150, random_state=42)

#KNN
knn_model = KNeighborsClassifier(n_neighbors=5)


#training the models
models = {
    "SVM": svm_model,
    "Random Forest": rf_model,
    "KNN": knn_model
}

results = {}

for name, model in models.items():
    print(f"Training:{name}")
    model.fit(X_train, y_train)
    
    #make predictions
    y_pred = model.predict(X_val)
    
    #compute accuracy and store
    acc = accuracy_score(y_val, y_pred)
    results[name] = acc
    
    print(f"\n{name} Results")
    print(f"Accuracy:{acc:.4f}")
    print("\nClassification report:")
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
print("\nSummary")
for name, acc in results.items():
    print(f"{name}: {acc:.4f}")
#saving to csv
results_df = pd.DataFrame([
    {"model": "SVM", "accuracy": results["SVM"]},
    {"model": "RF", "accuracy": results["Random Forest"]},
    {"model": "KNN", "accuracy": results["KNN"]},
])

results_df.to_csv("traffic_light_baseline_results.csv", index=False)
print("\nsaved traffic_light_baseline_results.csv")