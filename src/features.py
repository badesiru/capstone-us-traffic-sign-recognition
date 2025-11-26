import cv2
import numpy as np
from skimage.feature import hog
from skimage import color
import os

#Extracts color histogram features from an RGB image
def extract_color_hist(image, bins=(8, 8, 8)):

    #Convert image to RGB if it's not already
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #Compute the color histogram across all 3 channels
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])

    #Normalize and flatten
    hist = cv2.normalize(hist, hist).flatten()
    return hist


#Extracts HOG features from the image
def extract_hog_features(image, resize=(64, 64)):

    #Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, resize)

    #Computing HOG features
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        transform_sqrt=True,
        feature_vector=True
    )
    return features


#Extracts features from HOG and color histograms and combines them
def extract_features(image_path):

    image = cv2.imread(image_path)

    #Edge case if image is not read properly
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    #Extracting color hist features and HOG features
    color_hist = extract_color_hist(image)
    hog_feats = extract_hog_features(image)

    #Combine both feature types into one vector
    combined = np.hstack([color_hist, hog_feats])
    return combined

#Goes through each dataset folder and extracts features and labels
def extract_dataset_features(image_folder):
    X, y = [], []
    for label in os.listdir(image_folder):      #Looping over eahc subfolder
        label_path = os.path.join(image_folder, label)
        if not os.path.isdir(label_path):       #Skip if not a directory
            continue
        for file in os.listdir(label_path):
            img_path = os.path.join(label_path, file)
            if not (file.lower().endswith(".jpg") or file.lower().endswith(".png")): #Skip non-image files
                continue
            try:
                features = extract_features(img_path)   #Extract features
                X.append(features)
                y.append(label)
            except Exception as e:
                print(f"[WARN] Skipping {img_path}: {e}")
    return np.array(X), np.array(y)     #Convert lists to numpy arrays for model training


if __name__ == "__main__":
    train_dir = "C:/Users/eshaa/MLProject/capstone-us-traffic-sign-recognition/lisa_dataset/lisa/train"
    val_dir   = "C:/Users/eshaa/MLProject/capstone-us-traffic-sign-recognition/lisa_dataset/lisa/val"

    print("Extracting TRAIN features...")
    X_train, y_train = extract_dataset_features(train_dir)
    print(f"Train features: {X_train.shape}, Labels: {len(y_train)}")

    print("Extracting VALIDATION features...")
    X_val, y_val = extract_dataset_features(val_dir)
    print(f"Validation features: {X_val.shape}, Labels: {len(y_val)}")

    
    np.savez("traffic_light_features_train.npz", X=X_train, y=y_train)
    np.savez("traffic_light_features_val.npz", X=X_val, y=y_val)

    print("Saved features to:")
    print(" - traffic_light_features_train.npz")
    print(" - traffic_light_features_val.npz")
