#using rbg color histograms and HOG, feautres, 
#those features used to train svm, ranodm forest and knn
import cv2
import numpy as np
from skimage.feature import hog
from skimage import color
import os

#extracts color histogram feats from RGB img
def extract_color_hist(image, bins=(8, 8, 8)):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


#extracts HOG features from the image
def extract_hog_features(image, resize=(64, 64)):


    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, resize)

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


#extracts features from HOG and color histograms and combines them
def extract_features(image_path):

    image = cv2.imread(image_path)

    #edge 0 if ig not read properly 
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")


    color_hist = extract_color_hist(image)
    hog_feats = extract_hog_features(image)

    #all in one sector
    combined = np.hstack([color_hist, hog_feats])
    return combined

#extracts features and labels form each dataset 
def extract_dataset_features(image_folder):
    X, y = [], []
    #Looping over eahc subfolder
    for label in os.listdir(image_folder):      
        label_path = os.path.join(image_folder, label)
        if not os.path.isdir(label_path):       
            continue
        for file in os.listdir(label_path):
            img_path = os.path.join(label_path, file)
            if not (file.lower().endswith(".jpg") or file.lower().endswith(".png")): 
                continue
            try:
                features = extract_features(img_path)   
                X.append(features)
                y.append(label)
            except Exception as e:
                print(f"skip{img_path}: {e}")
    return np.array(X), np.array(y)   


if __name__ == "__main__":
    train_dir = "data/lisa/train"
    val_dir   = "data/lisa/val"


    X_train, y_train = extract_dataset_features(train_dir)
    X_val, y_val = extract_dataset_features(val_dir)

    np.savez("traffic_light_features_train.npz", X=X_train, y=y_train)
    np.savez("traffic_light_features_val.npz", X=X_val, y=y_val)
