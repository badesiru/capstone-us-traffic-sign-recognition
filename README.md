# capstone-us-traffic-sign-recognition

# Traffic Light State Recognition Using Classical ML and CNN Models

# 1. Intro
Accurate traffic light recognition is essential for driver assistance systems and autonomous vehicles. Classical computer vision methods like HOG features and color histograms have historically been used for traffic-related classification but have limitations when dealing with visual variability. Modern deep learning models, specifically CNNs, offer improved robustness to lighting, occlusion, and viewpoint changes.  
This project implements and compares both approaches to determine which yields higher reliability for traffic light state recognition.

# 2. Dataset
We used the LISA Traffic Light Dataset, which contains annotated bounding boxes for traffic signals across various driving scenes.

# Preprocessing
Custom preprocessing was performed to convert the raw annotations into a model-ready dataset:
- Parsing annotation CSV files  
- Using bounding boxes to crop each traffic-light region  
- Converting crops to RGB  
- Resizing images to 32×32 (classical ML) and 64×64 (CNNs)  
- Grouping images into: `red`, `yellow`, `green`, `inactive`

# Training/Validation Split
Annotations were randomized and split at an 85/15 ratio into:
data/lisa/train/<class>/
data/lisa/val/<class>/

# Manifest Files
Two manifest CSVs were generated:
data/lisa/train_manifest.csv
data/lisa/val_manifest.csv

Each row contains:
filepath and label

# 3. Methods

# Classical Machine Learning Pipeline
Classical models were trained using hand-crafted features extracted from each cropped image.

# Feature Extraction
Each image was converted into a feature vector using:
- RGB color histograms  
- Histogram of Oriented Gradients (HOG)

The resulting training and validation feature sets were stored as:
traffic_light_features_train.npz
traffic_light_features_val.npz

# Baseline Models Evaluated
We trained and evaluated three classical ML models:
- Support Vector Machine (RBF kernel)
- Random Forest
- K-Nearest Neighbors

Each model outputs:
- Accuracy  
- Classification report  
- Confusion matrix visualization  

These provided baseline performance for comparison against the CNN.

# Convolutional Neural Network Pipeline

# CNN Architecture
The deep learning approach uses a custom CNN consisting of:
- Three convolution - ReLU - max-pool blocks  
- A fully connected classifier with dropout  
- Output logits for the three target classes  

# Training Details
- Input image size: 64×64 RGB 
- Optimizer: Adam
- Loss function: CrossEntropyLoss* 
- Epochs: 10 
- Data augmentation: color jitter, rotations, horizontal flips  

Metrics recorded per epoch include:
- Training loss  
- Validation loss  
- Validation accuracy  

These are saved to:
experiments/results/cnn_epoch_metrics.csv

A final trained model is saved as:
cnn_model.pth

# 4. Experiments

# Classical Baseline Models
The classical ML models performed well given the feature-based approach. However, they were sensitive to variations in lighting, motion blur, and the smaller number of yellow-light samples.

# CNN Experiments
The CNN demonstrated far greater robustness. Its ability to learn spatial and color features directly from the images allowed it to adapt to real-world visual noise and class imbalance more effectively.

Confusion matrices and classification reports were generated for detailed evaluation.

# 5. Results

# Classical Models
Classical ML models provided a strong baseline, with reasonable separation between red and green classes. Yellow remained more challenging due to sample imbalance and overlapping color features.

# CNN Model
The CNN consistently outperformed the classical ML models. It produced:
- Higher overall accuracy  
- Fewer misclassifications  
- Better generalization on the validation set  

The CNN confusion matrix showed clearer separation between all three classes.

# 6. Discussion
Key takeaways:
- Classical feature-based methods are simple and interpretable but struggle with lighting variation and naturally ambiguous samples.  
- Deep learning (CNN) is much more effective for traffic light recognition because it learns features automatically.  
- Augmentation improved CNN generalization to challenging examples.  
- Yellow lights remained the most difficult class, largely due to dataset imbalance and similarity to both red and green under different lighting.

# 7. Conclusion
This project presents a complete traffic-light recognition system using both classical machine learning and deep learning. While classical models performed reasonably, the CNN was significantly more reliable and adaptable to real-world scenarios.  
CNN-based methods appear more suitable for deployment in driver assistance or autonomous driving applications.

# 8. Instructions

# Create a virtual environment
python -m venv venv
venv\Scripts\activate # Windows

# Install project dependencies
pip install -r requirements.txt

# Run classical baseline models
python src/classical_baselines.py

# Train the CNN
python src/train_cnn.py