# Plant-Disease-Detection-And-Classification-Using-Deep-Learning-With-EfficientNetB4
📌 Overview

This project presents a deep learning-based plant disease detection system that classifies leaf images into multiple disease categories using Convolutional Neural Networks (CNN) and transfer learning with EfficientNetB4.

The model is trained on the PlantVillage dataset (39 classes) and achieves approximately 95% accuracy, enabling accurate and efficient identification of plant diseases.

🚀 Features
🌱 Detects 39 different plant disease classes
🧠 Uses EfficientNetB4 (pretrained on ImageNet)
⚡ Two-phase training: Feature Extraction + Fine-Tuning
📊 Achieves ~95% test accuracy
🔄 Optimized data pipeline using tf.data + prefetch()
🛡️ Regularization using Dropout (0.2)
📈 Supports evaluation with Accuracy & F1-score (recommended)

Model Architecture
Input (160×160×3)
        ↓
EfficientNetB4 (Feature Extractor)
        ↓
GlobalAveragePooling2D
        ↓
Dropout (0.2)
        ↓
Dense Layer (39 Classes)

📂 Dataset
📦 Dataset: PlantVillage
🖼️ Total Images: ~54,000
🏷️ Classes: 39
🔀 Split:
Training: 80%
Validation: 10%
Test: 10%

Includes diseases like:

Tomato Late Blight
Apple Black Rot
Potato Healthy

⚙️ Technologies Used
Python 🐍
TensorFlow / Keras
EfficientNetB4
NumPy, Matplotlib
Scikit-learn
🧪 Training Strategy
🔹 Phase 1: Feature Extraction
Freeze EfficientNetB4
Train top layers
Epochs: 6
🔹 Phase 2: Fine-Tuning
Unfreeze deeper layers (after layer 100)
Train entire model
Epochs: 10

📌 Key Highlights

✔ High accuracy with minimal training time
✔ Efficient transfer learning approach
✔ Scalable for real-world agricultural applications
✔ No manual feature extraction required

⚠️ Limitations
Dataset captured in controlled conditions
Real-world performance may vary
F1-score not explicitly computed in current implementation
Activation mismatch (sigmoid instead of softmax)
🔮 Future Improvements
✅ Use softmax activation
📊 Add confusion matrix & F1-score evaluation
🌍 Train on real-world field images
🧪 Apply data augmentation
📱 Deploy as mobile/web application
