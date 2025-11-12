# Kylberg - Texture Classification using Image Processing techniques

This project is an academic work for Bennett University under course CSET-344 (Image and Video Processing), this project aims to implement all the topics taught in the course in action for a task.

## Topics Covered (as per Syllabus)

### **Module 1 – Spatial Domain Processing**
| Topic | Description | Implemented In |
|-------|--------------|----------------|
| Sampling & Quantization | Image loading, resizing, normalization | `preprocessing.py` |
| Gray to Binary Conversion | Thresholding for binary masks | `filtering.py` |
| Gray-level Transformations | Intensity normalization | `preprocessing.py` |
| Histogram Processing | Conceptual intensity balancing | `filtering.py` |
| Spatial Filters | Mean, Median, Gaussian smoothing | `filtering.py` |
| Edge Detection | Sobel, Prewitt, Laplacian, LoG filters | `filtering.py` |
| Convolution Concept | Used in all spatial filters | `filtering.py` |

---

### **Module 2 – Color, Morphology & Texture**
| Topic | Description | Implemented In |
|-------|--------------|----------------|
| Canny Edge Detection | Texture edge extraction | `features.py` |
| Harris Corner Detector | Corner-based texture features | `features.py` |
| Morphological Operations | Dilation, Erosion, Opening, Closing | `features.py` |
| Boundary Detection | Morphological gradient method | `features.py` |
| Hole Filling | Flood-fill based hole filling | `features.py` |
| Texture Analysis | GLCM-based texture feature extraction | `features.py`, `ml_features.py` |
| Shape Representation | Region-based descriptors | `features.py` |

---

### **Module 3 – Frequency Domain & Compression**
| Topic | Description | Implemented In |
|-------|--------------|----------------|
| Fourier Transform (FFT) | Frequency domain representation | `frequency.py` |
| Inverse Fourier Transform | Reconstruction from frequency | `frequency.py` |
| Frequency Filters | Ideal, Gaussian, Butterworth (LPF & HPF) | `frequency.py` |
| Discrete Cosine Transform (DCT) | Block-based image compression | `compression.py` |
| ZigZag Scanning | Coefficient ordering for compression | `compression.py` |
| PCA for Feature Reduction | Dimensionality reduction before SVM | `train_model.py` |

---

### **Module 4 – Machine Learning & Evaluation**
| Topic | Description | Implemented In |
|-------|--------------|----------------|
| Feature Extraction | Spatial + frequency + GLCM | `ml_features.py` |
| SVM Classification | Trained on extracted texture features | `train_model.py` |
| PCA Integration | Reduces feature dimensionality | `train_model.py` |
| Model Evaluation | Accuracy, confusion matrix, report | `test_model.py` |
| Result Visualization | Confusion matrix heatmap | `test_model.py` |

#### Dataset: https://www.kaggle.com/datasets/phattanunthabarsa/kylberg-texture-dataset