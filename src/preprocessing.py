import os
import cv2
import numpy as np

def load_image(path:str, size:tuple =(256,256), as_gray:bool = True):
    """
    Function to load the image at specific path and a
    pply the standard transformations of resizing and grayscarl conversion.
    """
    if as_gray:
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path,cv2.IMREAD_COLOR)
    
    if img is None:
        raise FileExistsError(f"Could not find the image at : {path}")
    
    img = cv2.resize(img,size,interpolation=cv2.INTER_AREA)

    return img

def preprocess_image(img, hist_eq=True):
    """
    Normalize and optionally apply histogram equalization.
    """
    if hist_eq:
        img = cv2.equalizeHist(img)
    img = img.astype(np.float32) / 255.0
    return img

def load_dataset(root_dir:str='data/KylbergDataset',limit_per_class:int=None, size:tuple=(256,256)):
    """
    Load the complete dataset and return as (X,Y,classes) and apply preprocessing if any applicable.
    """
    X, Y = [], []
    classes = []

    classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    for label, cls in enumerate(classes):
        cls_dir = os.path.join(root_dir, cls)
        files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if limit_per_class:
            files = files[:limit_per_class]
        print(f"Loading {len(files)} images from {cls}")

        for f in files:
            path = os.path.join(cls_dir, f)
            try:
                img = load_image(path, size=size, as_gray=True)
                img = preprocess_image(img)
                X.append(img)
                Y.append(label)
            except Exception as e:
                print(f" Error loading {path}: {e}")

    X = np.array(X)
    Y = np.array(Y)

    return X, Y, classes