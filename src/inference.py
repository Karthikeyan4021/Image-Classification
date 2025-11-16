
# src/inference.py
# Unified inference for age/gender + medical models
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model

def load_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (224,224))
    return img[..., ::-1] / 255.0

def predict_age_gender(model, img):
    g, a = model.predict(img[np.newaxis, ...])
    return float(g[0][0]), float(a[0][0])

def predict_medical(model, img, class_names):
    p = model.predict(img[np.newaxis, ...])[0]
    return class_names[int(np.argmax(p))], float(np.max(p))
