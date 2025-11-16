
# Image Projects — Multi-task Age/Gender + Medical Classification

Includes:
- Age/Gender model (multi-task EfficientNet)
- Eye/Liver/Skin classifier skeleton
- Grad-CAM explainability
- Inference pipeline
- Starter training scripts
- Result screenshots

## How to Run
1. Install dependencies:
   pip install tensorflow opencv-python matplotlib numpy pandas

2. Train:
   python src/train_age_gender.py
   python src/train_medical.py

3. Inference:
   python src/inference.py

4. Explainability (Grad-CAM):
   python src/explainability.py
