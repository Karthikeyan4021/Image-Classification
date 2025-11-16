
# src/train_age_gender.py
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, Model

def build_model():
    base = EfficientNetB0(include_top=False, input_shape=(224,224,3), pooling='avg')
    x = base.output
    gender_out = layers.Dense(1, activation='sigmoid', name='gender')(x)
    age_out = layers.Dense(1, activation='linear', name='age')(x)
    model = Model(inputs=base.input, outputs=[gender_out, age_out])
    model.compile(
        optimizer='adam',
        loss={'gender':'binary_crossentropy', 'age':'mse'},
        metrics={'gender':'accuracy', 'age':'mae'}
    )
    return model

if __name__ == "__main__":
    print("Training skeleton — add dataset loading & training loops.")
