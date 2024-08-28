import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import numpy as np

def create_dummy_model(output_size=1):
    model = Sequential([
        Flatten(input_shape=(224, 224, 3)),  # Assuming input images are 224x224x3
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(output_size, activation='sigmoid')  # Binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def save_dummy_model(model, filename):
    # Save the model
    model.save(filename)
    print(f"Model saved as {filename}")

# Create and save dummy models
model_names = [
    'conjunctivitis_model.h5',
    'dry_eye_model.h5',
    'keratoconus_model.h5',
    'cataract_model.h5',
    'stye_chalazion_model.h5',
    'blepharitis_model.h5',
    'ocular_surface_model.h5',
    'allergic_reaction_model.h5'
]

for model_name in model_names:
    dummy_model = create_dummy_model()
    save_dummy_model(dummy_model, f"models/{model_name}")

print("All dummy models created and saved.")
