import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load model
model = tf.keras.models.load_model("disease_model.h5")

class_names = ['COVID19', 'NORMAL', 'PNEUMONIA', 'TURBERCULOSIS']

# Change this path to any test image
img_path = "data/PNEUMONIA/person3_virus_17.jpeg"

img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

predictions = model.predict(img_array)
predicted_class = class_names[np.argmax(predictions)]
confidence = np.max(predictions) * 100

print(f"Prediction: {predicted_class}")
print(f"Confidence: {confidence:.2f}%")