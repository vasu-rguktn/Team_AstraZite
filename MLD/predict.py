import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load trained model
model = tf.keras.models.load_model("microplastic_multiclass_model.h5")

# Class labels (order from training)
class_names = ['algae', 'filament', 'fragment', 'pellet']

# Load test image
img_path = "test.jpg"   # put any image here
img = image.load_img(img_path, target_size=(224, 224))
img = image.img_to_array(img) / 255.0
img = np.expand_dims(img, axis=0)

# Predict
pred = model.predict(img)
predicted_class = class_names[np.argmax(pred)]

print("Predicted class:", predicted_class)
