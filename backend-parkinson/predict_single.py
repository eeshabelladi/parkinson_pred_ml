import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load model
model = tf.keras.models.load_model('cnn_spiral_model_final.keras')

# Path to your test image
img_path = 'testing_imgs/test1.png'

# Load image as grayscale
img = image.load_img(img_path, target_size=(128, 128), color_mode='grayscale')
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize to [0,1]

# Predict
pred = model.predict(img_array)
class_names = ['healthy', 'parkinson']

# Since output is sigmoid, it's a single value
prob_parkinson = float(pred[0][0])
prob_healthy = 1 - prob_parkinson
predicted_class = class_names[1] if prob_parkinson > 0.5 else class_names[0]

print(f"✅ Predicted: {predicted_class}")
print(f"🧾 Probabilities: Healthy={prob_healthy:.4f}, Parkinson={prob_parkinson:.4f}")
