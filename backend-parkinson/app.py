from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import tempfile

app = Flask(__name__)
CORS(app)  # allow React frontend access

# Load your trained model once at startup
MODEL_PATH = "cnn_spiral_model_final.keras"
model = tf.keras.models.load_model(MODEL_PATH)
class_names = ['healthy', 'parkinson']

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'handwriting' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['handwriting']
    
    # Save to a temporary file
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, file.filename)
    file.save(file_path)
    
    try:
        # Load as grayscale, preprocess, and predict
        img = image.load_img(file_path, target_size=(128, 128), color_mode='grayscale')
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        pred = model.predict(img_array)
        prob_parkinson = float(pred[0][0])
        prob_healthy = 1 - prob_parkinson
        predicted_class = class_names[1] if prob_parkinson > 0.5 else class_names[0]
        confidence = round(max(prob_parkinson, prob_healthy) * 100, 2)
        
        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        os.remove(file_path)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
