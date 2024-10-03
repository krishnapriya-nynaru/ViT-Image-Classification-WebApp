import base64
import numpy as np
import time  # Import time module
from flask import Flask, request, jsonify, render_template
from tflite_runtime.interpreter import Interpreter
from PIL import Image
import cv2
import io
import json

app = Flask(__name__)

# Load model and labels
interpreter = Interpreter(model_path='vit.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open('labels.json', 'r') as f:
    labels = json.load(f)

def preprocess_image(image_frame, target_size=(224, 224)):
    """Preprocess the input image to the required input size."""
    image = image_frame.resize(target_size)
    image_np = np.array(image).astype(np.float32)
    image_np = image_np / 255.0  # Normalize to [0,1]
    return np.expand_dims(image_np, axis=0)

def run_inference(input_data):
    """Run inference on the model."""
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Start timing the inference
    start_time = time.time()
    interpreter.invoke()
    inference_time = time.time() - start_time  # Calculate inference time
    
    output_data = [interpreter.get_tensor(output['index']) for output in output_details]
    
    return output_data, inference_time

@app.route('/')
def index():
    return render_template('index.html')  # Return the HTML frontend

@app.route('/process_frame', methods=['POST'])
def process_frame():
    # Decode the frame sent from the frontend
    frame_data = request.json['frame']
    image_data = base64.b64decode(frame_data.split(',')[1])
    image = Image.open(io.BytesIO(image_data))

    # Preprocess and run inference
    input_data = preprocess_image(image)
    output_data, inference_time = run_inference(input_data)

    # Get top prediction
    predictions = output_data[0][0]
    top_prediction_index = np.argmax(predictions)
    predicted_label = labels[top_prediction_index]
    confidence_score = predictions[top_prediction_index]

    # Get device info (could be static for tflite runtime)
    device_info = "CPU"  # You can change this based on your setup

    # Return prediction result
    return jsonify({
        'label': predicted_label,
        'confidence': float(confidence_score),
        'inference_time': inference_time,
        'device_info': device_info  # Add device info to the response
    })

if __name__ == "__main__":
    app.run(debug=True)
