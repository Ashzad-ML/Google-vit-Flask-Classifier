from flask import Flask, request, render_template, send_file, send_from_directory
import tensorflow as tf
import numpy as np
import cv2
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

model_path = 'static/tf_model'
model = tf.saved_model.load(model_path)

classes = ["Pattern", "Solid"]

def preprocess_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = tf.convert_to_tensor(image_rgb, dtype=tf.float32)
    image_resized = tf.image.resize(image_tensor, [224, 224])
    image_normalized = image_resized / 255.0
    image_transposed = tf.transpose(image_normalized, perm=[2, 0, 1])
    image_batched = image_transposed[tf.newaxis, :]
    return image_batched

def predict_image(image):
    inputs = {'pixel_values': preprocess_image(image)}
    outputs = model(inputs)
    class_scores = outputs['logits'].numpy().flatten()
    softmax_scores = tf.nn.softmax(class_scores).numpy()
    percentages = softmax_scores * 100
    formatted_percentages = [f"{p:.2f}%" for p in percentages]
    return formatted_percentages, classes[np.argmax(percentages)]

@app.route('/', methods=['GET', 'POST'])
def index():
    file_path = None
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join('uploads', filename) 
            file.save(file_path)

            # Load the image and predict
            image = cv2.imread(file_path)
            percentages, result = predict_image(image)

            return render_template('index.html', percentages=percentages, result=result, file_name=file.filename)

    return render_template('index.html', percentages=None, result=None)


@app.route('/uploads/<file_name>')
def serve_file(file_name):
    path_to_file = os.path.join("uploads", file_name)
    if os.path.exists(path_to_file):
        return send_file(path_to_file)
    return None



@app.route('/js-app')
def js_app():
    return render_template('index_js.html')

# @app.route('/js-app/<path:path>')
# def js_app_files(path):
#     return send_from_directory('static/js_app', path)


if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    app.run(debug=True,port=3000)
