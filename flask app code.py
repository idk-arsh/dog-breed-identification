from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import requests
import logging
from flask_ngrok import run_with_ngrok

app = Flask(__name__, template_folder='/content/drive/MyDrive/dog-vision/template')
run_with_ngrok(app) # starts ngrok when the app is run

UPLOAD_FOLDER = 'upload/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

IMG_SIZE = 224
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
                         

            image = tf.io.read_file(file_path)
            image = tf.image.decode_jpeg(image,channels=3)
            image = tf.image.convert_image_dtype(image,tf.float32)
            image = tf.image.resize(image,size=[IMG_SIZE,IMG_SIZE])
            image = tf.expand_dims(image, axis=0)

            pred=model.predict(image)
            pred_labels = get_preds_labels(pred,breed_list)

            return render_template('result.html', prediction=pred_labels, image_file=os.path.join(app.config['UPLOAD_FOLDER'], filename))


    return redirect(url_for('r.html'))

def get_preds_labels(preds, blist):
    pred_labels = []
    for pred in preds:
        pred_label = blist[np.argmax(pred)]
        pred_labels.append(pred_label)
    return pred_labels

if __name__ == '__main__':
    app.logger.setLevel(logging.DEBUG)
    app.run()
