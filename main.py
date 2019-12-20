from flask import Flask, request, jsonify, render_template, url_for
from werkzeug.utils import secure_filename
import os
import time
import numpy as np
#import tensorflow as tf

from cv2 import cv2

import pose_detector
#import clothing_classfiers

#model_face = tf.keras.models.load_model("gender_model.h5")
#model_whole_body = tf.keras.models.load_model("model_whole.h5")
#model_upper_body = tf.keras.models.load_model("model_upper.h5")
#model_lower_body = tf.keras.models.load_model("model_lower.h5")


UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def render_home():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        if 'img' not in request.files:
            return render_template('index.html')

        file = request.files['img']

        if file.filename == '':
            return render_template('index.html')

        if file and allowed_file(file.filename):
            img_path = secure_filename(file.filename)
            #file.save(os.path.join(UPLOAD_FOLDER, img_path))
            full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_pictures', 'RomainBarraudPhoto.jpg')


            #return render_template("display_image.html",\
            #     img_input=img_input,
            #     img_pose=img_pose,
            #     img_whole=img_upper,
            #     img_upper=img_whole,
            #     img_lower=img_lower)
            return pose_detector.pose_main("./client_images/images_input/black-male.jpg")


@app.route("/recommendations", methods=["POST"])
def master_function():
    if 'img' not in request.files:
        return render_template('index.html')

    file = request.files['img']

    if file.filename == '':
        return render_template('index.html')

    if file and allowed_file(file.filename):
        img_path = secure_filename(file.filename)
        file.save(os.path.join(UPLOAD_FOLDER, img_path))

    #recs = p.get_recs('static/'+img_path)
    return pose_detector.pose_main()


@app.route('/recs', methods=['POST'])
def make_recs():

    if 'img' not in request.files:
        return render_template('index.html')

    file = request.files['img']

    if file.filename == '':
        return render_template('index.html')

    if file and allowed_file(file.filename):
        img_path = secure_filename(file.filename)
        file.save(os.path.join(UPLOAD_FOLDER, img_path))

    recs = p.get_recs('static/'+img_path)
    return render_template('show_rec_imgs.html', img_path=url_for('static', filename=img_path), images=recs[0], urls=recs[2])


if __name__ == "__main__":
    app.debug = True
    app.run(host="127.0.0.1", port=5001)


