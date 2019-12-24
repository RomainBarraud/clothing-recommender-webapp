from flask import Flask, request, jsonify, render_template, url_for
from werkzeug.utils import secure_filename
import os
import time
import json
import requests
import numpy as np
import tensorflow as tf

from cv2 import cv2

import pose_detector
import clothing_classfiers

API_URL_UPPER = 'http://127.0.0.1:5011'

IMG_GENDER_HEIGHT = 150
IMG_GENDER_WIDTH = 150
IMG_HEIGHT = 256
IMG_WIDTH = 256

UPLOAD_FOLDER = './static/uploaded_pictures'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])

INVENTORY_PATH = './static/inventory'


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
            img_name = secure_filename(file.filename)
            img_name_short = img_name.rsplit('.', 1)[0]
            img_extension = img_name.rsplit('.', 1)[1]
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(img_path)

            #return render_template("display_image.html",\
            #     img_input=img_input,
            #     img_pose=img_pose,
            #     img_whole=img_upper,
            #     img_upper=img_whole,
            #     img_lower=img_lower)
            #return pose_detector.pose_main("./client_images/images_input/black-male.jpg")

            photo_analyser = pose_detector.pose_main(img_path, img_name_short)

            image_head = cv2.imread(os.path.join("./static/uploaded_pictures" , str(img_name_short) + '_Head.jpg'), cv2.IMREAD_GRAYSCALE)
            gender = clothing_classfiers.predict_gender(image_head, IMG_GENDER_HEIGHT, IMG_GENDER_WIDTH)

            image_upper = cv2.imread(os.path.join("./static/uploaded_pictures" , str(img_name_short) + '_Upper.jpg'), )
            print("image upper list: ", image_upper.tolist())
            upper_load = {'image': image_upper.tolist(), 'IMG_HEIGHT': IMG_HEIGHT, 'IMG_WIDTH': IMG_WIDTH}
            print("Will call remote instance")
            clothing_upper = requests.post(url=API_URL_UPPER, json=upper_load)
            print("clothing_upper: ", clothing_upper)
            print("insideText: ", clothing_upper.text)
            print("insideText_2: ", clothing_upper.json()['clothing'])
            #clothing_upper = clothing_classfiers.predict_upper(image_upper, IMG_HEIGHT, IMG_WIDTH)
            clothing_upper_path = os.path.join(INVENTORY_PATH, clothing_upper.json()['clothing'])

            if photo_analyser: # found lower clothing
                image_lower = cv2.imread(os.path.join("./static/uploaded_pictures" , str(img_name_short) + '_Lower.jpg'))
                clothing_lower = clothing_classfiers.predict_lower(image_lower, IMG_HEIGHT, IMG_WIDTH)
                clothing_lower_path = os.path.join(INVENTORY_PATH, clothing_lower)

            image_whole = cv2.imread(os.path.join("./static/uploaded_pictures" , str(img_name_short) + '_Whole.jpg'))
            clothing_whole = clothing_classfiers.predict_whole(image_whole, IMG_HEIGHT, IMG_WIDTH)
            clothing_whole_path = os.path.join(INVENTORY_PATH, clothing_whole)

            clothing_boxed = cv2.imread(os.path.join("./static/uploaded_pictures" , str(img_name_short) + '_Boxed.jpg'))

            return render_template('recommendations.html')


if __name__ == "__main__":
    app.debug = True
    app.run(host="127.0.0.1", port=5001)


